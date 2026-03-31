import numpy as np
import sysOps
import os
import shutil
import re
import pandas as pd
import shlex 
import textwrap
import optimOps

def get_alignment_length(cigar):
    length = 0
    numbers = re.findall('\d+', cigar)  # find all numbers in the cigar string
    letters = re.findall('\D', cigar)   # find all non-numbers (letters) in the cigar string
    for i, letter in enumerate(letters):
        if letter in ['M', 'D', 'N', '=', 'X']:  # these operators all consume the reference
            length += int(numbers[i])
    return length

def get_internal_mismatches(cigar, md, start):
    mismatches = []
    indels = 0
    alignment_length = 0
    current_pos = int(start)
    aln_len = get_alignment_length(cigar)
    # Process MD field to detect mismatches and their positions
    for item in re.findall(r'(\d+|\^[ACGTN]+|[ACGTN]+)', md):
        if item.isdigit():
            # It's a match, move the position
            current_pos += int(item)
        elif item.startswith('^'):
            # It's a deletion, increment indels
            indels += len(item) - 1
            current_pos += len(item) - 1
        else:
            # It's a mismatch, record each mismatch in the string
            for base in item:
                mismatches.append(str(current_pos) + "~" + str(start) + "~" + str(aln_len) + ">" + base)
                current_pos += 1
    
    # Create mutation string
    mutation_string = '+'.join(mismatches) if mismatches else "None"

    return mutation_string

def process_gene_info(row,omit_splice_status=True):
        
    gene_name_index = 5
    biotype_index = 6
    transcript_id_index = 7
    
    # Return None if any necessary information is missing
    if pd.isna(row[gene_name_index]) or pd.isna(row[biotype_index]) or pd.isna(row[transcript_id_index]):
        return None
    
    biotypes = str(row[biotype_index]).split('|')
    transcript_ids = str(row[transcript_id_index]).split('|')
    gene_names = str(row[gene_name_index]).split('|')

    # Check for Mt_rRNA or rRNA in biotypes
    if "Mt_rRNA" in biotypes:
        gene_names = ["Mt_rRNA"]
    elif "rRNA" in biotypes:
        gene_names = ["rRNA"]

    # Determine splicing status
    is_spliced = any(tid not in ['NA', 'None','NaN','nan', 'NONE'] for tid in transcript_ids)
    
    if omit_splice_status:
        return gene_names
    
    return [(gene_name, 'spliced' if is_spliced else 'unspliced') for gene_name in gene_names]

def get_gene_stats(filenames):
    """
    Memory-safe gene stats.

    The old implementation loaded the full sorted_umi_seq_assignments*.txt files
    into a pandas DataFrame (pd.read_csv + pd.concat) and then built an in-memory
    dict of gene lists. On large libraries this can exceed available RAM and OOM.

    This implementation streams the input files line-by-line and accumulates
    the same weighted gene support statistics without ever holding the full
    table in memory.
    """
    from collections import defaultdict

    # Match pandas.read_csv(..., keep_default_na=True) NA parsing as closely as possible,
    # because the old code relied on pandas to convert e.g. 'NA'/'None'/'' -> NaN.
    try:
        from pandas._libs.parsers import STR_NA_VALUES  # type: ignore
        _na_strings = set(STR_NA_VALUES)
    except Exception:
        _na_strings = {
            "", "NA", "NaN", "nan", "N/A", "n/a", "#N/A", "#NA",
            "NULL", "null", "None",
            "-NaN", "-nan", "1.#IND", "1.#QNAN", "-1.#IND", "-1.#QNAN",
            "#N/A N/A", "<NA>",
        }

    def _is_missing(v) -> bool:
        if v is None:
            return True
        return str(v) in _na_strings

    # These indices match process_gene_info() expectations for sorted_umi_seq_assignments*.txt
    gene_name_index = 5
    biotype_index = 6
    transcript_id_index = 7

    element_weights = defaultdict(float)

    for filename in filenames:
        fpath = sysOps.globaldatapath + filename
        if not sysOps.check_file_exists(filename):
            continue

        with open(fpath, "r") as fh:
            for line in fh:
                if not line:
                    continue

                # Only parse up to transcript_id_index; avoid splitting the trailing long sequence column.
                fields = line.rstrip("\n").split(",", transcript_id_index + 1)
                if len(fields) <= transcript_id_index:
                    continue

                gene_name_val = fields[gene_name_index]
                biotype_val = fields[biotype_index]
                transcript_id_val = fields[transcript_id_index]

                if _is_missing(gene_name_val) or _is_missing(biotype_val) or _is_missing(transcript_id_val):
                    continue

                biotypes = str(biotype_val).split('|')
                gene_names = str(gene_name_val).split('|')

                # Preserve original collapse behaviour for rRNA categories
                if "Mt_rRNA" in biotypes:
                    gene_names = ["Mt_rRNA"]
                elif "rRNA" in biotypes:
                    gene_names = ["rRNA"]

                if not gene_names:
                    continue

                unique_elements = set(gene_names)
                weight = 1.0 / len(unique_elements)
                for element in unique_elements:
                    element_weights[element] += weight

    return [str(sum(weight >= min_umi for weight in element_weights.values())) for min_umi in [1, 5, 10]]



def escape_awk_var_for_shell(s_val):
    """Quotes a value to be passed via awk -v var=value for shell safety."""
    return shlex.quote(str(s_val))

def escape_for_awk_printf_csv_field(s_val_raw):
    """Prepares a string to be printed as a CSV field by AWK's printf, quoting it and escaping internal quotes."""
    s_val = str(s_val_raw)
    s_val = s_val.replace('"', '""')  # CSV standard: " -> ""
    return f'"{s_val}"' # Enclose in double quotes


# --- Main Function ---
def get_amp_consensus(seq_terminate_list, 
                      filter_umi0_amp_len, filter_umi1_amp_len, 
                      filter_umi0_quickmatch, filter_umi1_quickmatch, 
                      STARindexdir=None, gtffile=None, 
                      uei_matchfilepath=None, add_sequences_to_labelfiles=False):
    """
    Generates amplicon consensus sequences, performs quick matching, aligns, annotates.
    Includes k-mer based sub-clustering and robust intermediate file handling.
    """
    match_str_list = [filter_umi0_quickmatch, filter_umi1_quickmatch]
    amp_len_list = [filter_umi0_amp_len, filter_umi1_amp_len]
    gd = sysOps.globaldatapath
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    ephemeral_temp_files_overall = [] 

    common_tmp_dir_for_sort = os.path.join(gd, "tmp")
    if not os.path.exists(common_tmp_dir_for_sort):
        os.makedirs(common_tmp_dir_for_sort, exist_ok=True)
    sort_temp_dir_option = f"-T {shlex.quote(common_tmp_dir_for_sort)}"
    
    # Main try block for the entire function to ensure finally clause executes
    if True: 
        for amp_ind in range(2):
            sysOps.throw_status(f"--- Processing amp_ind: {amp_ind} ---")
            # --- Define all file paths for this amp_ind iteration ---
            amp_file_path = gd + f'amp{amp_ind}.txt'
            line_sorted_clust_uxi_path = gd + f"line_sorted_clust_uxi{amp_ind}.txt"
            sorted_assignments_file_path = gd + f"sorted_umi_seq_assignments{amp_ind}.txt"
            
            consensus_fasta_file_path = gd + f'amp{amp_ind}_seqcons_trimmed.fasta'
            amp_trimmed_txt_out_file = gd + f"amp{amp_ind}_seqcons_trimmed.txt"
            trimmed_fragments_file_path = gd + f"amp{amp_ind}_seqcons_trimmed_fragments.txt"

            tmp_umi_amp_path = gd + f"tmp_umi_amp_{amp_ind}.txt"
            tmp_umi_amp_k_path = gd + f"tmp_umi_amp_k_{amp_ind}.txt"
            tmp_sorted_umi_amp_k_path   = gd + f"tmp_sorted_umi_amp_k_{amp_ind}.txt"
            tmp_unsorted_umi_amp_k_path = gd + f"tmp_unsorted_umi_amp_k_{amp_ind}.txt"

            seqconsensus_out_file = gd + f"amp{amp_ind}_seqconsensus.txt" # UMI_idx, TotalReads, "cons1:reads1;..."
            subclusters_raw_out_file = gd + f"amp{amp_ind}_subclusters.txt" # full_query_name, raw_sub_consensus
            consensus_awk_script_path = gd + f"tmp_consensus_script_{amp_ind}.awk"
            
            label_pt_file = gd + f"label_pt{amp_ind}.txt"
            quick_match_awk_script_path = gd + f"tmp_quick_match_{amp_ind}.awk"

            star_output_dir = gd + f"STARalignment{amp_ind}/"
            star_sam_file = star_output_dir + "Aligned.out.sam"
            processed_alignments_txt = star_output_dir + "Aligned.out.sam.txt"
            tmp_sorted_aligned_sam_path = star_output_dir + "tmp_sorted_sam.txt"
            sorted_aligned_sam_path_intermediate = star_output_dir + "sorted_sam.txt"
            final_sorted_alignments_for_merge = star_output_dir + "resorted_sam.txt"
            
            gtf_awk_script_path = gd + f"tmp_gtf_parser_{amp_ind}.awk"
            gtf_processed_file = star_output_dir + "gtf.txt"
            gtf_sorted_file = star_output_dir + "sorted_gtf.txt"
            merged_algn_gtf_file = star_output_dir + "sorted_aligned_gtf.txt"
            
            tmp_umi_assignments_file = gd + f"tmp_umi_assignments_{amp_ind}.txt"
            tmp_sorted_umi_assignments_file = gd + f"tmp_sorted_umi_assignments_{amp_ind}.txt"
            consolidated_umi_assignments_file = gd + f"umi_assignments_{amp_ind}.txt" # Output of consolidation
            
            seqindex_sorted_uxi_path = gd + f"seqindex_sorted_uxi_{amp_ind}.txt"
            tmp_seq_sort_clust_path = gd + f"tmp_seq_sort_clust_uxi_{amp_ind}.txt"
            clust_sort_clust_path = gd + f"clust_sort_clust_uxi_{amp_ind}.txt"
            umi_seq_assignments_file_temp = gd + f"umi_seq_assignments_temp_{amp_ind}.txt" 
            final_umi_seq_assignments_path = gd + f"umi_seq_assignments_{amp_ind}.txt" 
            
            tmp_assign_qname_path = gd + f"tmp_assign_qname_{amp_ind}.txt"
            tmp_frags_qname_path = gd + f"tmp_frags_qname_{amp_ind}.txt"

            # Add AWK script paths to overall cleanup list
            ephemeral_temp_files_overall.extend([consensus_awk_script_path, quick_match_awk_script_path, gtf_awk_script_path])

            if not (sysOps.check_file_exists(amp_file_path) and sysOps.check_file_exists(line_sorted_clust_uxi_path)):
                sysOps.throw_status(f"Skipping amp_ind {amp_ind}: Input files missing {amp_file_path}, {line_sorted_clust_uxi_path}")
                continue
            if sysOps.check_file_exists(sorted_assignments_file_path):
                sysOps.throw_status(f"Skipping amp_ind {amp_ind}: Final output {sorted_assignments_file_path} already exists.")
                continue
            
            # --- STAGE 1: Generate Consensus Sequences ---
            stage1_outputs_exist = (sysOps.check_file_exists(consensus_fasta_file_path) and
                                    sysOps.check_file_exists(amp_trimmed_txt_out_file) and
                                    sysOps.check_file_exists(trimmed_fragments_file_path))
            if not stage1_outputs_exist:
                sysOps.throw_status(f"STAGE 1: Generating consensus for amp{amp_ind}")
                
                # Clean potential stale files for this stage before creating them
                stage1_files_to_clean_before_run = [tmp_umi_amp_path, tmp_umi_amp_k_path, tmp_sorted_umi_amp_k_path,
                                   seqconsensus_out_file, subclusters_raw_out_file,
                                   amp_trimmed_txt_out_file, trimmed_fragments_file_path, consensus_fasta_file_path]
                for f_path in stage1_files_to_clean_before_run:
                    if sysOps.check_file_exists(f_path): os.remove(f_path)
                
                maxtally_file_path = gd + f"amp{amp_ind}_maxtally.txt" # No longer generated
                if sysOps.check_file_exists(maxtally_file_path): os.remove(maxtally_file_path)

                # Build: src_line,UMI_idx,kmer,seq  (sorted by UMI,kmer) with consistent C collation
                # (1) join|awk → unsorted tmp; (2) big_sort for speed/parallelism
                sysOps.sh(
                    f"join -t ',' -1 1 -2 1 -o 1.1,1.2,2.4 "
                    f"{shlex.quote(line_sorted_clust_uxi_path)} {shlex.quote(amp_file_path)} | "
                    "awk -F',' 'BEGIN{OFS=\",\"}{k=substr($3,1,10); print $1,$2,k,$3}' "
                    f"> {shlex.quote(tmp_unsorted_umi_amp_k_path)}"
                )
                sysOps.big_sort(f" {sort_temp_dir_option} -t',' -k2,2 -k3,3 ",
                                tmp_unsorted_umi_amp_k_path, tmp_sorted_umi_amp_k_path)

                ephemeral_temp_files_overall.append(tmp_sorted_umi_amp_k_path)
                try: os.remove(tmp_unsorted_umi_amp_k_path)
                except: pass
                sysOps.throw_status(f'Writing k-mer sub-cluster consensus sequences via AWK for amp{amp_ind}')
                
                cmd_awk_consensus = (
                    f"awk -f " + os.path.join(script_dir, "sub_consensus.awk") + " " 
                    f"{shlex.quote(tmp_sorted_umi_amp_k_path)} "
                    f"{shlex.quote(seqconsensus_out_file)} "
                    f"{shlex.quote(subclusters_raw_out_file)}"
                )
                sysOps.sh(cmd_awk_consensus)

                if sysOps.check_file_exists(tmp_sorted_umi_amp_k_path) and os.path.exists(tmp_sorted_umi_amp_k_path):
                    os.remove(tmp_sorted_umi_amp_k_path)

                sysOps.throw_status(f"Trimming sub-consensuses and generating FASTA/Fragments for amp{amp_ind}")
                
                with open(subclusters_raw_out_file, 'r') as f_subclusters_in, \
                     open(trimmed_fragments_file_path, 'w') as f_trimmed_fragments_out, \
                     open(consensus_fasta_file_path, 'w') as f_final_fasta_out:
                    for line in f_subclusters_in:
                        parts = line.strip().split(",", 1)
                        if len(parts) != 2: continue
                        full_query_name, raw_sub_consensus_seq = parts
                        current_umi_idx = full_query_name.split('.')[0]
                        trimmed_seq = raw_sub_consensus_seq
                        min_term_idx = len(trimmed_seq)
                        if seq_terminate_list:
                            for term_str in seq_terminate_list:
                                term_pos = trimmed_seq.find(term_str)
                                if term_pos != -1 and term_pos < min_term_idx: min_term_idx = term_pos
                        trimmed_seq = trimmed_seq[:min_term_idx]
                        if amp_len_list[amp_ind] is not None and len(trimmed_seq) < amp_len_list[amp_ind]:
                            trimmed_seq = "N"
                        if trimmed_seq != "N": 
                            f_final_fasta_out.write(f">{full_query_name}:N:N\n{trimmed_seq}\n")
                            f_trimmed_fragments_out.write(f"{full_query_name},{trimmed_seq}\n")

                # We already wrote all per-subcluster fragments to:
                #   trimmed_fragments_file_path:  FQN,trimmed_seq
                # We also have per-UMI totals in:
                #   seqconsensus_out_file:        UMI,total_reads,cons1:reads1;...

                # (a) Filter out trimmed_seq == "N", then group by UMI and join fragments with ';'
                tmp_valid_frags = gd + f"tmp_valid_frags_{amp_ind}.txt"
                tmp_sorted_frags = gd + f"tmp_sorted_frags_{amp_ind}.txt"
                tmp_agg_frags    = gd + f"tmp_agg_frags_{amp_ind}.txt"
                tmp_seqcons_sort = gd + f"tmp_seqcons_sort_{amp_ind}.txt"

                sysOps.sh(
                    f"awk -F',' 'BEGIN{{OFS=\",\"}} $2!=\"N\"{{"
                    " split($1,a,\".\"); umi=a[1]; print umi,$2 "
                    f"}}' {shlex.quote(trimmed_fragments_file_path)} > {shlex.quote(tmp_valid_frags)}"
                )

                # sort fragments by UMI
                sysOps.big_sort(f" {sort_temp_dir_option} -t ',' -k1,1 ", tmp_valid_frags, tmp_sorted_frags)

                # aggregate per UMI -> UMI,seq1;seq2;...
                sysOps.sh(
                    "awk -F',' 'BEGIN{OFS=\",\"} "
                    "{ if($1!=p){ if(p!=\"\") print p,buf; p=$1; buf=$2 }"
                    "  else      { buf = buf \";\" $2 } } "
                    "END{ if(p!=\"\") print p,buf }' "
                    + shlex.quote(tmp_sorted_frags)
                    + " > " + shlex.quote(tmp_agg_frags)
                )
                # sort seqcons by UMI for the join (stage to tmp then big_sort)
                _tmp_seqcons_unsorted = gd + f"tmp_seqcons_unsorted_{amp_ind}.txt"
                sysOps.sh(
                    "awk -F',' 'BEGIN{OFS=\",\"} {print $1,$2}' "
                    + shlex.quote(seqconsensus_out_file)
                    + " > " + shlex.quote(_tmp_seqcons_unsorted)
                )
                sysOps.big_sort(" -t',' -k1,1 ", _tmp_seqcons_unsorted, tmp_seqcons_sort)
                try: os.remove(_tmp_seqcons_unsorted)
                except: pass

                # (b) Join totals to aggregated fragments -> amp{amp_ind}_seqcons_trimmed.txt
                # output: UMI,total_reads,seq1;seq2;...
                # Preserve UMIs even if all trimmed fragments were 'N' (empty on the right)
                sysOps.sh(
                    "join -t ',' -1 1 -2 1 -a 1 -e 'N' "
                    "-o 1.1,1.2,2.2 "
                    + shlex.quote(tmp_seqcons_sort) + " "
                    + shlex.quote(tmp_agg_frags)
                    + " > " + shlex.quote(amp_trimmed_txt_out_file)
                )

                # Guard against a corner case where seqconsensus_out_file itself is empty
                if os.path.exists(amp_trimmed_txt_out_file) and os.path.getsize(amp_trimmed_txt_out_file) == 0:
                    try:
                        os.remove(amp_trimmed_txt_out_file)
                    except OSError:
                        pass

                # (c) Update cluster_counts_overall directly from the joined file
                cluster_counts_overall = [0,0,0]
                with open(amp_trimmed_txt_out_file) as _f:
                    for _ln in _f:
                        try:
                            _parts = _ln.split(',',2)
                            if len(_parts) < 3 or _parts[2].strip() == 'N':
                                continue
                            _tot = int(_parts[1])
                            if _tot == 1: cluster_counts_overall[0] += 1
                            elif _tot == 2: cluster_counts_overall[1] += 1
                            elif _tot >= 3: cluster_counts_overall[2] += 1
                        except:
                            pass

                # cleanup temps
                for _p in [tmp_valid_frags, tmp_sorted_frags, tmp_agg_frags, tmp_seqcons_sort]:
                    try: os.remove(_p)
                    except: pass

                stats_file_to_append = gd + "pairing_stats.txt" if sysOps.check_file_exists("pairing_stats.txt") else gd + "umi_stats.txt"
                with open(stats_file_to_append, 'a') as statsfile:
                    statsfile.write(f"{amp_ind}amp:{','.join(map(str, cluster_counts_overall))}\n")
                if sysOps.check_file_exists(subclusters_raw_out_file) and os.path.exists(subclusters_raw_out_file):
                    os.remove(subclusters_raw_out_file)
            else:
                sysOps.throw_status(f"STAGE 1: Skipped for amp{amp_ind}, key output files exist.")

            # --- STAGE 2: Quick Matching ---
            sysOps.throw_status(f"STAGE 2: Quick matching for amp{amp_ind}")
            if sysOps.check_file_exists(amp_trimmed_txt_out_file):
                if not sysOps.check_file_exists(label_pt_file):
                    if sysOps.check_file_exists(label_pt_file) and os.path.exists(label_pt_file): os.remove(label_pt_file)
                    current_match_file_path_setting = match_str_list[amp_ind]
                    if current_match_file_path_setting:
                        current_match_file_full_path = gd + current_match_file_path_setting
                        if sysOps.check_file_exists(current_match_file_full_path):
                            sysOps.throw_status(f'Performing quick-match using {current_match_file_full_path}')
                            
                            # ---------- build quick-match awk programme ----------
                            fmt_base      = "%s,%s,%d" + (",%s" if add_sequences_to_labelfiles else "") + "\\n"
                            printf_arg_3  = ",$3" if add_sequences_to_labelfiles else ""
                            patterns_for_awk_vars = []
                            awk_pattern_rules = []
                            with open(current_match_file_full_path) as mf:
                                for idx, raw in enumerate(mf):
                                    pattern = raw.strip()
                                    if not pattern:
                                        continue
                                    m_idx = next((i for i, c in enumerate(pattern) if c != "N"), -1)
                                    if m_idx == -1:
                                        continue
                                    awk_var = f"p{idx}"
                                    patterns_for_awk_vars.append((awk_var, pattern[m_idx:]))
                                    awk_pattern_rules.append(
                                        f"""n=split($3,subs,";");
                                           for(i=1;i<=n;i++) if(length(subs[i])>={m_idx+len(pattern[m_idx:])} &&
                                                substr(subs[i],{m_idx+1},{len(pattern[m_idx:])})=={awk_var}){{
                                                    printf("{fmt_base}",$1,$2,{idx}{printf_arg_3});
                                                    printed=1; next;
                                           }}"""
                                    )
                            
                            quick_match_awk = f"""BEGIN{{FS=",";OFS=","}}{{printed=0;{chr(10).join(awk_pattern_rules)}if(!printed) printf("{fmt_base}",$1,$2,-1{printf_arg_3});}}"""
                            with open(quick_match_awk_script_path,"w") as fh: fh.write(quick_match_awk)
                            vars = " ".join(f"-v {v}={escape_awk_var_for_shell(val)}" for v,val in patterns_for_awk_vars)
                            sysOps.sh(f"awk {vars} -f {shlex.quote(quick_match_awk_script_path)} "
                                      f"{shlex.quote(amp_trimmed_txt_out_file)} > {shlex.quote(label_pt_file)}")
                        else: 
                            sysOps.throw_status(f"Warning: Quick match file not found: {current_match_file_full_path}.")
                    
                    if not sysOps.check_file_exists(label_pt_file): 
                        fmt_default = "%s,%s,-1" + (",%s" if add_sequences_to_labelfiles else "") + "\\n"
                        default_args = ",$3" if add_sequences_to_labelfiles else ""
                        sysOps.sh(
                            f"""awk -F',' 'BEGIN{{OFS=","}}
                               {{printf("{fmt_default}",$1,$2{default_args})}}' \
                               {shlex.quote(amp_trimmed_txt_out_file)} > {shlex.quote(label_pt_file)}"""
                        )
            else:
                sysOps.throw_status(f"STAGE 2: Skipped quick-match for amp{amp_ind} (no trimmed sub-consensus).")

            # --- STAGE 3: STAR Alignment and Annotation ---
            if STARindexdir and gtffile:
                sysOps.throw_status(f"STAGE 3: STAR Alignment for amp{amp_ind}")
                if not os.path.exists(star_output_dir): os.makedirs(star_output_dir, exist_ok=True)
                
                if not sysOps.check_file_exists(star_sam_file):
                    cmd_star = (f"STAR --genomeDir {shlex.quote(gd + STARindexdir)} --runThreadN 2 "
                                f"--readFilesIn {shlex.quote(consensus_fasta_file_path)} "
                                f"--outFileNamePrefix {shlex.quote(star_output_dir)} --outSAMtype SAM "
                                f"--outSAMunmapped Within --outSAMattributes NH HI AS nM NM MD")
                    sysOps.sh(cmd_star)
                
                if not sysOps.check_file_exists(final_sorted_alignments_for_merge):
                    filter_sam_awk_path = os.path.join(script_dir, "filter_sam.awk") 
                    if not os.path.exists(filter_sam_awk_path): raise FileNotFoundError(f"AWK script not found: {filter_sam_awk_path}")
                    sysOps.sh(f"awk -f {shlex.quote(filter_sam_awk_path)} {shlex.quote(star_sam_file)} > {shlex.quote(processed_alignments_txt)}")
                    sysOps.big_sort(f" {sort_temp_dir_option} -t',' -k3,3 -k7rn,7 -k5rn,5 ", processed_alignments_txt, tmp_sorted_aligned_sam_path)
                    sysOps.sh(f"awk -F, 'BEGIN{{prev_query=-1; prev_score=-1;}} {{if($3 != prev_query || prev_score==$7){{print ; prev_query=$3; prev_score=$7;}}}}' {shlex.quote(tmp_sorted_aligned_sam_path)} > {shlex.quote(sorted_aligned_sam_path_intermediate)}")
                    sysOps.big_sort(f" {sort_temp_dir_option} -t',' -k2,2 -k4n,4 ", sorted_aligned_sam_path_intermediate, final_sorted_alignments_for_merge)
                    for f_s3_sam_proc in [processed_alignments_txt, tmp_sorted_aligned_sam_path, sorted_aligned_sam_path_intermediate]:
                        if sysOps.check_file_exists(f_s3_sam_proc) and os.path.exists(f_s3_sam_proc): os.remove(f_s3_sam_proc)

                if not sysOps.check_file_exists(gtf_sorted_file):
                    gtf_parser_content = textwrap.dedent(f"""\
                    BEGIN{{OFS=","}}
                    {{
                        gsub(/\\\"|;/,"", $0); 
                        src_contig_col = $1; feature_col=$3; start_pos_col=$4; end_pos_col=$5; 
                        gene_biotype_val="NONE"; gene_name_val="NONE"; transcript_name_val="NONE"; 
                        gene_id_val="NONE"; transcript_id_val="NONE"; transcript_biotype_val="NONE"; 
                        for(idx=1;idx<=NF-1;idx++){{
                            if($idx == "gene_biotype"){{gene_biotype_val=$(idx+1);}} 
                            else if($idx == "gene_name"){{gene_name_val=$(idx+1);}} 
                            else if($idx == "transcript_id"){{transcript_id_val=$(idx+1);}}
                            else if($idx == "transcript_name"){{transcript_name_val=$(idx+1);}} 
                            else if($idx == "transcript_biotype"){{transcript_biotype_val=$(idx+1);}} 
                            else if($idx == "gene_id"){{gene_id_val=$(idx+1);}} 
                        }} 
                        len_diff = start_pos_col - end_pos_col; if (len_diff < 0) len_diff = -len_diff;
                        if ( len_diff <= 100000 ) {{
                            print NR,src_contig_col,feature_col,start_pos_col,end_pos_col,gene_name_val,gene_biotype_val,(transcript_id_val!="NONE"?transcript_id_val:gene_id_val),transcript_name_val,transcript_biotype_val; 
                            print NR,src_contig_col,feature_col,end_pos_col,start_pos_col,gene_name_val,gene_biotype_val,(transcript_id_val!="NONE"?transcript_id_val:gene_id_val),transcript_name_val,transcript_biotype_val;
                        }}
                    }}""")
                    with open(gtf_awk_script_path, 'w') as f_awk_gtf: f_awk_gtf.write(gtf_parser_content)
                    sysOps.sh(f"awk -f {shlex.quote(gtf_awk_script_path)} {shlex.quote(gd + gtffile)} > {shlex.quote(gtf_processed_file)}")
                    sysOps.big_sort(f" {sort_temp_dir_option} -t',' -k2,2 -k4n,4 ", gtf_processed_file, gtf_sorted_file)
                    if sysOps.check_file_exists(gtf_processed_file) and os.path.exists(gtf_processed_file): os.remove(gtf_processed_file)
                
                if not sysOps.check_file_exists(merged_algn_gtf_file):
                    cmd_merge_sort = (f"sort -k2,2 -k4n,4 -t ',' -m -T {shlex.quote(common_tmp_dir_for_sort)} "
                                      f"{shlex.quote(final_sorted_alignments_for_merge)} {shlex.quote(gtf_sorted_file)} > {shlex.quote(merged_algn_gtf_file)}")
                    sysOps.sh(cmd_merge_sort)
                
                sysOps.throw_status(f"Annotating alignments for amp{amp_ind}")
                if sysOps.check_file_exists(tmp_umi_assignments_file) and os.path.exists(tmp_umi_assignments_file): os.remove(tmp_umi_assignments_file)

                # Write raw annotations first, then de-duplicate & filter to tmp_umi_assignments_file
                tmp_umi_assignments_file_raw = os.path.join(star_output_dir, "Aligned.out.sam.annot_raw.txt")
                if sysOps.check_file_exists(tmp_umi_assignments_file_raw) and os.path.exists(tmp_umi_assignments_file_raw):
                    os.remove(tmp_umi_assignments_file_raw)

                # For near-duplicate detection: (UMI, contig) -> FQN -> stats
                # stats: {min_start, reads, score_sum, score_n}
                dup_buckets = {}

                current_overlapping_gtf_data_annot = [] 
                current_gtf_nrs_active_annot = [] 

                with open(tmp_umi_assignments_file_raw, 'w') as f_umi_assign_out, \
                    open(merged_algn_gtf_file, 'r') as f_merged_in:

                    for line_content in f_merged_in:
                        fields = line_content.strip().split(',')
                        if not fields or not fields[0]: continue
                        if fields[0] == "UMI": 
                            if len(fields) < 8:
                                sysOps.throw_status(f"Warning: Malformed alignment line (exp 8+ fields): {line_content.strip()}"); 
                                continue

                            sam_src_contig_annot, sam_query_name_full_annot, sam_start_pos_annot = fields[1], fields[2], fields[3]
                            sam_cigar_annot = fields[5]
                            # Field 7 is the alignment score (used for tie-breaking). We still pass it to the mismatch helper as before.
                            sam_md_or_score = fields[7]

                            full_query_name_annot = sam_query_name_full_annot.split(':')[0]

                            try:
                                umi_idx_annot, sub_reads_annot, _ = full_query_name_annot.split('.')
                            except ValueError:
                                sysOps.throw_status(f"Warning: SAM query parse error '{full_query_name_annot}'")
                                continue

                            # (1) Ignore UMI-subclusters where the name starts with "-1"
                            if umi_idx_annot.startswith("-1") or full_query_name_annot.startswith("-1"):
                                continue

                            # Keep prior mutation-string behavior (even if field 7 is actually a score in your pipeline)
                            mutation_str_annot = get_internal_mismatches(sam_cigar_annot, sam_md_or_score, sam_start_pos_annot)

                            out_f = [umi_idx_annot, sam_start_pos_annot, mutation_str_annot]

                            if current_overlapping_gtf_data_annot:
                                unique_contigs = sorted(list(set(f[1] for f in current_overlapping_gtf_data_annot)))
                                
                                # PATCH A: Replace the problematic list comprehensions and fix the any() call
                                unique_gene_names    = sorted({row[5] for row in current_overlapping_gtf_data_annot
                                                               if len(row) > 5 and row[5] and row[5] != "NONE"})
                                unique_gene_biotypes = sorted({row[6] for row in current_overlapping_gtf_data_annot
                                                               if len(row) > 6 and row[6] and row[6] != "NONE"})
                                unique_gene_ids      = sorted({row[7] for row in current_overlapping_gtf_data_annot
                                                               if len(row) > 7 and row[7] and row[7] != "NONE"})
                                unique_tx_names      = sorted({row[8] for row in current_overlapping_gtf_data_annot
                                                               if len(row) > 8 and row[8] and row[8] != "NONE"})

                                agg_contigs_a = "|".join(unique_contigs) if unique_contigs else sam_src_contig_annot 
                                agg_gene_names_a = "|".join(unique_gene_names) if unique_gene_names else "NA"
                                agg_gene_biotypes_a = "|".join(unique_gene_biotypes) if unique_gene_biotypes else "NA"
                                agg_gene_ids_a = "|".join(unique_gene_ids) if unique_gene_ids else "NA" 
                                agg_tx_names_a = "|".join(unique_tx_names) if unique_tx_names else "NA"
                                
                                gene_name_to_use_a = agg_gene_names_a if agg_gene_names_a != "NA" else agg_gene_ids_a 
                                attn_rank_a = "3"
                                if "rRNA" in agg_gene_biotypes_a:
                                    attn_rank_a = "1"
                                elif any(part.isdigit() for part in agg_contigs_a.split('|')) or "MT" in agg_contigs_a:
                                    attn_rank_a = "2"
                                
                                out_f.extend([agg_contigs_a, gene_name_to_use_a, agg_gene_biotypes_a, agg_tx_names_a, 
                                              sub_reads_annot, attn_rank_a, full_query_name_annot])
                            else:
                                out_f.extend([sam_src_contig_annot, "NA", "genome", "NA", sub_reads_annot, "4", full_query_name_annot])
                                                        
                            # Collect stats for near-duplicate grouping
                            try:
                                score_val = float(sam_md_or_score)
                            except Exception:
                                score_val = 0.0
                            try:
                                sub_reads_val = int(sub_reads_annot)
                            except Exception:
                                sub_reads_val = 0
                            try:
                                start_val = int(sam_start_pos_annot)
                            except Exception:
                                start_val = 0

                            key_uc = (umi_idx_annot, sam_src_contig_annot)
                            fqn_key = full_query_name_annot
                            if key_uc not in dup_buckets: dup_buckets[key_uc] = {}
                            st = dup_buckets[key_uc].get(fqn_key)
                            if st is None:
                                dup_buckets[key_uc][fqn_key] = {"min_start": start_val, "reads": sub_reads_val, "score_sum": score_val, "score_n": 1}
                            else:
                                st["min_start"] = min(st["min_start"], start_val)
                                st["reads"] = max(st["reads"], sub_reads_val)  # defensive: keep the largest read tally seen
                                st["score_sum"] += score_val
                                st["score_n"] += 1

                            f_umi_assign_out.write(",".join(map(str, out_f)) + "\n")
                        else:
                            # unconditional assignment — safe due to earlier guard
                            gtf_nr_annot = fields[0]
                            if gtf_nr_annot in current_gtf_nrs_active_annot:
                                current_gtf_nrs_active_annot.remove(gtf_nr_annot)
                                current_overlapping_gtf_data_annot = [
                                    f_item for f_item in current_overlapping_gtf_data_annot if f_item[0] != gtf_nr_annot
                                ]
                            else:
                                current_gtf_nrs_active_annot.append(gtf_nr_annot)
                                current_overlapping_gtf_data_annot.append(fields)

                # ---------- Near-duplicate detection & resolution ----------
                near_dup_report = os.path.join(star_output_dir, "nearby_subcluster_sets.txt")
                allowed_pairs = set()  # (UMI, FQN) that survive de-dup

                with open(near_dup_report, 'w') as dup_out:
                    # Header: UMI,contig,anchor_start, members(fqn:reads:avgScore:start;...), chosen
                    dup_out.write("UMI,contig,anchor_start,members,chosen\n")
                    for (umi_key, contig_key), fqn_map in dup_buckets.items():
                        # Sort members by their representative start
                        entries = sorted(
                            ((stats["min_start"], fqn, stats) for fqn, stats in fqn_map.items()),
                            key=lambda x: x[0]
                        )

                        group = []
                        anchor = None

                        def finalize_group(g):
                            if not g:
                                return
                            # Singletons pass through and are not reported as duplicates
                            if len(g) == 1:
                                _, fqn_single, _st = g[0]
                                allowed_pairs.add((umi_key, fqn_single))
                                return
                            # Choose authoritative: max reads, then max average score
                            def key_fn(t):
                                st = t[2]
                                avg = st["score_sum"] / max(1, st["score_n"])
                                return (st["reads"], avg)

                            best = max(g, key=key_fn)
                            _, best_fqn, best_st = best
                            allowed_pairs.add((umi_key, best_fqn))

                            # Emit the full duplicate set with stats
                            members = []
                            for s, fqn, st in g:
                                avg = st["score_sum"] / max(1, st["score_n"])
                                members.append(f"{fqn}:{st['reads']}:{avg:.6f}:{st['min_start']}")
                            dup_out.write(f"{umi_key},{contig_key},{g[0][0]},{';'.join(members)},{best_fqn}\n")

                        for s, fqn, st in entries:
                            if anchor is None or abs(s - anchor) <= 10:
                                if anchor is None:
                                    anchor = s
                                group.append((s, fqn, st))
                            else:
                                finalize_group(group)
                                group = [(s, fqn, st)]
                                anchor = s
                        finalize_group(group)

                # Filter RAW annotations → keep only authoritative subclusters (plus singletons)
                with open(tmp_umi_assignments_file_raw, 'r') as src, open(tmp_umi_assignments_file, 'w') as dst:
                    for ln in src:
                        fld = ln.rstrip().split(',')
                        if len(fld) < 10:
                            continue
                        umi = fld[0]
                        fqn = fld[9]
                        if (umi, fqn) in allowed_pairs:
                            dst.write(ln)

                # Raw file served its purpose
                try:
                    os.remove(tmp_umi_assignments_file_raw)
                except Exception:
                    pass


                # tmp_umi_assignments_file: 10 COLS (UMI_idx, ..., FQN)

                sysOps.big_sort(f" {sort_temp_dir_option} -t',' -k1,1 -k9n,9 -k10,10 ", tmp_umi_assignments_file, tmp_sorted_umi_assignments_file)
                if sysOps.check_file_exists(consolidated_umi_assignments_file) and os.path.exists(consolidated_umi_assignments_file): os.remove(consolidated_umi_assignments_file)
                with open(consolidated_umi_assignments_file, 'w') as f_consol_out, \
                     open(tmp_sorted_umi_assignments_file, 'r') as f_sorted_assign_in:
                    # ---- consolidation ----
                    def flush_consol_line(key_tuple, recs, fh):
                        """
                        key_tuple = (UMI_idx, FQN)
                        recs      = list of alignment-row slices (currently cols 1-8: aln_start..attn_rank)
                        Writes:   UMI_idx, merged cols, FQN

                        • Deduplicates identical ALIGNMENT ROWS (not individual columns).
                          This preserves cross-column integrity when multiple alignments exist.
                        • Preserves left-to-right order inside each pipe-list.
                        """
                        if not key_tuple or not recs:
                            return
                        umi, fqn = key_tuple

                        # 1) Deduplicate entire rows first (preserve order)
                        unique_recs = []
                        seen = set()
                        for r in recs:
                            t = tuple(r)
                            if t not in seen:
                                seen.add(t)
                                unique_recs.append(r)

                        if not unique_recs:
                            return

                        # 2) Transpose -> columns
                        # Guard against malformed record-widths (zip truncates to shortest).
                        ncols = len(unique_recs[0])
                        if ncols == 0:
                            return
                        if any(len(r) != ncols for r in unique_recs):
                            unique_recs = [r for r in unique_recs if len(r) == ncols]
                            if not unique_recs:
                                return

                        cols = list(zip(*unique_recs))
                        merged = ["|".join(col) for col in cols]

                        out_row = [umi] + merged + [fqn]
                        fh.write(",".join(out_row) + "\n")
                    
                    current_key = None
                    bucket      = []                     # accumulates 8‑field slices
                    for ln in f_sorted_assign_in:
                        fld = ln.rstrip().split(',')
                        if len(fld) < 10:
                            continue
                        k = (fld[0], fld[9])             # (UMI, FQN)
                        if k != current_key and current_key is not None:
                            flush_consol_line(current_key, bucket, f_consol_out)
                            bucket = []
                        current_key = k
                        bucket.append(fld[1:9])          # keep cols 1‑8 (incl. attn_rank)
                    if current_key is not None:
                        flush_consol_line(current_key, bucket, f_consol_out)
                # consolidated_umi_assignments_file: UMI_idx(1), pipe_starts(2)...pipe_sub_reads(8), FQN(9) -- 9 COLS
                
                sysOps.big_sort(f" {sort_temp_dir_option} -t',' -k2,2 ", gd + f"max_base_use_uxi{amp_ind}.txt", seqindex_sorted_uxi_path)
                sysOps.sh(f"join -t',' -1 2 -2 1 -o1.1,1.2,2.2 {shlex.quote(seqindex_sorted_uxi_path)} {shlex.quote(gd + f'seq_sort_clust_uxi{amp_ind}.txt')} > {shlex.quote(tmp_seq_sort_clust_path)}")
                sysOps.big_sort(f" {sort_temp_dir_option} -t',' -k3,3 ", tmp_seq_sort_clust_path, clust_sort_clust_path)
                # Ensure RHS (consolidated_umi_assignments_file) is sorted by its join key (col 1)
                tmp_sorted_consolidated = gd + f"tmp_sorted_consolidated_{amp_ind}.txt"
                sysOps.big_sort(f" {sort_temp_dir_option} -t',' -k1,1 ",
                                consolidated_umi_assignments_file,
                                tmp_sorted_consolidated)

                sysOps.sh(
                    f"join -t',' -1 3 -2 1 "
                    f"-o1.1,1.2,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,2.10 "
                    f"{shlex.quote(clust_sort_clust_path)} "
                    f"{shlex.quote(tmp_sorted_consolidated)} "
                    f"> {shlex.quote(umi_seq_assignments_file_temp)}"
                )

                try:
                    os.remove(tmp_sorted_consolidated)
                except:
                    pass

                # umi_seq_assignments_file_temp: 11 COLS (UMI_seq, UniqSeqIdx, pipe_starts...pipe_sub_reads, attn_rank, FQN)

                if add_sequences_to_labelfiles and sysOps.check_file_exists(trimmed_fragments_file_path) \
                                               and os.path.getsize(trimmed_fragments_file_path) > 0:
                    # (#1) Keep FQN AND append sequence -> output 12 columns
                    sysOps.big_sort(f" {sort_temp_dir_option} -t',' -k11,11 ", umi_seq_assignments_file_temp, tmp_assign_qname_path)
                    sysOps.big_sort(f" {sort_temp_dir_option} -t',' -k1,1 ", trimmed_fragments_file_path, tmp_frags_qname_path)
                    sysOps.sh(
                        f"join -t',' -1 11 -2 1 "
                        f"-o1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,1.11,2.2 "
                        f"{shlex.quote(tmp_assign_qname_path)} {shlex.quote(tmp_frags_qname_path)} "
                        f"> {shlex.quote(final_umi_seq_assignments_path)}"
                    )
                else:
                    # normalize to 10 columns: take $1..$9 and $11 (drop $10=attn_rank, keep $11=FQN)
                    if os.path.exists(final_umi_seq_assignments_path) \
                       and final_umi_seq_assignments_path != umi_seq_assignments_file_temp:
                        os.remove(final_umi_seq_assignments_path)
                    sysOps.sh(
                        f"awk -F',' 'BEGIN{{OFS=\",\"}}"
                        f" {{print $1,$2,$3,$4,$5,$6,$7,$8,$9,$11}}' "
                        f"{shlex.quote(umi_seq_assignments_file_temp)} > {shlex.quote(final_umi_seq_assignments_path)}"
                   )
                    try: 
                        os.remove(umi_seq_assignments_file_temp)
                    except: 
                        pass

                sysOps.big_sort(f" {sort_temp_dir_option} -t',' -k1,1 ", final_umi_seq_assignments_path, sorted_assignments_file_path)

                # Clean up Stage 3 intermediates
                stage3_intermediates_to_clean = [
                    final_sorted_alignments_for_merge, gtf_sorted_file, merged_algn_gtf_file,
                    consolidated_umi_assignments_file, tmp_umi_assignments_file, tmp_sorted_umi_assignments_file,
                    seqindex_sorted_uxi_path, tmp_seq_sort_clust_path, clust_sort_clust_path,
                    tmp_assign_qname_path, tmp_frags_qname_path
                ]
                if add_sequences_to_labelfiles and os.path.exists(final_umi_seq_assignments_path) and final_umi_seq_assignments_path != sorted_assignments_file_path:
                     stage3_intermediates_to_clean.append(final_umi_seq_assignments_path)
                if os.path.exists(umi_seq_assignments_file_temp) and umi_seq_assignments_file_temp != final_umi_seq_assignments_path : # if it wasn't renamed
                    stage3_intermediates_to_clean.append(umi_seq_assignments_file_temp)

                if sysOps.check_file_exists(sorted_assignments_file_path):
                    for f_s3_intermed in stage3_intermediates_to_clean:
                        if sysOps.check_file_exists(f_s3_intermed) and os.path.exists(f_s3_intermed):
                            try: os.remove(f_s3_intermed)
                            except OSError as e: sysOps.throw_status(f"Warning: S3 intermed cleanup {f_s3_intermed} failed: {e}")
            else:
                # -------------------------------
                # STAGE 3 (STAR-less fabrication)
                # -------------------------------
                sysOps.throw_status(f"STAGE 3: STAR-less fabrication for amp{amp_ind}")
                # Build clust→(umi_seq, uniq_seq_index)
                sysOps.big_sort(f" {sort_temp_dir_option} -t',' -k2,2 ", gd + f"max_base_use_uxi{amp_ind}.txt", seqindex_sorted_uxi_path)
                sysOps.sh(f"join -t',' -1 2 -2 1 -o1.1,1.2,2.2 {shlex.quote(seqindex_sorted_uxi_path)} {shlex.quote(gd + f'seq_sort_clust_uxi{amp_ind}.txt')} > {shlex.quote(tmp_seq_sort_clust_path)}")
                sysOps.big_sort(f" {sort_temp_dir_option} -t',' -k3,3 ", tmp_seq_sort_clust_path, clust_sort_clust_path)
                # Fabricate consolidated per-subcluster assignments (10 columns) to mirror STAR output schema:
                # (UMI_idx, start=-1, mut=None, contig=NA, gene_id=NA, biotype=NA, tx_name=NA, sub_reads, attn_rank=4, FQN)
                # We key off subcluster FQNs in amp{amp_ind}_seqcons_trimmed_fragments.txt so the join for sequence appends matches STAR behavior.
                if not sysOps.check_file_exists(trimmed_fragments_file_path):
                    sysOps.throw_status(f"STAR-less path: no {trimmed_fragments_file_path}; cannot fabricate per-subcluster assignments for amp{amp_ind}.")
                else:
                    # Derive UMI index and subcluster read count from FQN: UMIIdx.<7-digit> .subK
                    sysOps.sh(
                        "awk -F',' 'BEGIN{OFS=\",\"} {"
                        "  fqn=$1; split(fqn,a,\".\"); umi=a[1]; rp=a[2]; "
                        "  subreads = (rp+0); if(subreads<=0) subreads=1; "
                        "  print umi,-1,\"None\",\"NA\",\"NA\",\"NA\",\"NA\",subreads,4,fqn"
                        "}' "
                        + shlex.quote(trimmed_fragments_file_path)
                        + " > " + shlex.quote(consolidated_umi_assignments_file)
                    )
                    # Join to produce umi_seq_assignments (10 cols): UMI_seq,uniqIdx,then 9 cols from consolidated file (2..10)
                    # Ensure RHS is sorted by col 1 for join
                    tmp_sorted_consolidated = gd + f"tmp_sorted_consolidated_{amp_ind}.txt"
                    sysOps.big_sort(f" {sort_temp_dir_option} -t',' -k1,1 ",
                                    consolidated_umi_assignments_file,
                                    tmp_sorted_consolidated)

                    sysOps.sh(
                        f"join -t',' -1 3 -2 1 "
                        f"-o1.1,1.2,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,2.10 "
                        f"{shlex.quote(clust_sort_clust_path)} "
                        f"{shlex.quote(tmp_sorted_consolidated)} "
                        f"> {shlex.quote(umi_seq_assignments_file_temp)}"
                    )

                    try:
                        os.remove(tmp_sorted_consolidated)
                    except:
                        pass

                    # umi_seq_assignments_file_temp: 11 COLS (UMI_seq, UniqSeqIdx, pipe_starts...attn_rank, FQN)
                    # Optionally append trimmed sub-consensus sequence and KEEP FQN
                    if add_sequences_to_labelfiles and sysOps.check_file_exists(trimmed_fragments_file_path) \
                                                  and os.path.getsize(trimmed_fragments_file_path) > 0:
                        # (#1) Fix FQN key column to 11 (not 10) and keep FQN in the final output
                        sysOps.big_sort(f" {sort_temp_dir_option} -t',' -k11,11 ", umi_seq_assignments_file_temp, tmp_assign_qname_path)
                        sysOps.big_sort(f" {sort_temp_dir_option} -t',' -k1,1 ", trimmed_fragments_file_path, tmp_frags_qname_path)
                        sysOps.sh(
                            f"join -t',' -1 11 -2 1 "
                            f"-o1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,1.11,2.2 "
                            f"{shlex.quote(tmp_assign_qname_path)} {shlex.quote(tmp_frags_qname_path)} "
                            f"> {shlex.quote(final_umi_seq_assignments_path)}"
                        )
                    else:
                        if os.path.exists(final_umi_seq_assignments_path) and final_umi_seq_assignments_path != umi_seq_assignments_file_temp:
                            os.remove(final_umi_seq_assignments_path)
                        sysOps.sh(
                            f"awk -F',' 'BEGIN{{OFS=\",\"}}"
                            f" {{print $1,$2,$3,$4,$5,$6,$7,$8,$9,$11}}' "
                            f"{shlex.quote(umi_seq_assignments_file_temp)} > {shlex.quote(final_umi_seq_assignments_path)}"
                        )
                        try:
                            os.remove(umi_seq_assignments_file_temp)
                        except:
                            pass
                    # Materialize the canonical output used downstream
                    sysOps.big_sort(f" {sort_temp_dir_option} -t',' -k1,1 ", final_umi_seq_assignments_path, sorted_assignments_file_path)
                    # Cleanup STAR-less intermediates
                    for f_s3_nostar in [
                        seqindex_sorted_uxi_path, tmp_seq_sort_clust_path, clust_sort_clust_path,
                        consolidated_umi_assignments_file, tmp_assign_qname_path, tmp_frags_qname_path
                    ]:
                        if sysOps.check_file_exists(f_s3_nostar) and os.path.exists(f_s3_nostar):
                            try: os.remove(f_s3_nostar)
                            except OSError as e: sysOps.throw_status(f"Warning: STAR-less cleanup {f_s3_nostar} failed: {e}")
            # End of STAR / STAR-less check (Stage 3)
        # End of top-level file existence check for amp_ind loop
        # End of for amp_ind loop

        # --- Write amp: stats from existing seqcons_trimmed files if missing ---
        stats_file_to_check = gd + "pairing_stats.txt" if sysOps.check_file_exists("pairing_stats.txt") else gd + "umi_stats.txt"
        if os.path.isfile(stats_file_to_check):
            existing_amp_prefixes = set()
            with open(stats_file_to_check) as _sf:
                for _sln in _sf:
                    if 'amp:' in _sln:
                        existing_amp_prefixes.add(_sln.split('amp:')[0])
            for amp_ind in range(2):
                if str(amp_ind) in existing_amp_prefixes:
                    continue
                trimmed = gd + f"amp{amp_ind}_seqcons_trimmed.txt"
                if not os.path.isfile(trimmed):
                    continue
                counts = [0, 0, 0]
                with open(trimmed) as _f:
                    for _ln in _f:
                        _parts = _ln.strip().split(',', 2)
                        if len(_parts) < 3 or _parts[2].strip() == 'N':
                            continue
                        _tot = int(_parts[1])
                        if _tot == 1: counts[0] += 1
                        elif _tot == 2: counts[1] += 1
                        elif _tot >= 3: counts[2] += 1
                with open(stats_file_to_check, 'a') as _sf:
                    _sf.write(f"{amp_ind}amp:{counts[0]},{counts[1]},{counts[2]}\n")


        # --- STAGE 4: Calculate Gene Statistics (whenever the pair exists, STAR or STAR-less) ---
        if sysOps.check_file_exists("sorted_umi_seq_assignments0.txt") and sysOps.check_file_exists("sorted_umi_seq_assignments1.txt"):
            gene_stats_path = os.path.join(gd, "gene_stats.txt")
            try:
                newest_assign = max(
                    os.path.getmtime(os.path.join(gd, f"sorted_umi_seq_assignments{a}.txt"))
                    for a in (0, 1)
                )
                if os.path.exists(gene_stats_path) and os.path.getmtime(gene_stats_path) >= newest_assign:
                    sysOps.throw_status("STAGE 4: gene_stats.txt is up-to-date; skipping.")
                else:
                    sysOps.throw_status("STAGE 4: Getting gene stats ...")
                    input_files_for_gene_stats = ["sorted_umi_seq_assignments0.txt", "sorted_umi_seq_assignments1.txt"]
                    gene_stats_results_list = get_gene_stats(input_files_for_gene_stats)
                    with open(gene_stats_path, 'w') as outfile:
                        outfile.write(",".join(map(str,gene_stats_results_list)))
                    sysOps.throw_status("Done getting gene stats.")
            except Exception as _e:
                sysOps.throw_status("STAGE 4: gene stats skipped/failed (" + str(_e) + ").")


        sysOps.throw_status("STAGE 5: UEI Matching.")
        if uei_matchfilepath is not None:
            uei_paths = [p for p in str(uei_matchfilepath).split('+') if p]
            for amp_ind in range(2):
                # Ensure clust→(UMI seq, uniq-seq index) table exists
                clust_sort_clust_path = gd + f"clust_sort_clust_uxi{amp_ind}.txt"
                if not sysOps.check_file_exists(f"clust_sort_clust_uxi{amp_ind}.txt"):
                    seqindex_sorted_uxi_path = gd + f"seqindex_sorted_uxi_{amp_ind}.txt"
                    tmp_seq_sort_clust_path  = gd + f"tmp_seq_sort_clust_uxi{amp_ind}.txt"
                    sysOps.big_sort(f" {sort_temp_dir_option} -t',' -k2,2 ",
                                    f"max_base_use_uxi{amp_ind}.txt", f"seqindex_sorted_uxi_{amp_ind}.txt")
                    sysOps.sh("join -t ',' -1 2 -2 1 -o1.1,1.2,2.2 "
                            + shlex.quote(gd + f"seqindex_sorted_uxi_{amp_ind}.txt") + " "
                            + shlex.quote(gd + f"seq_sort_clust_uxi{amp_ind}.txt")
                            + " > " + shlex.quote(gd + f"tmp_seq_sort_clust_uxi{amp_ind}.txt"))
                    sysOps.big_sort(f" {sort_temp_dir_option} -t',' -k3,3 ",
                                    f"tmp_seq_sort_clust_uxi{amp_ind}.txt", f"clust_sort_clust_uxi{amp_ind}.txt")

                    try:
                        os.remove(gd + f"seqindex_sorted_uxi_{amp_ind}.txt")
                        os.remove(gd + f"tmp_seq_sort_clust_uxi{amp_ind}.txt")
                    except:
                        pass

                for uei_path in uei_paths:
                    # Skip if label_pt is already up-to-date for this (amp_ind, uei_path)
                    try:
                        uei_out_dir = os.path.join(gd, uei_path)
                        out_label   = os.path.join(uei_out_dir, f"label_pt{amp_ind}.txt")
                        in_assign   = os.path.join(gd, f"sorted_umi_seq_assignments{amp_ind}.txt")
                        in_max      = os.path.join(uei_out_dir, f"max_base_use_uxi{amp_ind}.txt")
                        in_map      = os.path.join(uei_out_dir, f"seq_sort_clust_uxi{amp_ind}.txt")
                        if (
                            os.path.exists(out_label)
                            and os.path.exists(in_assign)
                            and os.path.exists(in_max)
                            and os.path.exists(in_map)
                        ):
                            newest_in = max(
                                os.path.getmtime(in_assign),
                                os.path.getmtime(in_max),
                                os.path.getmtime(in_map),
                            )
                            if os.path.getmtime(out_label) >= newest_in:
                                sysOps.throw_status(
                                    f"STAGE 5: Skipping UEI match amp{amp_ind} -> {uei_out_dir} (label_pt up-to-date)."
                                )
                                continue
                    except Exception:
                        pass

                    # 1) Build UEI dataset UMI→(uniq_idx,cluster,UEI_readcount)
                    # Normalize RHS file (from UEI dataset) to avoid CRLF join issues
                    rhs = gd + uei_path + f"seq_sort_clust_uxi{amp_ind}.txt"
                    sysOps.sh(
                        f"tr -d '\\r' < {shlex.quote(rhs)} > {shlex.quote(rhs + '.nocr')} && "
                        f"mv {shlex.quote(rhs + '.nocr')} {shlex.quote(rhs)}"
                    )

                    sysOps.big_sort(f" {sort_temp_dir_option} -t',' -k2,2 ",
                                    uei_path + f"max_base_use_uxi{amp_ind}.txt",
                                    f"sorted_UEIdata_uxi{amp_ind}.txt")
                    
                    sysOps.sh("join -t ',' -1 2 -2 1 -o1.1,1.2,2.2,1.4 "
                            + shlex.quote(gd + f"sorted_UEIdata_uxi{amp_ind}.txt") + " "
                            + shlex.quote(gd + uei_path + f"seq_sort_clust_uxi{amp_ind}.txt")
                            + " > " + shlex.quote(gd + f"tmp_UEIdata_uxi{amp_ind}.txt"))

                    # 2) Exact-match join (inputs sorted on key; C locale for bytewise collation)
                    # file→file sort: use big_sort
                    sysOps.big_sort(" -t',' -k1,1 ",
                                    gd + f"tmp_UEIdata_uxi{amp_ind}.txt",
                                    gd + f"UEIdata_uxi{amp_ind}.txt")
                    # Ensure join precondition: col-1 sorted (C locale via sysOps.big_sort’s sort)
                    _tmp_sua = f"tmp_sorted_umi_seq_assignments{amp_ind}.txt"
                    sysOps.big_sort(" -t ',' -k1,1 ",
                                    f"sorted_umi_seq_assignments{amp_ind}.txt",
                                    _tmp_sua)
                    os.replace(gd + _tmp_sua, gd + f"sorted_umi_seq_assignments{amp_ind}.txt")

                    # Strip CRLF on both final join inputs now that both exist
                    for _fname in (f"sorted_umi_seq_assignments{amp_ind}.txt", f"UEIdata_uxi{amp_ind}.txt"):
                        _src = gd + _fname
                        sysOps.sh(
                            f"tr -d '\\r' < {shlex.quote(_src)} > {shlex.quote(_src + '.nocr')} && "
                            f"mv {shlex.quote(_src + '.nocr')} {shlex.quote(_src)}"
                        )

                    extra_field = "1.11" if add_sequences_to_labelfiles else "1.10"

                    sysOps.sh(
                        "join -t ',' -1 1 -2 1 "
                        f"-o 2.3,1.3,1.4,1.5,1.6,1.7,1.8,1.9,{extra_field},2.4 "
                        + shlex.quote(gd + f"sorted_umi_seq_assignments{amp_ind}.txt") + " "
                        + shlex.quote(gd + f"UEIdata_uxi{amp_ind}.txt") + " "
                        + "> " + shlex.quote(gd + f"unsorted_label_pt{amp_ind}.txt")
                    )

                    # 3) Order rows: UEI cluster, sub_reads desc, UEI_reads desc
                    #    (We keep *all* rows; this order drives the semicolon order.)
                    sysOps.big_sort(" -t',' -k1,1 -k8rn,8 -k10rn,10 ",
                                    gd + f"unsorted_label_pt{amp_ind}.txt",
                                    gd + f"tmp_label_pt{amp_ind}.txt")

                    # 4) Aggregate ALL sub-clusters per UEI into semicolon-aligned lists.
                    #
                    # Input cols (tmp_label_pt):
                    #   1=UEI_cluster,
                    #   2=start, 3=mut, 4=contig, 5=gene_id, 6=biotype, 7=tx, 8=sub_reads, 9=FQN, 10=UEI_reads
                    # Output:
                    #   1=UEI_cluster,
                    #   2..10 = semicolon-joined lists, *same length across all fields*
                    uei_out_dir = os.path.join(gd, uei_path)
                    os.makedirs(uei_out_dir, exist_ok=True)
                    # If sequences were requested, copy raw per-subcluster trimmed fragments into the UEI match-path.
                    # This retains the "ground truth" benchmarking information (FQN -> cDNA-insert sequence) inside the UEI directory.
                    if add_sequences_to_labelfiles:
                        try:
                            for _src in (
                                os.path.join(gd, f"amp{amp_ind}_seqcons_trimmed_fragments.txt"),
                                os.path.join(gd, f"amp{amp_ind}_seqcons_trimmed.txt"),
                            ):
                                if os.path.exists(_src) and os.path.getsize(_src) > 0:
                                    _dst = os.path.join(uei_out_dir, os.path.basename(_src))
                                    # Avoid copying onto itself (can happen if uei_out_dir == gd)
                                    if os.path.abspath(_src) != os.path.abspath(_dst):
                                        shutil.copy2(_src, _dst)
                        except Exception as _e:
                            sysOps.throw_status(
                                f"Warning: could not copy benchmarking sequence files for amp{amp_ind} to {uei_out_dir}: {_e}"
                            )
                    agg_out = os.path.join(uei_out_dir, f"label_pt{amp_ind}.txt")

                    sysOps.sh(
                        "awk -F',' 'BEGIN{OFS=\",\"}"
                        " function append(a, v){ a[0]++; a[a[0]]=v }"
                        " function join_sc(a,   i,s){ s=\"\"; for(i=1;i<=a[0];i++){ if(i>1)s=s\";\"; s=s a[i] } return s }"
                        " { key=$1;"
                        "   if(NR==1){ prev=key }"
                        "   if(key!=prev){"
                        "     print prev, join_sc(c2), join_sc(c3), join_sc(c4), join_sc(c5), join_sc(c6), join_sc(c7), join_sc(c8), join_sc(c9), join_sc(c10);"
                        "     delete c2; delete c3; delete c4; delete c5; delete c6; delete c7; delete c8; delete c9; delete c10;"
                        "     prev=key"
                        "   }"
                        "   append(c2,$2); append(c3,$3); append(c4,$4); append(c5,$5);"
                        "   append(c6,$6); append(c7,$7); append(c8,$8); append(c9,$9); append(c10,$10);"
                        " }"
                        " END{ if(NR>0){"
                        "   print prev, join_sc(c2), join_sc(c3), join_sc(c4), join_sc(c5), join_sc(c6), join_sc(c7), join_sc(c8), join_sc(c9), join_sc(c10)"
                        " } }' "
                        + shlex.quote(gd + f"tmp_label_pt{amp_ind}.txt")
                        + " > " + shlex.quote(agg_out)
                    )

                    # cleanup
                    for p in [f"unsorted_label_pt{amp_ind}.txt",
                            f"tmp_label_pt{amp_ind}.txt",
                            f"UEIdata_uxi{amp_ind}.txt",
                            f"sorted_UEIdata_uxi{amp_ind}.txt",
                            f"tmp_UEIdata_uxi{amp_ind}.txt"]:
                        try: os.remove(gd + p)
                        except: pass

        else:
            sysOps.throw_status("STAGE 5: UEI Matching skipped (uei_matchfilepath = " + str(uei_matchfilepath) + ").")

        sysOps.throw_status(f"get_amp_consensus processing completed.")
        
    if True: 
        sysOps.throw_status("Cleaning up ephemeral temporary files (AWK scripts etc.)...")
        unique_ephemeral_files = sorted(list(set(ephemeral_temp_files_overall))) 
        for temp_file_path in unique_ephemeral_files:
            if os.path.exists(temp_file_path):
                try: os.remove(temp_file_path)
                except OSError as e: sysOps.throw_status(f"Warning: Could not remove ephemeral temp file {temp_file_path}. Error: {e}")
        
        for amp_ind_final_clean in range(2):
            star_dir_to_clean_final = gd + f"STARalignment{amp_ind_final_clean}/"
            if STARindexdir and gtffile and os.path.isdir(star_dir_to_clean_final):
                if sysOps.check_file_exists(f"sorted_umi_seq_assignments{amp_ind_final_clean}.txt"):
                    sysOps.throw_status(f"Cleaning up completed STAR directory: {star_dir_to_clean_final}")
                    try: sysOps.sh(f"rm -rf {shlex.quote(star_dir_to_clean_final)}") 
                    except Exception as e: sysOps.throw_status(f"Warning: Could not remove STAR dir {star_dir_to_clean_final}. Error: {e}")
        sysOps.throw_status("Ephemeral temporary file cleanup attempt complete.")
    
    
    sysOps.throw_status("Completed consensus, returning.")
    return


    
def assign_umi_pairs(uei_ind):
    # uei_ind will always be >=2; outputfile is uei*_assoc.txt
    
    # line_sorted_clust_* has columns
    # 1. source file line (ascending order LEXICOGRAPHICALLY)
    # 2. cluster index
    
    sysOps.throw_status("Finalizing consensus UMI sequences ...")
        
    line_num0 = int(sysOps.sh("wc -l < " + sysOps.globaldatapath + "line_sorted_clust_uxi0.txt"))
    line_num1 = int(sysOps.sh("wc -l < " + sysOps.globaldatapath + "line_sorted_clust_uxi1.txt"))
    line_num2 = int(sysOps.sh("wc -l < " + sysOps.globaldatapath + "line_sorted_clust_uxi" + str(uei_ind) + ".txt"))
    line_num_uei = int(sysOps.sh("wc -l < " + sysOps.globaldatapath + "uxi" + str(uei_ind) + ".txt"))
    
    if line_num0 != line_num1 or line_num0 != line_num2 or line_num0 != line_num_uei:
        sysOps.throw_status('Error: [line_num0,line_num1,line_num2,line_num_uei] = ' + str([line_num0,line_num1,line_num2,line_num_uei]))
        sysOps.exitProgram()
    
    sysOps.sh("paste -d, " + sysOps.globaldatapath + "line_sorted_clust_uxi" + str(uei_ind) + ".txt " + sysOps.globaldatapath + "line_sorted_clust_uxi0.txt " + sysOps.globaldatapath + "line_sorted_clust_uxi1.txt " + sysOps.globaldatapath + "uxi" + str(uei_ind) + ".txt" + " > " + sysOps.globaldatapath + "tmp_uei_pairing.txt")
    
    sysOps.sh("awk -F, '{if($2 >= 0 && $4 >= 0 && $6 >= 0){print $2 \",\" $4 \",\" $6 \",\" $8 \",\" $9}}' " + sysOps.globaldatapath + "tmp_uei_pairing.txt > " + sysOps.globaldatapath + "filtered_uei" + str(uei_ind) + "_pairing.txt")
    
    os.remove(sysOps.globaldatapath + "tmp_uei_pairing.txt")
    
    sysOps.throw_status('Sorting UEI-pairings ...')
        
    sysOps.big_sort(" -k1n,1 -k2n,2 -k3n,3 -t \",\" ","filtered_uei" + str(uei_ind) + "_pairing.txt", 
                    "sorted_filtered_uei" + str(uei_ind) + "_pairing.txt")
    os.remove(sysOps.globaldatapath + "filtered_uei" + str(uei_ind) + "_pairing.txt")
    
    sysOps.throw_status('Collapsing unique pairings.')
    
    sysOps.sh("uniq -c "
              + sysOps.globaldatapath + "sorted_filtered_uei" + str(uei_ind) + "_pairing.txt" + " | sed -e 's/^ *//;s/ /,/' > " 
              + sysOps.globaldatapath + "tmp_enum_uniq_pairing.txt")
    # tmp_enum_uniq_sorted_indexed_* now has the following columns:
    # 1. number of unique entries (reads) from consecutive sequences of sorted_uei_pairing.txt
    # 2. UEI cluster
    # 3-4. UMI cluster pairings
    # 5-6. read-formats
    
    sysOps.big_sort(" -k2,2 -k1rn,1 -t \",\"  ","tmp_enum_uniq_pairing.txt","sorted_enum_uniq_uxi" + str(uei_ind) + "_pairing.txt")
    
    os.remove(sysOps.globaldatapath + "sorted_filtered_uei" + str(uei_ind) + "_pairing.txt")
    os.remove(sysOps.globaldatapath + "tmp_enum_uniq_pairing.txt")
    
    # retain only top read-count pairing for each UEI, unless there's a tie, in which case exclude altogether 
    sysOps.sh("awk -F, 'BEGIN{prev_uei_index=-1;top_readnum=-1;top_umi1=-1;top_umi2=-1;top_read1_form=-1;top_read2_form=-1;}"
              + "{if($2!=prev_uei_index){if(top_readnum>0){print top_readnum \",\" prev_uei_index \",\" top_umi1 \",\" top_umi2 \",\" top_read1_form \",\" top_read2_form;} top_readnum=$1;top_umi1=$3;top_umi2=$4;top_read1_form=$5;top_read2_form=$6;}"
              + "else if(top_readnum==$1){top_readnum=-1;}prev_uei_index=$2;}"
              + "END{if(top_readnum>0){print top_readnum \",\" prev_uei_index \",\" top_umi1 \",\" top_umi2  \",\" top_read1_form \",\" top_read2_form;}}' "
              + sysOps.globaldatapath + "sorted_enum_uniq_uxi" + str(uei_ind) + "_pairing.txt > " + sysOps.globaldatapath + "consensus_pairings_uxi" + str(uei_ind) + ".txt")
    
    # consensus_pairings_uxi*.txt contains the following columns
    # 1. number of unique entries (reads)
    # 2. UEI cluster (sorted lexicographically)
    # 3-4. UMI cluster pairings
    # 5-6. Read formats
    
    sysOps.throw_status('Done.')

    return

def output_inference_inp_files(min_reads_per_assoc, min_uei_per_umi, min_uei_per_assoc, uei_classification=None, rarefaction_only=False):
    # all inputs are lists having length equal to the number of UEI types
    # concatenate all consensus pairings files
    
    sysOps.throw_status('Outputting inference input-files with the following parameters: min_reads_per_assoc=' + str(min_reads_per_assoc) + ', min_uei_per_umi=' + str(min_uei_per_umi) + ', min_uei_per_assoc=' + str(min_uei_per_assoc))
    
    if not sysOps.check_file_exists("uei_assoc.txt"):
        if sysOps.check_file_exists("all_consensus_pairings.txt"):
            os.remove(sysOps.globaldatapath + "all_consensus_pairings.txt")
        if sysOps.check_file_exists("pairing_stats.txt"):
            os.remove(sysOps.globaldatapath + "pairing_stats.txt")
        # consensus_pairings_uxi*.txt contains the following columns
        # 1. number of unique entries (reads)
        # 2. UEI cluster
        # 3-4. UMI cluster pairings
        # 5-6. Read formats
        uei_ind = 2
        while True:
            if sysOps.check_file_exists("consensus_pairings_uxi" + str(uei_ind) + ".txt"):
                # replace UEI cluster indices with uei_ind (just specifying the type of UEI), append to file all_consensus_pairings.txt
                # determine read-abundances for UEIs, append to pairing_stats.txt
                sysOps.sh("awk -F, 'BEGIN{n1read=0;n2read=0;n3read=0;}"
                          + "{print $1 \",\" " + str(uei_ind) + " \",\" $3 \",\" $4  \",\" $5 \",\" $6 >> \"" + sysOps.globaldatapath + "all_consensus_pairings.txt\";"
                          + "if($1==1)n1read++;else if($1==2)n2read++; else n3read++;}END{print \""+ str(uei_ind) +":\" n1read \",\" n2read \",\" n3read >> \"" + sysOps.globaldatapath + "pairing_stats.txt\"}' "
                          + sysOps.globaldatapath + "consensus_pairings_uxi" + str(uei_ind) + ".txt")
                
                
            else:
                break
            uei_ind += 1
            
        # complete pairing_stats.txt
        # all_consensus_pairings.txt contains columns:
        # 1. number of unique entries (reads)
        # 2. UEI type-index
        # 3-4. UMI-cluster pairings
        # 5-6. Read formats
        for uxi_ind in range(2):
            sysOps.big_sort(" -k" + str(uxi_ind+3) + "n," + str(uxi_ind+3) + " -t ','  ","all_consensus_pairings.txt","tmp_sorted_uxi" + str(uxi_ind) + ".txt")
            # sort by UMI index
            sysOps.sh("awk -F, 'BEGIN{n1read=0;n2read=0;n3read=0;prev_uxi_ind=-1;my_readnum=0;}"
                      + "{if($" + str(uxi_ind+3) + "==prev_uxi_ind){my_readnum+=$1;}"
                      + "else{if(my_readnum==1){n1read++;}else if(my_readnum==2){n2read++;}else if(my_readnum>=3){n3read++;}"
                      + "my_readnum=$1; prev_uxi_ind=$" + str(uxi_ind+3) + ";}}"
                      + "END{if(my_readnum==1){n1read++;}else if(my_readnum==2){n2read++;}else if(my_readnum>=3){n3read++;}"
                      + "print \""+ str(uxi_ind) +":\" n1read \",\" n2read \",\" n3read >> \"" + sysOps.globaldatapath + "pairing_stats.txt\"}' "
                      + sysOps.globaldatapath + "tmp_sorted_uxi" + str(uxi_ind) + ".txt")
            os.remove(sysOps.globaldatapath + "tmp_sorted_uxi" + str(uxi_ind) + ".txt")
        
                
        # min_uei_per_assoc and min_uei_per_umi, although taken as lists in the settings file, are only used for their first element
        conditional_assoc_str = "(my_ueinum>="+str(min_uei_per_umi)+")"
        
        #perform iterative filter using the other 3 function-input filters
        # sort LEXICOGRAPHICALLY by all association triples (note UMI2 is first sort argument)
            
        sysOps.big_sort(" -k4,4 -k3,3 -t ',' ","all_consensus_pairings.txt","tmp_sorted_all.txt")
                
        # tmp_sorted_all.txt
        # all_consensus_pairings.txt contains columns:
        # 1. number of unique entries (reads)
        # 2. UEI type-index
        # 3-4. UMI-cluster pairings (sorted on UMI2, all associations together)
        # 5-6. Read formats
        
        # enumerate unique associations
        sysOps.sh("awk -F, 'BEGIN{prev_col1=-1;prev_col2=-1;prev_col3=-1;prev_col4=-1;prev_col5=-1;prev_col6=-1;assoc_num=0;}"
                  + "{if(prev_col1 >= 0){print assoc_num \",\" prev_col1 \",\" prev_col2 \",\" prev_col3 \",\" prev_col4 \",\" prev_col5 \",\" prev_col6;"
                  + "if(prev_col3!=$3 || prev_col4!=$4){ assoc_num++;}}"
                  + "prev_col1=$1; prev_col2=$2; prev_col3=$3; prev_col4=$4; prev_col5=$5; prev_col6=$6;}"
                  + "END{print assoc_num \",\" prev_col1 \",\" prev_col2 \",\" prev_col3 \",\" prev_col4 \",\" prev_col5 \",\" prev_col6;}' " + sysOps.globaldatapath + "tmp_sorted_all.txt > " + sysOps.globaldatapath + "sorted_assoc.txt")
                
        # sorted_assoc.txt:
        # 1. association index
        # 2. number of unique entries (reads)
        # 3. UEI type-index
        # 4-5. UMI-cluster pairings (sorted on UMI2, all associations together)
        # 6-7. Read formats
        
        num_assoc = -1
        filter_iter = 0
        while True:
            sysOps.big_sort(" -k1,1 -t ',' ","sorted_assoc.txt","resorted_assoc.txt") # sort lex
            
            os.remove(sysOps.globaldatapath + "sorted_assoc.txt")
            if filter_iter > 0 and num_assoc == num_assoc_init:
            
                # at this point, consolidate read-formats based on UEI-classification; if no uei_classification=None, set all classification indices to 0
                # consolidate UEIs into associations
                sysOps.sh("awk -F, 'BEGIN{prev_col1=-1;prev_col3=-1;prev_col4=-1;prev_col5=-1;my_ueicount1=0;my_readcount=0;}"
                          + "{if(prev_col1!=$1){if(prev_col1>=0){print (prev_col3 \",\" prev_col4 \",\" prev_col5 \",\" my_ueicount1 \",\" my_readcount);} my_ueicount1=0;my_readcount=0;}prev_col1=$1;my_readcount+=$2;prev_col3=$3;prev_col4=$4;prev_col5=$5;my_ueicount1++;}"
                          + "END{print (prev_col3 \",\" prev_col4 \",\" prev_col5 \",\" my_ueicount1 \",\" my_readcount);}' "
                          + sysOps.globaldatapath + "resorted_assoc.txt > "
                          + sysOps.globaldatapath + "uei_assoc.txt")
                os.remove(sysOps.globaldatapath + "resorted_assoc.txt")
                break
                
            num_assoc_init = int(sysOps.sh("wc -l < " + sysOps.globaldatapath + "resorted_assoc.txt").strip('\n'))
            
            # generate list of all associations passing filter
            sysOps.sh("awk -F, 'BEGIN{prev_col1=-1;my_readnum=0;my_ueinum=0;}"
                      + "{if(prev_col1==$1){my_readnum+=$2;my_ueinum+=1;}"
                      + "else{if(my_readnum>=" + str(min_reads_per_assoc) + " && my_ueinum>=" +str(min_uei_per_assoc)+ "){print prev_col1;}"
                      + "my_readnum=$2;prev_col1=$1;my_ueinum=1;}}"
                      + "END{if(my_readnum>=" + str(min_reads_per_assoc) + " && my_ueinum>=" +str(min_uei_per_assoc)+"){print prev_col1;}}' "
                      + sysOps.globaldatapath + "resorted_assoc.txt > "
                      + sysOps.globaldatapath + "passed_assoc.txt")
                    
            sysOps.sh("join -t ',' -1 1 -2 1 -o1.1,1.2,1.3,1.4,1.5,1.6,1.7 "
                      + sysOps.globaldatapath + "resorted_assoc.txt " + sysOps.globaldatapath + "passed_assoc.txt > "
                      + sysOps.globaldatapath + "sorted_assoc.txt")
            
            # sorted_assoc.txt:
            # 1. association index (lex-sorted)
            # 2. number of unique entries (reads)
            # 3. UEI type-index
            # 4-5. UMI-cluster pairings
            # 6-7. Read formats
                    
            os.remove(sysOps.globaldatapath + "resorted_assoc.txt")
            sysOps.big_sort(" -k4,4 -k1,1 -t ',' ","sorted_assoc.txt","resorted_assoc.txt") # sort lex
            os.remove(sysOps.globaldatapath + "sorted_assoc.txt")
            
            sysOps.sh("awk -F, 'BEGIN{prev_col4=-1;prev_col1=-1;my_assocnum=0;my_ueinum=0;}"
                      + "{if(prev_col4==$4){my_ueinum++; if(prev_col1!=$1){my_assocnum++;}}"
                      + "else{if(my_ueinum>=" + str(min_uei_per_umi) + "){print prev_col4;}"
                      + "my_ueinum=1;my_assocnum=1;prev_col1=$1;prev_col4=$4;}}"
                      + "END{if(my_ueinum>=" + str(min_uei_per_umi) + "){print prev_col4;}}' "
                      + sysOps.globaldatapath + "resorted_assoc.txt > "
                      + sysOps.globaldatapath + "passed_assoc.txt")
            
            sysOps.sh("join -t ',' -1 4 -2 1 -o1.1,1.2,1.3,1.4,1.5,1.6,1.7 "
                      + sysOps.globaldatapath + "resorted_assoc.txt " + sysOps.globaldatapath + "passed_assoc.txt > "
                      + sysOps.globaldatapath + "sorted_assoc.txt")
                      
            num_assoc = int(sysOps.sh("wc -l < " + sysOps.globaldatapath + "sorted_assoc.txt").strip('\n'))
            if num_assoc == 0:
                sysOps.throw_status("No UMIs passed filter.")
                return
            
            os.remove(sysOps.globaldatapath + "resorted_assoc.txt")
            sysOps.big_sort(" -k5,5 -k1,1 -t ',' ","sorted_assoc.txt","resorted_assoc.txt") # sort lex
            os.remove(sysOps.globaldatapath + "sorted_assoc.txt")
            
            sysOps.sh("awk -F, 'BEGIN{prev_col5=-1;prev_col1=-1;my_assocnum=0;my_ueinum=0;}"
                      + "{if(prev_col5==$5){my_ueinum++; if(prev_col1!=$1){my_assocnum++;}}"
                      + "else{if(my_ueinum>=" + str(min_uei_per_umi) + "){print prev_col5;}"
                      + "my_ueinum=1;my_assocnum=1;prev_col1=$1;prev_col5=$5;}}"
                      + "END{if(my_ueinum>=" + str(min_uei_per_umi) + "){print prev_col5;}}' "
                      + sysOps.globaldatapath + "resorted_assoc.txt > "
                      + sysOps.globaldatapath + "passed_assoc.txt")
            
            sysOps.sh("join -t ',' -1 5 -2 1 -o1.1,1.2,1.3,1.4,1.5,1.6,1.7 "
                      + sysOps.globaldatapath + "resorted_assoc.txt " + sysOps.globaldatapath + "passed_assoc.txt > "
                      + sysOps.globaldatapath + "sorted_assoc.txt")
            num_assoc = int(sysOps.sh("wc -l < " + sysOps.globaldatapath + "sorted_assoc.txt").strip('\n'))
            if num_assoc == 0:
                sysOps.throw_status("No UMIs passed filter.")
                if rarefaction_only:
                    # Ensure rarefaction consumers have a stable (single-line) output to parse.
                    with open(sysOps.globaldatapath + "sorted_sl_counts.txt", "w") as out_f:
                        out_f.write("-1,0\n")
                return
            
            sysOps.throw_status('Deleted ' + str(num_assoc_init-num_assoc) + '/' + str(num_assoc_init) + ' UEIs on iteration ' + str(filter_iter))
            
            filter_iter += 1
    
    # uei_assoc.txt now has the following columns:
    # 1. UEI type
    # 2-3. UMI cluster pairings
    # 4. number of UEIs for this association
    # 5. number of reads
    
    sl_clust_assoc("uei_assoc.txt",filter_if_umis_labeled=True) # final clustering (NO FURTHER FILTERING OF ASSOCIATIONS)
    return    

def sl_clust_assoc(out_file, filter_if_umis_labeled = False, top_grps = 10, min_grp_size = 1000, add_extra=False):
    # will output top_grps sl_grps, requiring each to have >=min_grp_size UMIs
    # NOTE: THIS FUNCTION DOES NOT FURTHER FILTER UMI-UMI ASSOCIATIONS
    
    uei_assoc_path = sysOps.globaldatapath + out_file
            
    # partition uei_assoc into non-contiguous matrices
    # SL groups are seeded by UMI indices ADJUSTED TO BE NON-OVERLAPPING
    
    uei_assoc = np.loadtxt(uei_assoc_path,dtype=np.float64,delimiter=',')[:,1:]
    
    if sysOps.check_file_exists("label_pt0.txt") and sysOps.check_file_exists("label_pt1.txt") and filter_if_umis_labeled:
        sysOps.throw_status("Found UMI labels. Filtering ...")
        sysOps.sh("awk -F, '{print $1;}' " + sysOps.globaldatapath + "label_pt0.txt > " + sysOps.globaldatapath + "label_pt0_indices.txt") # just print cluster index
        sysOps.sh("awk -F, '{print $1;}' " + sysOps.globaldatapath + "label_pt1.txt > " + sysOps.globaldatapath + "label_pt1_indices.txt")
        label_pt0_indices = np.loadtxt(sysOps.globaldatapath + "label_pt0_indices.txt",dtype=np.int64)
        label_pt1_indices = np.loadtxt(sysOps.globaldatapath + "label_pt1_indices.txt",dtype=np.int64)
        max_ind0 = int(max(np.max(label_pt0_indices),np.max(uei_assoc[:,0])))
        max_ind1 = int(max(np.max(label_pt1_indices),np.max(uei_assoc[:,1])))
        passed0 = np.zeros(max_ind0+1,dtype=np.bool_)
        passed1 = np.zeros(max_ind1+1,dtype=np.bool_)
        passed0[label_pt0_indices] = True
        passed1[label_pt1_indices] = True
        prev_n_assoc = uei_assoc.shape[0]
        uei_assoc = uei_assoc[np.add(passed0[np.int64(uei_assoc[:,0])],passed1[np.int64(uei_assoc[:,1])]),:]
        sysOps.throw_status("Filtered " + str(uei_assoc.shape[0]) + "/" + str(prev_n_assoc) + " associations.")
        os.remove(sysOps.globaldatapath + "label_pt0_indices.txt")
        os.remove(sysOps.globaldatapath + "label_pt1_indices.txt")
        os.rename(uei_assoc_path, sysOps.globaldatapath + "unfiltered_" + out_file)
        np.savetxt(uei_assoc_path,np.concatenate([2*np.ones([uei_assoc.shape[0],1]),uei_assoc],axis=1),fmt='%i',delimiter=',')
    
    max_umi1_index = int(np.max(uei_assoc[:,0]))
    uei_assoc[:,1] += max_umi1_index+1
    max_all_index = int(np.max(uei_assoc[:,1]))
    index_link_array = np.arange(max_all_index+1,dtype=np.int64)
    sysOps.throw_status('Performing SL clustering ...')
    optimOps.min_contig_edges(index_link_array,np.ones(max_all_index+1,dtype=np.int64),uei_assoc,uei_assoc.shape[0])
    uei_assoc[:,1] -= max_umi1_index+1
    sysOps.throw_status('Completed SL clustering. Writing ...')
    unique_umi1 = np.unique(np.int64(uei_assoc[:,0]))
    unique_umi2 = np.unique(np.int64(uei_assoc[:,1]))
    
    sl_assignments_1 = np.zeros([unique_umi1.shape[0],3],dtype=np.int64)
    sl_assignments_1[:,0] = index_link_array[unique_umi1]
    sl_assignments_1[:,2] = unique_umi1
    sl_assignments_2 = np.ones([unique_umi2.shape[0],3],dtype=np.int64)
    sl_assignments_2[:,0] = index_link_array[max_umi1_index+1+unique_umi2]
    sl_assignments_2[:,2] = unique_umi2
    
    if sysOps.check_file_exists("sl_assignments.txt"):
        os.remove(sysOps.globaldatapath + "sl_assignments.txt")
    
    np.savetxt(sysOps.globaldatapath + "sl_assignments.txt",np.concatenate([sl_assignments_1,sl_assignments_2],axis=0),fmt='%i',delimiter=',')
    
    # re-load uei_assoc in original form
    uei_assoc = np.loadtxt(uei_assoc_path,dtype=np.int64,delimiter=',')
    np.savetxt(sysOps.globaldatapath + "uei_assoc_slgrps.txt", np.concatenate([uei_assoc[:,:3].T,[index_link_array[uei_assoc[:,1]]],[index_link_array[uei_assoc[:,2]+max_umi1_index+1]],[uei_assoc[:,3]],[uei_assoc[:,4]]],axis=0).T,delimiter=',',fmt='%i')
    # uei_assoc_slgrps.txt has columns:
    # 1. UEI type
    # 2. UMI1 cluster index
    # 3. UMI2 cluster index
    # 4. UMI1 SL index
    # 5. UMI2 SL index
    # 6. UEI count (classification 1)
    # 7. UEI count (classification 2)
    
    del sl_assignments_1, sl_assignments_2, uei_assoc
    sysOps.throw_status('Wrote SL clustering.')
    
    # sl_assignments.txt has columns:
    # 1. SL index
    # 2. 0 if UMI1, 1 if UMI2
    # 3. UMI index
    
    sysOps.big_sort(" -k1n,1 -t \",\" ","sl_assignments.txt","sorted_sl_assignments.txt",parallel=True)
    
    sysOps.sh("awk -F, 'BEGIN{prev_index=-1;mycount=0;}{if($1!=prev_index){if(mycount>0)print(prev_index \",\" mycount);prev_index=$1;mycount=1;}else{mycount++;}}END{if(mycount>0)print(prev_index \",\" mycount);}' " 
              + sysOps.globaldatapath + "sorted_sl_assignments.txt" + " > " + sysOps.globaldatapath + "sl_counts.txt")
    
    # sl_counts.txt has columns:
    # 1. SL index
    # 2. SL index UMI count
    
    sysOps.big_sort(" -k2rn,2 -t \",\" ","sl_counts.txt","sorted_sl_counts.txt",parallel=True)
    sysOps.sh("awk -F, '{print NR-1 \",\" $1 \",\" $2}' " + sysOps.globaldatapath + "sorted_sl_counts.txt > " + sysOps.globaldatapath + "enum_sorted_sl_counts.txt")
    
    # enum_sorted_sl_counts.txt has columns:
    # 1. SL index rank (starting with 0 = most abundant)
    # 2. SL index 
    # 3. SL index UMI count

    sysOps.big_sort(" -k2,2 -t \",\" ","enum_sorted_sl_counts.txt","sorted_enum_sl_counts.txt",parallel=True)
    # sorted_enum_sl_counts.txt has columns:
    # 1. SL index rank 
    # 2. SL index (sorted lexicographic ascending)
    # 3. SL index UMI count
    
    sysOps.big_sort(" -k4,4 -t \",\" ","uei_assoc_slgrps.txt","uei_assoc_sorted_slgrps.txt",parallel=True)
    # uei_assoc_sorted_slgrps_*.txt has columns:
    # 1. UEI type
    # 2. UMI1 cluster index  
    # 3. UMI2 cluster index
    # 4. UMI1 SL index  (sorted lexicographic ascending)
    # 5. UMI2 SL index
    # 6. UEI count (classification 1)
    # 7. UEI count (classification 2)

    sysOps.sh("join -t \",\" -1 2 -2 4 -o2.1,2.2,2.3,2.6,2.7,1.1,1.3 "
              + sysOps.globaldatapath + "sorted_enum_sl_counts.txt " + sysOps.globaldatapath + "uei_assoc_sorted_slgrps.txt" 
              + " > " + sysOps.globaldatapath + "uei_assoc_ranked_sl.txt")
    # uei_assoc_ranked_sl.txt has columns:
    # 1. UEI type
    # 2. UMI1 cluster index
    # 3. UMI2 cluster index
    # 4. UEI count (classification 1)
    # 5. UEI count (classification 2)
    # 6. SL index rank
    # 7. SL UMI count
    
    sysOps.throw_status("Printing data subsets.")
    for i_dir in range(top_grps):
        try:
            os.mkdir(sysOps.globaldatapath + "uei_grp" + str(i_dir))
        except:
            sysOps.throw_status(sysOps.globaldatapath + "uei_grp" + str(i_dir) + " already exists.")
            
                
    # write UEI subsets to different directories
    sysOps.sh("awk -F, '{sl_rank = $6; if(sl_rank<" + str(top_grps) + " && $7>=" + str(min_grp_size)
              + "){print($1 \",\" $2 \",\" $3 \",\" $4 \",\" $5) >> (\"" + sysOps.globaldatapath + "uei_grp\" sl_rank \"//link_assoc.txt\")}}' "
              + sysOps.globaldatapath + "uei_assoc_ranked_sl.txt")
    sysOps.throw_status("Inference inputs written to:")
    num_grps = 0
    for i_dir in range(top_grps):
        if sysOps.check_file_exists("uei_grp" + str(i_dir) + "//link_assoc.txt"):
            sysOps.throw_status(sysOps.globaldatapath + "uei_grp" + str(i_dir))
            num_grps += 1
        else:
            os.rmdir(sysOps.globaldatapath + "uei_grp" + str(i_dir))
    if num_grps == 0:
        sysOps.throw_status("No groups exceeded " + str(min_grp_size) + " UMI minimum.")
    
    # ------------------------------------------------------------
    # EXTRA EDGES (grp0): build uei_grp0/link_assoc_extra.txt
    #   1) both endpoints are already nodes in uei_grp0/link_assoc.txt
    #   2) >= 2 READS support the pairing (summed over UEIs)
    #   3) pairing key (UMI1,UMI2) is NOT already present in link_assoc.txt
    #
    # Uses all_consensus_pairings.txt, whose columns are:
    #   1 reads, 2 UEI-type, 3 UMI1, 4 UMI2, 5-6 read-formats
    # ------------------------------------------------------------
    if add_extra and sysOps.check_file_exists("uei_grp0//link_assoc.txt") and sysOps.check_file_exists("all_consensus_pairings.txt"):

        grp0_path = sysOps.globaldatapath + "uei_grp0//"

        # remove any prior output
        if sysOps.check_file_exists("uei_grp0//link_assoc_extra.txt"):
            os.remove(grp0_path + "link_assoc_extra.txt")

        # ---- (A) node-sets from the existing grp0 graph ----
        # UMI1 nodes are in column 2; UMI2 nodes are in column 3 of link_assoc.txt
        sysOps.sh("awk -F, '{print $2}' " + grp0_path + "link_assoc.txt > " + grp0_path + "tmp_extra_umi1_nodes.txt")
        sysOps.big_sort(" -k1,1 -u ", "uei_grp0//tmp_extra_umi1_nodes.txt", "uei_grp0//extra_umi1_nodes.txt", parallel=True)
        os.remove(grp0_path + "tmp_extra_umi1_nodes.txt")

        sysOps.sh("awk -F, '{print $3}' " + grp0_path + "link_assoc.txt > " + grp0_path + "tmp_extra_umi2_nodes.txt")
        sysOps.big_sort(" -k1,1 -u ", "uei_grp0//tmp_extra_umi2_nodes.txt", "uei_grp0//extra_umi2_nodes.txt", parallel=True)
        os.remove(grp0_path + "tmp_extra_umi2_nodes.txt")

        # ---- (B) filter all_consensus_pairings down to UEIs whose endpoints are both in grp0 ----
        # join #1: keep only rows whose UMI1 (col3) is in grp0 UMI1-node set
        sysOps.big_sort(" -t ',' -k3,3 -k4,4 ", "all_consensus_pairings.txt", "tmp_extra_allcons_u1sort.txt", parallel=True)
        sysOps.sh("join -t ',' -1 1 -2 3 -o2.1,2.2,2.3,2.4,2.5,2.6 "
                + grp0_path + "extra_umi1_nodes.txt " + sysOps.globaldatapath + "tmp_extra_allcons_u1sort.txt > "
                + sysOps.globaldatapath + "tmp_extra_allcons_u1filt.txt")
        os.remove(sysOps.globaldatapath + "tmp_extra_allcons_u1sort.txt")

        # join #2: keep only rows whose UMI2 (col4) is in grp0 UMI2-node set
        sysOps.big_sort(" -t ',' -k4,4 -k3,3 ", "tmp_extra_allcons_u1filt.txt", "tmp_extra_allcons_u2sort.txt", parallel=True)
        os.remove(sysOps.globaldatapath + "tmp_extra_allcons_u1filt.txt")

        sysOps.sh("join -t ',' -1 1 -2 4 -o2.1,2.2,2.3,2.4,2.5,2.6 "
                + grp0_path + "extra_umi2_nodes.txt " + sysOps.globaldatapath + "tmp_extra_allcons_u2sort.txt > "
                + sysOps.globaldatapath + "tmp_extra_allcons_grp0_nodes.txt")
        os.remove(sysOps.globaldatapath + "tmp_extra_allcons_u2sort.txt")

        # ---- (C) aggregate per (UEI-type,UMI1,UMI2), require >=2 READS; emit key=UMI1_UMI2 for fast anti-join ----
        # tmp_extra_allcons_grp0_nodes.txt columns:
        #   1 reads, 2 UEI-type, 3 UMI1, 4 UMI2, 5-6 read-formats
        sysOps.big_sort(" -t ',' -k2,2 -k3,3 -k4,4 ", "tmp_extra_allcons_grp0_nodes.txt",
                        "tmp_extra_allcons_grp0_nodes_sorted.txt", parallel=True)
        os.remove(sysOps.globaldatapath + "tmp_extra_allcons_grp0_nodes.txt")

        # candidate edges: key, UEI-type, UMI1, UMI2, UEIcount, READcount
        sysOps.sh(
            "awk -F, -v OFS=, -v us=_ "
            "'BEGIN{p2=-1;p3=-1;p4=-1;ueic=0;rc=0;}"
            "{if(p2<0){p2=$2;p3=$3;p4=$4;}"
            " if($2!=p2 || $3!=p3 || $4!=p4){"
            "   if(rc>=2){print p3 us p4, p2, p3, p4, ueic, rc;}"
            "   p2=$2;p3=$3;p4=$4;ueic=0;rc=0;"
            " }"
            " ueic++; rc+=$1;"
            "}"
            "END{if(p2>=0 && rc>=2){print p3 us p4, p2, p3, p4, ueic, rc;}}' "
            + sysOps.globaldatapath + "tmp_extra_allcons_grp0_nodes_sorted.txt > "
            + sysOps.globaldatapath + "tmp_extra_candidate_edges.txt"
        )
        os.remove(sysOps.globaldatapath + "tmp_extra_allcons_grp0_nodes_sorted.txt")

        # ---- (D) build key-list of edges already in grp0/link_assoc.txt ----
        # IMPORTANT: this treats "pairing exists" ignoring UEI-type (key uses only UMI1_UMI2)
        sysOps.sh("awk -F, -v us=_ '{print $2 us $3}' " + grp0_path + "link_assoc.txt > "
                + sysOps.globaldatapath + "tmp_extra_existing_keys.txt")
        sysOps.big_sort(" -k1,1 -u ", "tmp_extra_existing_keys.txt", "tmp_extra_existing_keys_sorted.txt", parallel=True)
        os.remove(sysOps.globaldatapath + "tmp_extra_existing_keys.txt")

        # ---- (E) anti-join (drop edges already present), then strip key and write link_assoc_extra.txt ----
        sysOps.big_sort(" -t ',' -k1,1 ", "tmp_extra_candidate_edges.txt",
                        "tmp_extra_candidate_edges_sorted.txt", parallel=True)
        os.remove(sysOps.globaldatapath + "tmp_extra_candidate_edges.txt")

        # Output columns match link_assoc.txt’s first 5 columns:
        #   UEI-type, UMI1, UMI2, UEIcount, READcount
        sysOps.sh("join -t ',' -1 1 -2 1 -v1 -o1.2,1.3,1.4,1.5,1.6 "
                + sysOps.globaldatapath + "tmp_extra_candidate_edges_sorted.txt "
                + sysOps.globaldatapath + "tmp_extra_existing_keys_sorted.txt > "
                + grp0_path + "link_assoc_extra.txt")

        # cleanup temp sorts
        os.remove(sysOps.globaldatapath + "tmp_extra_candidate_edges_sorted.txt")
        os.remove(sysOps.globaldatapath + "tmp_extra_existing_keys_sorted.txt")

    
    return

