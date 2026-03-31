import threads_bootstrap
from threads_bootstrap import NTHREADS
import os
import libOps
import sysOps
import dnamicOps
import hashAlignments
import time
import optimOps
import itertools
import numpy as np

class masterProcess:
    def __init__(self):
            
        self.my_starttime = time.time()
        
    def generate_uxi_library(self,path):
    
        original_datapath = str(sysOps.globaldatapath)

        sysOps.initiate_runpath(path)
        
        myLibObj = libOps.libObj(settingsfilename = 'lib.settings')
        if not sysOps.check_file_exists('rejected.txt'):
            if sysOps.check_file_exists('rejected.txt.gz') and not sysOps.check_file_exists('readcounts.txt'):
                sysOps.sh('gunzip ' + sysOps.globaldatapath + 'rejected.txt.gz')
            elif not sysOps.check_file_exists('readcounts.txt'):
                myLibObj.partition_fastq_library()
            
        if not sysOps.check_file_exists('readcounts.txt'):
            myLibObj.stack_uxis()
        
        if sysOps.check_file_exists('rejected.txt'):
            sysOps.sh('gzip ' + sysOps.globaldatapath + 'rejected.txt')
        
        if not sysOps.check_file_exists('ncrit.txt') or not (sysOps.check_file_exists('umi_stats.txt') or sysOps.check_file_exists('pairing_stats.txt')):
            self.generate_cluster_analysis(myLibObj.min_reads_per_assoc,myLibObj.min_uei_per_umi,myLibObj.min_uei_per_assoc,myLibObj.uei_classification)
            
        # umi index, amplicon sequence
        dnamicOps.get_amp_consensus(seq_terminate_list = myLibObj.seq_terminate_list,
                                    filter_umi0_amp_len = myLibObj.filter_umi0_amp_len,
                                    filter_umi1_amp_len = myLibObj.filter_umi1_amp_len,
                                    filter_umi0_quickmatch = myLibObj.filter_umi0_quickmatch,
                                    filter_umi1_quickmatch = myLibObj.filter_umi1_quickmatch,
                                    STARindexdir = myLibObj.STARindexdir, gtffile = myLibObj.gtffile,
                                    uei_matchfilepath = myLibObj.uei_matchfilepath,
                                    add_sequences_to_labelfiles = myLibObj.add_sequences_to_labelfiles)

        if True:
            try:
                from annotation import build_cdna_umi_gene_anndata
                cdna_h5ad_path = os.path.join(sysOps.globaldatapath, "cdna_only.h5ad")

                src_paths = []
                for _amp in (0, 1):
                    p = os.path.join(sysOps.globaldatapath, f"sorted_umi_seq_assignments{_amp}.txt")
                    if sysOps.check_file_exists(p):
                        src_paths.append(p)

                if src_paths:
                    newest_src = max(os.path.getmtime(p) for p in src_paths)
                    if (not sysOps.check_file_exists(cdna_h5ad_path)) or (os.path.getmtime(cdna_h5ad_path) < newest_src):
                        adata = build_cdna_umi_gene_anndata(
                            group_path=sysOps.globaldatapath,
                            assignments_root=sysOps.globaldatapath,
                            binary=False,
                            include_sequences=None,  # auto (see annotation.build_cdna_umi_gene_anndata)
                            include_nonunique_genes=False,
                        )
                        if hasattr(adata, "write_h5ad"):
                            adata.write_h5ad(cdna_h5ad_path)
                            sysOps.throw_status("Wrote cDNA-only AnnData to " + cdna_h5ad_path)
            except Exception as e:
                sysOps.throw_status(f"Skipped cdna_only.h5ad build ({type(e).__name__}: {e})")

        # Only run subsampling if subsample directories have not already been analysed
        [_subdirnames, _] = sysOps.get_directory_and_file_list()
        _existing_subs = [d for d in _subdirnames if d.startswith('sub')]
        if not _existing_subs:
            libOps.subsample(myLibObj.seqform_for_params,myLibObj.seqform_rev_params)
        else:
            sysOps.throw_status('Subsample directories already exist; skipping subsample().')

        [subdirnames, filenames] = sysOps.get_directory_and_file_list()
        dirnames = list([subdirname for subdirname in subdirnames if subdirname.startswith('sub')])
        sysOps.throw_status('Performing cluster analysis on sub-directories: ' + str(dirnames))
        del myLibObj
        for dirname in dirnames:
            sysOps.initiate_runpath(path + dirname + '//')
            # Skip fully-completed (pruned) subsample directories
            _is_pruned = (
                (sysOps.check_file_exists('umi_stats.txt') or sysOps.check_file_exists('pairing_stats.txt'))
                and not sysOps.check_file_exists('uxi0.txt')
            )
            if _is_pruned:
                sysOps.throw_status("Skipping " + sysOps.globaldatapath + " (already completed and pruned).")
                continue
            # Skip empty/partial subsample directories (e.g., from older subsample runs that did not
            # finish writing part_* files). stack_uxis() assumes at least one part_{for}_{rev}.txt exists.
            try:
                _has_parts = any(
                    fn.startswith('part_') and fn.endswith('.txt') and fn.count('_') == 2
                    for fn in os.listdir(sysOps.globaldatapath)
                )
            except Exception as e:
                sysOps.throw_status(f"Skipping {sysOps.globaldatapath} (could not list directory: {type(e).__name__}: {e})")
                continue
            if not _has_parts and not sysOps.check_file_exists('line_sorted_clust_uxi0.txt'):
                sysOps.throw_status("Skipping " + sysOps.globaldatapath + " (no part_*_* or cluster files found).")
                continue
            myLibObj = libOps.libObj(settingsfilename='lib.settings')

            # Step 1: stack_uxis
            if sysOps.check_file_exists('line_sorted_clust_uxi0.txt'):
                sysOps.throw_status('Cluster files already exist; skipping stack_uxis().')
            else:
                myLibObj.stack_uxis()

            # Rarefaction/subsampling optimization:
            # stack_uxis() can leave part_* intermediates behind; remove them early.
            try:
                for _fn in os.listdir(sysOps.globaldatapath):
                    if _fn.startswith("part_") and _fn.endswith(".txt"):
                        try:
                            os.remove(sysOps.globaldatapath + _fn)
                        except:
                            pass
            except:
                pass

            # Step 2: generate_cluster_analysis
            if sysOps.check_file_exists('umi_stats.txt') or sysOps.check_file_exists('pairing_stats.txt'):
                sysOps.throw_status('Rarefaction outputs already exist; skipping generate_cluster_analysis().')
            else:
                self.generate_cluster_analysis(myLibObj.min_reads_per_assoc,myLibObj.min_uei_per_umi,myLibObj.min_uei_per_assoc,myLibObj.uei_classification, rarefaction_only=True)

            # Step 3: get_amp_consensus (has internal idempotency guards)
            dnamicOps.get_amp_consensus(myLibObj.seq_terminate_list,myLibObj.filter_umi0_amp_len,myLibObj.filter_umi1_amp_len,myLibObj.filter_umi0_quickmatch,myLibObj.filter_umi1_quickmatch,myLibObj.STARindexdir,myLibObj.gtffile)

            # Step 4: prune to final state
            sysOps.prune_dir_except([
                "umi_stats.txt",
                "pairing_stats.txt",
                "gene_stats.txt",
                "sorted_sl_counts.txt",
                "lib.settings",
                "readcounts.txt",
                "ncrit.txt"
            ])

        sysOps.globaldatapath = str(original_datapath)
                
        return
        
    def dnamic_inference(self,path):
        original_datapath = str(sysOps.globaldatapath)
        sysOps.initiate_runpath(path)
        
        # Basic settings
        myLibObj = libOps.libObj(settingsfilename = 'lib.settings', output_prefix = '_')
        
        original_datataskpath = str(sysOps.globaldatapath)
        [subdirnames, filenames] = sysOps.get_directory_and_file_list()
        dirnames = list([".//"])
        dirnames.extend([subdirname + '//' for subdirname in subdirnames if subdirname.startswith('sub')])
        for dirname in dirnames:
            sl_grp = 0
            while sysOps.check_file_exists('uei_grp' + str(sl_grp) + '//link_assoc.txt'):
                #optimOps.test_ffgt()
                sysOps.initiate_runpath(original_datataskpath + dirname + 'uei_grp' + str(sl_grp) + '//')
                sysOps.throw_status('Initiated run path ' + original_datataskpath + dirname + 'uei_grp' + str(sl_grp) + '//')
                if not sysOps.check_file_exists('Xumi_GSE.txt'):
                    optimOps.run_GSE(output_name = 'Xumi_GSE.txt',params=myLibObj.mySettings)

                sl_grp += 1
        
        sysOps.globaldatapath = str(original_datapath)
        
        return 
        
    def generate_cluster_analysis(self,min_reads_per_assoc, min_uei_per_umi, min_uei_per_assoc, uei_classification = None, rarefaction_only=False):
        # Perform clustering analysis of UMI and UEI sequences, consolidate pairings and determine consenses of these pairings
        
        # ensure all cluster files are removed from directory (in case previously initiated)
        
        basecount_filter_val = 0.75 #maximum fraction of same-base permitted in a single UMI/UEI
        
        uxi_ind = 0
        while True:
            if(sysOps.check_file_exists('uxi' + str(uxi_ind) + '.txt')):
                if not sysOps.check_file_exists('line_sorted_clust_uxi' + str(uxi_ind) + '.txt'):
                    hashAlignments.initiate_hash_alignment('uxi' + str(uxi_ind) + '.txt',basecount_filter_val)
                    # line_sorted_clust_* has columns
                    # 1. uxi file line (ascending order)
                    # 2. cluster index
                else:
                    sysOps.throw_status(sysOps.globaldatapath + 'line_sorted_clust_uxi' + str(uxi_ind) + '.txt found, skipping.')
            else:
                break
            uxi_ind += 1
        
        sysOps.throw_status('Clustering completed. Beginning final output.')
        sysOps.throw_status('Getting amplicon consensus.')
        # amp*_seqcons_trimmed.txt
        
        for uei_ind in range(2,uxi_ind): #has UEI/s if enters loop
            consensus_pairings_filename = "consensus_pairings_uxi" + str(uei_ind) + ".txt"
            if not (sysOps.check_file_exists(consensus_pairings_filename)):
                dnamicOps.assign_umi_pairs(uei_ind)
                # uses line_sorted_clust_uxi(uei_index).txt as input:
                # line_sorted_clust_* has columns
                # 1. source file line (ascending order)
                # 2. cluster index
                # consensus_pairings_filename contains the following columns
                # 1. number of unique entries (reads)
                # 2. UEI cluster
                # 3-4. UMI cluster pairings
            else:
                sysOps.throw_status('Consensus-pairing file found pre-computed.')
                        
        if uxi_ind > 2:
            sysOps.throw_status('Outputting inference files.')
            dnamicOps.output_inference_inp_files(min_reads_per_assoc, min_uei_per_umi, min_uei_per_assoc, uei_classification, rarefaction_only=rarefaction_only)
        else:
            # UMIs only, no UEIs
            for my_uxi_ind in range(uxi_ind):
                sysOps.big_sort(" -k2,2 -t \",\" ","line_sorted_clust_uxi" + str(my_uxi_ind) + ".txt","tmp_clust_sort.txt")
                sysOps.sh("awk -F, 'BEGIN{prev_umi_index=-1;this_umi_reads=0; n1read=0;n2read=0;n3read=0;}"
                          + "{"
                          + "if(prev_umi_index!=$2){"
                          + "if(prev_umi_index>=0){if(this_umi_reads==1){n1read++;}else if(this_umi_reads==2){n2read++;} else if(this_umi_reads>=3){n3read++;}}"
                          + "this_umi_reads=0; prev_umi_index=$2;}"
                          + "this_umi_reads+=1;}"
                          + "END{if(this_umi_reads==1){n1read++;}else if(this_umi_reads==2){n2read++;} else if(this_umi_reads>=3){n3read++;}"
                          + "print \""+ str(my_uxi_ind) +":\" n1read \",\" n2read \",\" n3read;}' " 
                          + sysOps.globaldatapath + "tmp_clust_sort.txt >> " 
                          + sysOps.globaldatapath + "umi_stats.txt")
                os.remove(sysOps.globaldatapath + "tmp_clust_sort.txt")
        sysOps.throw_status('Getting ncrit.')
        libOps.write_ncrit()

        return
                
