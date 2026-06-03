import os
import csv
import re
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import sysOps

def _looks_like_dna_seq_list(s: str, alphabet: str = "ACGTN") -> bool:
    """Heuristic: does `s` look like a DNA string/list?

    Older pipeline variants repurposed label_pt column 8 (historically
    full_query_name_str) to carry cDNA insert consensus sequences (used by
    benchmarking / plot.py). This function tries to distinguish those two
    cases without forcing a rigid schema change.

    Accepts:
      - empty or whitespace -> False
      - semicolon-separated tokens
      - optional pipe-separated ambiguity inside tokens
      - characters in `alphabet` (case-insensitive)

    Rejects:
      - tokens containing digits, '.', '/', '_' etc (typical of query names)
      - common placeholders (NA/NONE/NULL)
    """
    if s is None:
        return False
    st = str(s).strip()
    if not st:
        return False
    up = st.upper()
    if up in {"NA", "N/A", "NONE", "NULL"}:
        return False

    allowed = set(alphabet.upper())
    for ch in up:
        if ch in {";", "|"}:
            continue
        if ch not in allowed:
            return False
    return True


def _find_upwards(start_path: str, filename: str, max_up: int = 6) -> str | None:
    """Search for filename in start_path or up to max_up parent directories."""
    p = os.path.abspath(start_path)
    for _ in range(max_up + 1):
        cand = os.path.join(p, filename)
        if os.path.exists(cand):
            return cand
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return None


def build_umi_gene_anndata(
    group_path: str,
    label_root: str | None = None,
    drop_genes: set[str] | None = None,
    binary: bool = False,
    return_anndata: bool = True,
    include_sequences: bool = False,
    include_nonunique_genes: bool = False,
    include_obs_strings: bool = False,
):
    """
    Build a sparse UMI/node-by-feature AnnData from label_pt files.

    The builder finds ``label_pt0/1.txt`` (or legacy ``label_pts0/1.txt``),
    maps raw UMI indices through ``index_key.npy``, and fills ``adata.X`` with
    subcluster read support per gene feature, or binary support when
    ``binary=True``. Large per-subcluster string columns are omitted by default
    and can be retained with ``include_obs_strings=True``.

    When ``include_sequences=True`` and the label field looks like DNA sequence
    rather than a query name, sequence features are added as ``SEQ:<sequence>``
    columns and as a separate ``adata.layers['seq']`` matrix. Multi-mapping
    genes are dropped unless ``include_nonunique_genes=True``.

    Returns an AnnData object when available and requested; otherwise returns
    ``(X_gene, gene_names, obs_df)``.

    """
    include_sequences = bool(include_sequences)
    include_nonunique_genes = bool(include_nonunique_genes)
    include_obs_strings = bool(include_obs_strings)

    PLACEHOLDERS = {"NA", "N/A", "NONE", "NULL"}
    GENOME_PREFIX = "__genome__:"

    def _is_placeholder(tok: str) -> bool:
        if tok is None:
            return False
        t = str(tok).strip()
        if not t:
            return True
        return t.upper() in PLACEHOLDERS

    def _split_tokens(s: str, sep: str = "|") -> list[str]:
        if not s:
            return []
        return [t.strip() for t in str(s).split(sep) if t.strip() != ""]

    def _pick_token(tokens: list[str], j: int) -> str:
        if not tokens:
            return ""
        if len(tokens) == 1:
            return tokens[0]
        if j < len(tokens):
            return tokens[j]
        return tokens[0]

    def _pick_biotype(counts: dict[str, int] | None) -> str:
        """Pick a single biotype from a read-weighted support dict (deterministic)."""
        if not counts:
            return "gene"
        return sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[0][0]

    # default gene blacklist
    if drop_genes is None:
        drop_genes = {"genome"}

    # Discover label_pt files (with backwards-compatible fallback to label_pts*.txt)
    if label_root is None:
        label_root = group_path

    label_pt_sources: list[tuple[int, str]] = []
    for amp in (0, 1):
        candidates = (f"label_pt{amp}.txt", f"label_pts{amp}.txt")
        found = None
        for fn in candidates:
            found = _find_upwards(label_root, fn)
            if found is None:
                found = _find_upwards(group_path, fn)
            if found is not None:
                break

        if found is None:
            sysOps.throw_status(
                "Label file missing for amp " + str(amp) + "; continuing without it. "
                + f"Tried {candidates} from {label_root} and {group_path}"
            )
            continue

        sysOps.throw_status("Found label path: " + str(found))
        label_pt_sources.append((amp, found))

    if not label_pt_sources:
        raise FileNotFoundError(
            "Could not locate any label_pt*.txt/label_pts*.txt starting from "
            + str(label_root) + " or " + str(group_path)
        )

    label_pt_paths: list[str] = [p for (_amp, p) in label_pt_sources]


    # Determine whether label_pt files include the final UEI-reads column (10th column).
    # Some label formats (e.g. cDNA-only / older runs) may omit it.
    has_uei_reads = False
    for lp in label_pt_paths:
        if not lp:
            continue
        try:
            with open(lp, "r") as _f:
                for _line in _f:
                    _line = _line.strip()
                    if not _line:
                        continue
                    _fields = _line.split(",")
                    if len(_fields) >= 10:
                        has_uei_reads = True
                    break
        except OSError:
            continue
        if has_uei_reads:
            break

    # Load index_key mapping (reindexed ordering used by GSEoutput.txt)
    idx_path = _find_upwards(group_path, "index_key.npy")
    if idx_path is None:
        idx_path = _find_upwards(label_root, "index_key.npy")
    if idx_path is None:
        raise FileNotFoundError("Could not locate index_key.npy upward from group_path/label_root")

    index_key = np.load(idx_path)
    if index_key.ndim != 2 or index_key.shape[1] < 3:
        raise ValueError(f"index_key.npy expected shape (N,>=3), got {index_key.shape}")

    # index_key columns: [type(0/1), raw_index, reindexed_index]
    n_nodes = int(index_key[:, 2].max()) + 1
    # fast lookup: (type, raw_index) -> reindexed row
    type_raw_to_row = {(int(t), int(r)): int(i) for (t, r, i) in index_key[:, :3]}


    # Prepare obs metadata for ALL nodes (aligned to reindexed node order).
    # NOTE: we intentionally do NOT store a redundant "node_id" column; the obs index
    # is the node id (0..n_nodes-1).
    obs = {
        "umi_type": np.full(n_nodes, -1, dtype=np.int8),
        "raw_umi_index": np.full(n_nodes, -1, dtype=np.int32),
        "has_label": np.zeros(n_nodes, dtype=np.bool_),
        "n_subclusters": np.zeros(n_nodes, dtype=np.int32),
        "n_annotated": np.zeros(n_nodes, dtype=np.int32),
        "total_sub_reads": np.zeros(n_nodes, dtype=np.int32),
    }
    if has_uei_reads:
        obs["total_uei_reads"] = np.zeros(n_nodes, dtype=np.int32)

    # These strings are *very* large at 10M+ nodes and are not needed to reproduce X/var.
    # Keep them only when explicitly requested.
    if include_obs_strings:
        obs.update(
            {
                "aln_start_str": np.array([""] * n_nodes, dtype=object),
                "aln_mut_str": np.array([""] * n_nodes, dtype=object),
                "contig_str": np.array([""] * n_nodes, dtype=object),
                "gene_str": np.array([""] * n_nodes, dtype=object),
                "tx_str": np.array([""] * n_nodes, dtype=object),
            }
        )

    if include_sequences:
        # Column 9 (1-based) of label_pt: historically full_query_name, but in sim/benchmark mode
        # may contain DNA/base4 sequence strings. These can bloat the resulting h5ad, so we only
        # include them when explicitly requested.
        obs["n_seqs"] = np.zeros(n_nodes, dtype=int)
        obs["qname_or_seq_str"] = np.array([""] * n_nodes, dtype=object)
        obs["seq_str"] = np.array([""] * n_nodes, dtype=object)

    # Fill umi_type/raw_umi_index from index_key
    for (t, raw, ridx) in index_key[:, :3]:
        obs["umi_type"][int(ridx)] = int(t)
        obs["raw_umi_index"][int(ridx)] = int(raw)

    # Gene and sequence feature maps
    gene_to_col: dict[str, int] = {}
    seq_to_col: dict[str, int] = {}  # only populated when include_sequences=True

    # Read-weighted biotype support per gene feature
    gene_biotype_support: dict[str, dict[str, int]] = {}

    # Sparse COO builder lists
    rows_g, cols_g, data_g = [], [], []
    rows_s, cols_s, data_s = [], [], []  # only populated when include_sequences=True

    def _parse_int_list(s: str) -> list[int]:
        if not s:
            return []
        out: list[int] = []
        for tok in str(s).split(";"):
            tok = tok.strip()
            if not tok:
                continue
            try:
                out.append(int(tok))
            except ValueError:
                # tolerate pipe-separated lists (e.g., "10|10" or "10|12")
                vals = []
                for part in tok.split("|"):
                    part = part.strip()
                    if not part:
                        continue
                    try:
                        vals.append(int(part))
                    except ValueError:
                        pass
                if vals:
                    out.append(max(vals))
        return out

    def _parse_subcluster_reads(tok: str) -> int:
        """Parse one semicolon element of sub_reads_str into an int (robust to pipes)."""
        if tok is None:
            return 1
        t = str(tok).strip()
        if t == "":
            return 1
        try:
            return int(t)
        except ValueError:
            vals = []
            for part in t.split("|"):
                part = part.strip()
                if not part:
                    continue
                try:
                    vals.append(int(part))
                except ValueError:
                    pass
            return int(max(vals)) if vals else 1

    # Iterate whichever label_pt files were found, keeping their true amp indices.
    for amp, label_path in label_pt_sources:
        # label_pt columns:
        # 0 raw_index
        # 1 aln_start_str (semicolon list)
        # 2 aln_mut_str
        # 3 contig_str
        # 4 gene_str
        # 5 biotype_str
        # 6 tx_str
        # 7 sub_reads_str
        # 8 full_query_name_str OR seq_str (depending on pipeline)
        # 9 uei_reads_str
        with open(label_path, "r", newline="") as f:
            reader = csv.reader(f, delimiter=",")
            for fields in reader:
                if not fields:
                    continue
                raw_index_str = fields[0].strip()
                if raw_index_str == "":
                    continue
                try:
                    raw_index = int(raw_index_str)
                except ValueError:
                    continue

                key = (amp, raw_index)
                if key not in type_raw_to_row:
                    continue
                row = type_raw_to_row[key]

                # Guard: require at least up to uei_reads
                if len(fields) < 8:
                    continue

                aln_start_str = fields[1].strip()
                aln_mut_str = fields[2].strip()
                contig_str = fields[3].strip()
                gene_str = fields[4].strip()
                biotype_str = fields[5].strip()
                tx_str = fields[6].strip()
                sub_reads_str = fields[7].strip()
                qname_or_seq = fields[8].strip() if len(fields) > 8 else ""
                uei_reads_str = fields[9].strip() if (has_uei_reads and len(fields) > 9) else ""
 

                obs["has_label"][row] = True
                if include_obs_strings:
                    obs["aln_start_str"][row] = aln_start_str
                    obs["aln_mut_str"][row] = aln_mut_str
                    obs["contig_str"][row] = contig_str
                    obs["gene_str"][row] = gene_str
                    obs["tx_str"][row] = tx_str

                if include_sequences:
                    obs["qname_or_seq_str"][row] = qname_or_seq

                # subcluster count
                sub_entries = [s for s in gene_str.split(";") if s != ""]
                obs["n_subclusters"][row] = len(sub_entries)

                # total reads bookkeeping (best-effort)
                sub_reads = _parse_int_list(sub_reads_str)
                if has_uei_reads:
                    uei_reads = _parse_int_list(uei_reads_str)
                    obs["total_uei_reads"][row] = int(sum(uei_reads)) if uei_reads else 0

                obs["total_sub_reads"][row] = int(sum(sub_reads)) if sub_reads else 0

                # Collect genes for sparse matrix.
                # gene_str and sub_reads_str are semicolon-aligned lists (one entry per subcluster);
                # each gene entry may contain a pipe-separated multi-map set.
                gene_counts: dict[str, int] = {}

                gene_parts = gene_str.split(";") if gene_str else []
                read_parts = sub_reads_str.split(";") if sub_reads_str else []
                biotype_parts = biotype_str.split(";") if biotype_str else []
                contig_parts = contig_str.split(";") if contig_str else []

                for i, gene_part in enumerate(gene_parts):
                    gene_part = gene_part.strip()
                    if not gene_part:
                        continue

                    sub_reads_i = _parse_subcluster_reads(read_parts[i] if i < len(read_parts) else "")

                    gene_tokens = _split_tokens(gene_part, "|")
                    bt_tokens = _split_tokens(biotype_parts[i] if i < len(biotype_parts) else "", "|")
                    ct_tokens = _split_tokens(contig_parts[i] if i < len(contig_parts) else "", "|")

                    feat_bt_pairs: list[tuple[str, str]] = []
                    seen_feats: set[str] = set()
                    for j, g in enumerate(gene_tokens):
                        g = g.strip()
                        if not g or g in drop_genes:
                            continue

                        bt_j = _pick_token(bt_tokens, j).strip()
                        ct_j = _pick_token(ct_tokens, j).strip()

                        # Convert placeholder gene IDs used for unannotated-but-mapped fragments
                        # into safe contig-specific pseudo-features.
                        if _is_placeholder(g) and (bt_j == "" or bt_j.lower() == "genome"):
                            contig_label = ct_j
                            if _is_placeholder(contig_label) or contig_label.lower() == "none":
                                contig_label = "unknown"
                            feat = f"{GENOME_PREFIX}{contig_label}"
                            bt_use = "genome"
                        else:
                            feat = g
                            bt_use = bt_j if bt_j else "gene"
                        
                        # Collapse duplicate feature calls within a subcluster (e.g. "geneA|geneA")
                        if feat in seen_feats:
                            continue
                        seen_feats.add(feat)
                        feat_bt_pairs.append((feat, bt_use))

                    if not feat_bt_pairs:
                        continue
                    if (not include_nonunique_genes) and (len(feat_bt_pairs) != 1):
                        continue

                    for feat, bt_use in feat_bt_pairs:
                        gene_counts[feat] = gene_counts.get(feat, 0) + int(sub_reads_i)
                        bt_dict = gene_biotype_support.setdefault(feat, {})
                        bt_dict[bt_use] = bt_dict.get(bt_use, 0) + int(sub_reads_i)

                obs["n_annotated"][row] = len(gene_counts)
                for g, c in gene_counts.items():
                    if g not in gene_to_col:
                        gene_to_col[g] = len(gene_to_col)
                    rows_g.append(row)
                    cols_g.append(gene_to_col[g])
                    data_g.append(c)

                # Collect sequences for sequence layer (only when requested)
                if include_sequences and _looks_like_dna_seq_list(qname_or_seq):
                    obs["seq_str"][row] = qname_or_seq
                    uniq_seqs = set()
                    for sub in [s for s in qname_or_seq.split(";") if s != ""]:
                        sub = sub.strip()
                        if not sub:
                            continue
                        for s in sub.split("|"):
                            s = s.strip()
                            if not s:
                                continue
                            uniq_seqs.add(s)
                    obs["n_seqs"][row] = len(uniq_seqs)
                    for s in uniq_seqs:
                        if s not in seq_to_col:
                            seq_to_col[s] = len(seq_to_col)
                        rows_s.append(row)
                        cols_s.append(seq_to_col[s])
                        data_s.append(1)

    # Build sparse matrices
    gene_names = [g for g, _ in sorted(gene_to_col.items(), key=lambda kv: kv[1])]
    n_genes = len(gene_names)

    if include_sequences:
        seq_names = [s for s, _ in sorted(seq_to_col.items(), key=lambda kv: kv[1])]
    else:
        seq_names = []
    n_seqs = len(seq_names)

    # Feature space = genes (+ sequences if enabled)
    var_names = gene_names + ([f"SEQ:{s}" for s in seq_names] if include_sequences else [])
    n_vars = len(var_names)
    # Choose the smallest safe dtype for X to reduce h5ad size.
    if binary:
        x_dtype = np.uint8
    else:
        max_c = int(max(data_g)) if data_g else 0
        if max_c <= np.iinfo(np.uint16).max:
            x_dtype = np.uint16
        elif max_c <= np.iinfo(np.uint32).max:
            x_dtype = np.uint32
        else:
            x_dtype = np.uint64
 

    X_gene = coo_matrix(
        (
            np.asarray(data_g, dtype=x_dtype),
            (np.asarray(rows_g, dtype=np.int64), np.asarray(cols_g, dtype=np.int64)),
        ),
        shape=(n_nodes, n_vars),
        dtype=x_dtype,
    ).tocsr()
    X_gene.sum_duplicates()
    if binary:
        X_gene.data[:] = 1

    X_seq = None
    if include_sequences and n_seqs:
        X_seq = coo_matrix(
            (
                np.asarray(data_s, dtype=np.uint8),
                (
                    np.asarray(rows_s, dtype=np.int64),
                    np.asarray(cols_s, dtype=np.int64) + np.int64(n_genes),
                ),
            ),
            shape=(n_nodes, n_vars),
            dtype=np.uint8,
        ).tocsr()
        X_seq.sum_duplicates()
        if binary:
            X_seq.data[:] = 1
    obs_df = pd.DataFrame(obs, index=pd.Index(np.arange(n_nodes, dtype=np.int32), name="node_id"))
 
    try:
        from anndata import AnnData  # optional dependency

        var_df = pd.DataFrame(index=var_names)

        # ---- gene feature metadata (biotype-aware) ----
        gene_feature_types = [_pick_biotype(gene_biotype_support.get(g)) for g in gene_names]

        # genome pseudo-features are prefixed, so we can expose contig cleanly
        gene_ids: list[str] = []
        contigs: list[str] = []
        for g in gene_names:
            if g.startswith(GENOME_PREFIX):
                gene_ids.append("")  # not a real gene identifier
                contigs.append(g.split(":", 1)[1] if ":" in g else "")
            else:
                gene_ids.append(g)
                contigs.append("")

        var_df["feature_type"] = gene_feature_types + (["sequence"] * n_seqs if include_sequences else [])
        var_df["gene_id"] = gene_ids + ([""] * n_seqs if include_sequences else [])
        var_df["contig"] = contigs + ([""] * n_seqs if include_sequences else [])
        var_df["sequence"] = ([""] * n_genes) + (seq_names if include_sequences else [])

        # Make index types explicit (avoid ImplicitModificationWarning and ensure parity)
        obs_df.index = obs_df.index.astype(str)
        var_df.index = var_df.index.astype(str)
        adata = AnnData(X=X_gene, obs=obs_df, var=var_df)

        # Optional sequence one-hot layer (only when include_sequences=True)
        if X_seq is not None:
            adata.layers["seq"] = X_seq

        adata.uns["label_pt_paths"] = label_pt_paths
        adata.uns["label_pt_amps"] = [int(amp) for (amp, _path) in label_pt_sources]
        adata.uns["index_key_path"] = idx_path
        adata.uns["include_sequences"] = include_sequences
        adata.uns["include_nonunique_genes"] = include_nonunique_genes
        adata.uns["include_obs_strings"] = include_obs_strings
        adata.uns["genome_feature_prefix"] = GENOME_PREFIX

        return adata if return_anndata else (X_gene, gene_names, obs_df)

    except Exception:
        # No anndata installed (or other creation error): return minimal tuple
        return (X_gene, gene_names, obs_df)


def build_cdna_umi_gene_anndata(
    group_path: str,
    assignments_root: str = None,
    drop_genes: set = None,
    binary: bool = False,
    return_anndata: bool = True,
    include_sequences: bool = None,
    include_nonunique_genes: bool = False,
    include_obs_strings: bool = False,
    probe_lines: int = 2000,
):
    """Build a UMI-vs-gene AnnData directly from cDNA sorted_umi_seq_assignments*.txt.

    This is intended for the *cDNA directory* (i.e. before UEI matching). Semantics are aligned
    with `build_umi_gene_anndata()` as closely as possible.

    Notes:
      - No UEI linkage exists here, so `total_uei_reads` is not created.
      - `include_sequences=None` enables an *auto* mode:
            include sequences iff sequences are present AND STAR alignment does NOT appear
            to have been performed (to avoid memory blowups in STAR-aligned runs).
    """
    # Local helper functions (kept parallel to build_umi_gene_anndata for consistent semantics)
    PLACEHOLDERS = {"NA", "N/A", "NONE", "NULL"}

    def _is_placeholder(tok: str) -> bool:
        if tok is None:
            return True
        t = str(tok).strip()
        if not t:
            return True
        return t.upper() in PLACEHOLDERS

    def _split_tokens(field: str, sep: str = "|") -> list:
        if field is None:
            return []
        field = field.strip()
        if field == "":
            return []
        # Match build_umi_gene_anndata(): drop empty tokens
        return [t.strip() for t in field.split(sep) if t.strip() != ""]

    def _pick_token(tokens: list, idx: int) -> str:
        if not tokens:
            return ""
        if idx < len(tokens):
            return tokens[idx]
        return tokens[0]

    def _parse_subcluster_reads(tok: str) -> int:
        """Parse one subcluster read token into an int (robust to pipes).

        Match build_umi_gene_anndata() exactly:
          - Empty / unparsable => 1 (best-effort, avoids silently dropping evidence)
          - Pipe-delimited alternatives => take max
          - No semicolon handling here: this function parses *one* subcluster token.
        """
        if tok is None:
            return 1
        t = str(tok).strip()
        if t == "":
            return 1
        try:
            return int(t)
        except ValueError:
            vals = []
            for part in t.split("|"):
                part = part.strip()
                if not part:
                    continue
                try:
                    vals.append(int(part))
                except ValueError:
                    pass
            return int(max(vals)) if vals else 1

    def _pick_biotype(counts: dict | None) -> str:
        """Deterministic biotype choice: (support, name) descending.

        Matches build_umi_gene_anndata() tie-breaking exactly.
        """
        if not counts:
            return "gene"
        return sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[0][0]


    GENOME_PREFIX = "__genome__:"

    if drop_genes is None:
        drop_genes = {"genome"}

    # Identify assignment files
    if assignments_root is None:
        assignments_root = group_path

    assign_paths = []
    for amp_ind in (0, 1):
        p = os.path.join(assignments_root, f"sorted_umi_seq_assignments{amp_ind}.txt")
        if os.path.exists(p):
            assign_paths.append((amp_ind, p))

    if not assign_paths:
        raise FileNotFoundError(
            f"No sorted_umi_seq_assignments*.txt found under: {assignments_root}"
        )

    # Auto mode: include sequences only when sequences are present AND STAR doesn't appear to have run.
    include_sequences_effective = bool(include_sequences) if include_sequences is not None else None
    if include_sequences_effective is None:
        star_performed = False
        # Prefer a filesystem signal for STAR having run (STARalignment*/ dirs are created by dnamicOps).
        for _amp, _ap in assign_paths:
            if os.path.isdir(os.path.join(assignments_root, f"STARalignment{_amp}")):
                star_performed = True
                break

        has_seq_col = False
        n_seen = 0
        for _amp, ap in assign_paths:
            try:
                with open(ap, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        parts = line.rstrip("\n").split(",", 9)
                        if len(parts) < 10:
                            continue
                        aln_start_str = parts[2].strip()
                        if aln_start_str not in ("", "-1") and (not _is_placeholder(aln_start_str)):
                            try:
                                if int(aln_start_str) >= 0:
                                    star_performed = True
                            except ValueError:
                                pass

                        rest = parts[9]
                        comma = rest.find(",")
                        if comma != -1:
                            # There is an extra column after qname (the insert/consensus sequence).
                            seq_candidate = rest[comma + 1 :].strip()
                            if _looks_like_dna_seq_list(seq_candidate):
                                has_seq_col = True

                        n_seen += 1
                        if (star_performed and has_seq_col) or (n_seen >= probe_lines):
                            break
            except OSError:
                continue

            if (star_performed and has_seq_col) or (n_seen >= probe_lines):
                break

        include_sequences_effective = (not star_performed) and has_seq_col

    # --- First pass: build mapping from (umi_type, raw_index) -> row
    type_raw_to_row = {}
    umi_type_list = []
    raw_idx_list = []
    has_label_list = []
    n_subclusters_list = []
    total_sub_reads_list = []

    # Optional raw string accumulation (only if requested)
    if include_obs_strings:
        aln_start_strs = []
        aln_mut_strs = []
        contig_strs = []
        gene_strs = []
        biotype_strs = []
        tx_strs = []
        sub_reads_strs = []

    def _new_row(amp_ind: int, raw_idx: int) -> int:
        row = len(umi_type_list)
        type_raw_to_row[(amp_ind, raw_idx)] = row
        umi_type_list.append(int(amp_ind))
        raw_idx_list.append(int(raw_idx))
        has_label_list.append(True)
        n_subclusters_list.append(0)
        total_sub_reads_list.append(0)
        if include_obs_strings:
            aln_start_strs.append([])
            aln_mut_strs.append([])
            contig_strs.append([])
            gene_strs.append([])
            biotype_strs.append([])
            tx_strs.append([])
            sub_reads_strs.append([])
        return row

    # --- Second pass: build sparse matrix triplets + (optional) sequence layer triplets
    gene_to_col = {}
    gene_names = []
    gene_ids = []
    gene_contigs = []
    gene_biotype_support = {}  # feature -> {biotype -> support}

    rows_g = []
    cols_g = []
    data_g = []

    seq_to_col = {}
    seq_names = []
    rows_s = []
    cols_s = []

    for amp_ind, ap in assign_paths:
        with open(ap, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                # Parse columns without unnecessarily copying the (potentially very long) sequence field
                parts = line.rstrip("\n").split(",", 9)
                if len(parts) < 10:
                    continue

                aln_start_str = parts[2].strip()
                aln_mut_str = parts[3].strip()
                contig_str = parts[4].strip()
                gene_str = parts[5].strip()
                biotype_str = parts[6].strip()
                tx_str = parts[7].strip()
                sub_reads_str = parts[8].strip()

                rest = parts[9].strip()
                comma = rest.find(",")
                if comma != -1:
                    qname = rest[:comma].strip()
                    seq = rest[comma + 1 :].strip() if include_sequences_effective else ""
                else:
                    qname = rest
                    seq = ""

                if qname == "":
                    continue

                # Extract raw UMI cluster index from qname (format: UMIIdx.subreads.subcluster[:...])
                q0 = qname.split(":", 1)[0]
                raw_idx_str = q0.split(".", 1)[0]
                try:
                    raw_idx = int(raw_idx_str)
                except ValueError:
                    continue

                key = (amp_ind, raw_idx)
                row = type_raw_to_row.get(key)
                if row is None:
                    row = _new_row(amp_ind, raw_idx)

                # Subcluster / reads bookkeeping mirrors build_umi_gene_anndata:
                if gene_str != "":
                    n_subclusters_list[row] += 1
                sub_reads_i = _parse_subcluster_reads(sub_reads_str)
                total_sub_reads_list[row] += int(sub_reads_i)

                if include_obs_strings:
                    aln_start_strs[row].append(aln_start_str)
                    aln_mut_strs[row].append(aln_mut_str)
                    contig_strs[row].append(contig_str)
                    gene_strs[row].append(gene_str)
                    biotype_strs[row].append(biotype_str)
                    tx_strs[row].append(tx_str)
                    sub_reads_strs[row].append(sub_reads_str)

                # Sequence layer (independent of gene ambiguity filtering)
                if include_sequences_effective and seq and _looks_like_dna_seq_list(seq):
                    scol = seq_to_col.get(seq)
                    if scol is None:
                        scol = len(seq_names)
                        seq_to_col[seq] = scol
                        seq_names.append(seq)
                    rows_s.append(row)
                    cols_s.append(scol)

                # Gene parsing for this subcluster
                gene_tokens = _split_tokens(gene_str, "|")
                bt_tokens = _split_tokens(biotype_str, "|")
                ct_tokens = _split_tokens(contig_str, "|")

                feat_bt_pairs = []
                seen_feats = set()
                for j, g in enumerate(gene_tokens):
                    if not g:
                        continue
                    if drop_genes and g in drop_genes:
                        continue

                    bt_j = _pick_token(bt_tokens, j).strip() if bt_tokens else ""
                    ct_j = _pick_token(ct_tokens, j).strip() if ct_tokens else ""

                    feat = g
                    bt_use = bt_j if bt_j else "gene"

                    # Placeholder gene IDs with biotype=genome get converted into a pseudo genome feature.
                    if _is_placeholder(g) and (bt_j == "" or bt_j.lower() == "genome"):
                        contig_label = ct_j
                        if _is_placeholder(contig_label):
                            contig_label = "unknown"
                        feat = GENOME_PREFIX + contig_label
                        bt_use = "genome"

                    if feat in seen_feats:
                        continue
                    seen_feats.add(feat)
                    feat_bt_pairs.append((feat, bt_use))

                if (not include_nonunique_genes) and (len(feat_bt_pairs) != 1):
                    continue

                for feat, bt_use in feat_bt_pairs:
                    col = gene_to_col.get(feat)
                    if col is None:
                        col = len(gene_names)
                        gene_to_col[feat] = col
                        gene_names.append(feat)
                        if feat.startswith(GENOME_PREFIX):
                            gene_ids.append("")
                            gene_contigs.append(feat[len(GENOME_PREFIX) :])
                        else:
                            gene_ids.append(feat)
                            gene_contigs.append("")
                        gene_biotype_support[feat] = {}

                    gene_biotype_support[feat][bt_use] = gene_biotype_support[feat].get(bt_use, 0) + int(sub_reads_i)
                    rows_g.append(row)
                    cols_g.append(col)
                    data_g.append(1 if binary else int(sub_reads_i))

    n_nodes = len(umi_type_list)
    n_genes = len(gene_names)
    n_seqs = len(seq_names) if include_sequences_effective else 0
    n_vars = n_genes + n_seqs

    # Choose a compact integer dtype for gene counts
    max_c = max(data_g) if data_g else 0
    if binary:
        x_dtype = np.uint8
    elif max_c <= np.iinfo(np.uint8).max:
        x_dtype = np.uint8
    elif max_c <= np.iinfo(np.uint16).max:
        x_dtype = np.uint16
    elif max_c <= np.iinfo(np.uint32).max:
        x_dtype = np.uint32
    else:
        x_dtype = np.int64

    # Gene matrix (X): gene columns first; seq columns reserved (zeros)
    X_gene = coo_matrix(
        (np.asarray(data_g, dtype=x_dtype),
         (np.asarray(rows_g, dtype=np.int64), np.asarray(cols_g, dtype=np.int64))),
        shape=(n_nodes, n_vars),
        dtype=x_dtype,
    ).tocsr()
    X_gene.sum_duplicates()
    if binary and X_gene.nnz:
        X_gene.data[:] = 1

    # Sequence layer: one-hot in the *sequence columns* (offset by n_genes)
    X_seq = None
    if include_sequences_effective and n_seqs and rows_s:
        X_seq = coo_matrix(
            (np.ones(len(rows_s), dtype=np.uint8),
             (np.asarray(rows_s, dtype=np.int64), np.asarray(cols_s, dtype=np.int64) + n_genes)),
            shape=(n_nodes, n_vars),
            dtype=np.uint8,
        ).tocsr()
        X_seq.sum_duplicates()
        if X_seq.nnz:
            X_seq.data[:] = 1

    # Feature types: pick the best-supported biotype per feature (read-weighted)
    gene_feature_types = []
    for feat in gene_names:
        if feat.startswith(GENOME_PREFIX):
            gene_feature_types.append("genome")
            continue
        bt_support = gene_biotype_support.get(feat, {})
        gene_feature_types.append(_pick_biotype(bt_support))

    var_names = list(gene_names)
    feature_types = list(gene_feature_types)
    var_gene_ids = list(gene_ids)
    var_contigs = list(gene_contigs)
    var_sequences = [""] * n_genes

    if include_sequences_effective and n_seqs:
        var_names += [f"SEQ:{s}" for s in seq_names]
        feature_types += ["sequence"] * n_seqs
        var_gene_ids += [""] * n_seqs
        var_contigs += [""] * n_seqs
        var_sequences += list(seq_names)

    var_df = pd.DataFrame(
        {
            "feature_type": feature_types,
            "gene_id": var_gene_ids,
            "contig": var_contigs,
            "sequence": var_sequences,
        },
        index=pd.Index(var_names, name="feature"),
    )

    obs = {
        "umi_type": np.asarray(umi_type_list, dtype=np.int8),
        "raw_umi_index": np.asarray(raw_idx_list, dtype=np.int64),
        "has_label": np.asarray(has_label_list, dtype=bool),
        "n_subclusters": np.asarray(n_subclusters_list, dtype=np.int32),
        "n_annotated": np.asarray(np.diff(X_gene.indptr), dtype=np.int32),
        "total_sub_reads": np.asarray(total_sub_reads_list, dtype=np.int32),
    }

    if include_obs_strings:
        obs["aln_start_str"] = np.asarray([";".join(v) for v in aln_start_strs], dtype=object)
        obs["aln_mut_str"] = np.asarray([";".join(v) for v in aln_mut_strs], dtype=object)
        obs["contig_str"] = np.asarray([";".join(v) for v in contig_strs], dtype=object)
        obs["gene_str"] = np.asarray([";".join(v) for v in gene_strs], dtype=object)
        obs["biotype_str"] = np.asarray([";".join(v) for v in biotype_strs], dtype=object)
        obs["tx_str"] = np.asarray([";".join(v) for v in tx_strs], dtype=object)
        obs["sub_reads_str"] = np.asarray([";".join(v) for v in sub_reads_strs], dtype=object)

    # If sequences were included, store a per-UMI convenience string (unique seqs) like build_umi_gene_anndata
    if include_sequences_effective and X_seq is not None:
        obs["n_seqs"] = np.asarray(np.diff(X_seq.indptr), dtype=np.int32)
        seq_str = np.empty(n_nodes, dtype=object)
        for r in range(n_nodes):
            s0, s1 = X_seq.indptr[r], X_seq.indptr[r + 1]
            cols = X_seq.indices[s0:s1]
            toks = [seq_names[c - n_genes] for c in cols if c >= n_genes]
            seq_str[r] = ";".join(toks)
        obs["seq_str"] = seq_str
        obs["qname_or_seq_str"] = seq_str

    obs_df = pd.DataFrame(obs, index=pd.Index(np.arange(n_nodes, dtype=np.int32), name="node_id"))

    if not return_anndata:
        # Mirror build_umi_gene_anndata's non-anndata return: (X, feature_names, obs_df)
        return X_gene, gene_names, obs_df

    try:
        from anndata import AnnData
    except ImportError:
        # Fallback
        return X_gene, gene_names, obs_df

    # Make index types explicit (avoid ImplicitModificationWarning and ensure parity)
    obs_df.index = obs_df.index.astype(str)
    var_df.index = var_df.index.astype(str)
    adata = AnnData(X=X_gene, obs=obs_df, var=var_df)
    if X_seq is not None:
        adata.layers["seq"] = X_seq

    adata.uns["assignments_paths"] = [p for (_amp, p) in assign_paths]
    adata.uns["include_nonunique_genes"] = bool(include_nonunique_genes)
    adata.uns["include_sequences"] = bool(include_sequences_effective)
    adata.uns["include_obs_strings"] = bool(include_obs_strings)
    adata.uns["binary"] = bool(binary)
    adata.uns["genome_feature_prefix"] = GENOME_PREFIX

    return adata
