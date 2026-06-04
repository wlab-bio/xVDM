# xVDM

This repository contains the analysis code for cross-linked volumetric DNA microscopy (xVDM). The pipeline builds on an earlier pipeline that you can find <a href="https://github.com/wlab-bio/vdnamic">here</a>. xVDM converts paired cDNA and UEI sequencing libraries into a graph of molecular proximity, embeds the graph with Geodesic Spectral Embedding (GSE), calls connectivity-derived cells, and can register aggregate cells to a matched spatial reference.

## What is in the repository

`main.py` is the supported entry point. It dispatches `lib` runs to library processing and `GSE` runs to embedding, final clustering, AnnData export, and optional coarsened registration. The test fixture in `sample/test_directory` contains a small read simulator, settings files, a `pos.csv` ground truth, a plotting script, and `run.sh`.

## Install

Start from the environment file supplied with this repository. A working environment needs Python plus the scientific Python stack used by the scripts, `bioawk`, GNU `sort`, `gzip`, and Infomap. STAR is needed only when `lib.settings` asks for genome alignment. OR-Tools is needed only for `register_zf` reference registration.

## Quick test

From the repository root:

```bash
cd sample/test_directory
sh run.sh
```

The script simulates paired UEI and cDNA FASTQs from `pos.csv`, copies `uei.lib.settings` and `cdna.lib.settings` into the generated library directories, runs the `lib` stage for each library, runs a 2D GSE reconstruction with sequence features included, and writes `reconstruction.png`.

For a clean rerun, remove `sample/test_directory/sim_fastq` and any generated FASTQ files.

## Input format

Each library directory must contain `lib.settings`. Use `-source_for` and `-source_rev` to point to paired FASTQ files. Use `-seqform_for` and `-seqform_rev` to describe how primers, UMIs, UEIs, and cDNA inserts appear in the reads. Use `-u0`, `-u1`, and `-u2` to assign UMI and UEI fields. Use `-a0` and `-a1` for cDNA inserts when present.

For paired xVDM runs, process the UEI library first. Then process the cDNA library with `-uei_matchfilepath` pointing to the UEI output so cDNA UMIs can be matched back to UEI graph nodes.

## Run library processing

```bash
python main.py lib path/to/library/
```

This stage reads `path/to/library/lib.settings`, filters reads by the settings, clusters UMI and UEI sequences, builds consensus pairings, and writes inference files under the library directory. UEI libraries produce `uei_grp*/` directories that contain the graph used by GSE. cDNA libraries produce label and assignment files used for gene or sequence annotation. cDNA libraries also write `cdna_only.h5ad` when assignment files are available.

## Run GSE without registration

```bash
python main.py GSE \
  -path path/to/uei_grp0/. \
  -inference_dim 2 \
  -inference_eignum 15 \
  -final_eignum 50 \
  -calc_final ../. \
  -h5ad_include_sequences
```

This stage writes `GSEoutput.txt`, final cluster labels, and, when `-calc_final` points to a valid label root, `final.h5ad`. Use `-h5ad_include_sequences` for simulation and other benchmark runs where cDNA insert sequences should be retained as sequence features.

## Run GSE with final coarsening and reference registration

```bash
REGISTER_ZF_ENSEMBLE_N_JOBS=4 \
REGISTER_ZF_ENSEMBLE_THREADS_PER_WORKER=1 \
python main.py GSE \
  -path path/to/uei_grp0/. \
  -inference_dim 3 \
  -inference_eignum 30 \
  -final_eignum 100 \
  -calc_final ../. \
  -coarsen_infomap \
  -slice_path path/to/reference_slice.h5ad \
  -register_zf 18hpf
```

Use `-register_zf` with the stage key expected by the zebrafish registration code, such as `12hpf`, `18hpf`, or `24hpf`. The registration route requires `-coarsen_infomap`, `-slice_path`, and `-calc_final`. It runs after the ordinary GSE output and final Infomap clustering, builds a coarsened aggregate AnnData object, and maps aggregate nodes to slice coordinates. It leaves the main `GSEoutput.txt` route unchanged.

Some legacy run wrappers pass `-register_zf_ensemble_n_jobs` and `-register_zf_ensemble_threads_per_worker` as CLI flags. The current code resolves those two runtime controls from the environment variables shown above.

## Simulation helper

`vdnamic_fastq_sim.py` can build a graph from a position CSV and emit paired FASTQ files for both libraries:

```bash
python vdnamic_fastq_sim.py \
  --build-from-posfile pos.csv \
  --cdna-settings cdna.lib.settings \
  --uei-settings uei.lib.settings \
  --avg-reads-uei 4.5 \
  --min-reads-uei 1 \
  --use-uei-weights \
  --cdna-secondary-insert \
  -o sim_fastq
```

`pos.csv` must contain numeric rows with `id,label,x[,y...]`. The simulator encodes point identity into base-4 cDNA inserts so `plot.py` can compare the reconstruction with the known positions.

## Plot the sample reconstruction

```bash
python plot.py \
  --group_dir sim_fastq/uei/uei_grp0 \
  --pos pos.csv \
  --out reconstruction.png
```

The plot script reads sequences from `final.h5ad` when possible and reads coordinates from `obs['GSE_1']`, `obs['GSE_2']`, or a nearby `GSEoutput*.txt`. It decodes base-4 sequence labels and aligns inferred positions to `pos.csv` by Procrustes.

## AnnData output files

The pipeline writes sparse AnnData objects for inspection in Scanpy and for downstream registration. These files are not normalized single-cell objects. Rows are molecular nodes or cDNA UMI clusters. Columns are gene features, contig features, and, only when requested or automatically allowed, raw sequence features. `adata.X` stores integer sub-consensus read support unless a helper was called in binary mode.

### `final.h5ad`

`final.h5ad` is written inside a UEI group directory, such as `uei_grp0/final.h5ad`, when GSE is run with `-calc_final`. The file joins three kinds of information: the reindexed UEI graph nodes from `index_key.npy`, cDNA labels found through `label_pt0.txt` and `label_pt1.txt`, and the final GSE and clustering outputs.

Rows follow the GSE node order. The AnnData index is the node id as a string. The main observation columns are `umi_type`, `raw_umi_index`, `has_label`, `n_subclusters`, `n_annotated`, and `total_sub_reads`. When the matched label files include UEI read support, `total_uei_reads` is also present. A completed GSE run adds `GSE_1`, `GSE_2`, and further `GSE_i` columns as needed, and also stores the same coordinate matrix in `adata.obsm['X_gse']`.

Cluster labels are copied from `cluster_labels.npy` when available. The usual current layout has `cluster_hdbscan` and `cluster_infomap`. Older or experimental layouts may instead expose `cluster`, `cluster_0`, or additional `cluster_*` columns. The label source, shape, and method names are recorded in `adata.uns`.

Features live in `adata.var`. Gene features use their gene name as the index. Unannotated genome alignments become contig features named `__genome__:<contig>`. Sequence features, when included, are named `SEQ:<sequence>`. The feature metadata columns are `feature_type`, `gene_id`, `contig`, and `sequence`. Multi-gene sub-consensus calls are dropped by default, and repeated calls to the same feature within one hub are collapsed before matrix construction.

`adata.X` is a sparse node-by-feature count matrix. Gene and contig entries are sub-consensus read support. If sequence features are present, `adata.layers['seq']` stores a sparse one-hot sequence layer in the same feature space. `adata.uns` records the label paths, the index-key path, whether sequences and nonunique genes were included, the genome pseudo-feature prefix, and the h5ad build stage.

Under STAR alignment, cDNA sub-consensuses are aligned before the label files are built. The GTF overlap supplies contig, gene, biotype, and transcript fields. rRNA and mitochondrial rRNA calls are prioritized and appear as `rRNA` or `Mt_rRNA`. Genome-only hits remain as `__genome__:<contig>` features. In automatic mode, sequence features are not added when STAR output or nonnegative alignment starts indicate that genome alignment was performed. This keeps STAR-aligned `final.h5ad` files focused on gene and contig counts. Use `-h5ad_include_sequences` only for simulation or benchmark runs where raw insert strings are needed.

### `cdna_only.h5ad`

`cdna_only.h5ad` is written in the cDNA library directory during `python main.py lib path/to/cdna/`. It is built directly from `sorted_umi_seq_assignments0.txt` and `sorted_umi_seq_assignments1.txt`, before UEI matching and before GSE. It is useful for checking cDNA consensus, STAR annotation, hub composition, and gene recovery without asking whether a cDNA UMI entered the UEI graph.

Rows are cDNA UMI clusters keyed by amplicon side and raw UMI index. The standard observation columns are `umi_type`, `raw_umi_index`, `has_label`, `n_subclusters`, `n_annotated`, and `total_sub_reads`. It does not contain `total_uei_reads`, `GSE_*`, `X_gse`, or cluster labels, because it has no UEI graph geometry.

The feature table and count matrix use the same conventions as `final.h5ad`: gene names, `__genome__:<contig>` pseudo-features, optional `SEQ:<sequence>` features, `feature_type`, `gene_id`, `contig`, and `sequence` metadata, and sparse sub-consensus read counts in `adata.X`. If sequences are included, `adata.layers['seq']`, `obs['n_seqs']`, `obs['seq_str']`, and `obs['qname_or_seq_str']` are written.

For STAR-aligned cDNA libraries, the default `cdna_only.h5ad` builder runs in automatic sequence mode. It detects STAR either from `STARalignment*` directories or from nonnegative alignment starts in the assignment files. If STAR is detected, raw sequence features are omitted even when a sequence column exists. The resulting file is therefore the practical cDNA UMI-by-gene matrix for the aligned library, with contig-only genome calls kept separate from gene calls.

### Reading the files

```python
import scanpy as sc

adata = sc.read_h5ad("sim_fastq/uei/uei_grp0/final.h5ad")
coords = adata.obsm.get("X_gse")
counts = adata.X
features = adata.var
nodes = adata.obs
```

For registration outputs, also inspect `final_coarsening/component0/final.h5ad`. That object is built from the fine-node `final.h5ad` after final Infomap coarsening. Rows are aggregate nodes, `obs['n_fine_nodes']` records their size, and the count matrix is the coarsened annotation matrix used by `register_zf`.

## Outputs to check

After a successful run, start with `statuslog.csv` for progress and errors. In a UEI group directory, `link_assoc_reindexed.npz`, `GSEoutput.txt`, `cluster_labels.npy`, and `final.h5ad` are the most useful downstream files. In a cDNA directory, check `sorted_umi_seq_assignments*.txt`, `label_pt*.txt`, and `cdna_only.h5ad`. In registration mode, inspect `final_coarsening/` and its `match_result_*` subdirectory.

## Notes

GSE coordinates are determined up to translation, rotation, and reflection. Compare reconstructions by aligning them to a reference or to ground truth. The `lib` stage is designed to resume from existing outputs; remove an output directory before a fully fresh rerun.
