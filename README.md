# xVDM

This repository contains the analysis code for cross-linked volumetric DNA microscopy (xVDM). xVDM converts paired cDNA and UEI sequencing libraries into a graph of molecular proximity, embeds the graph with Geodesic Spectral Embedding (GSE), calls connectivity-derived cells, and can register aggregate cells to a matched spatial reference.

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

This stage reads `path/to/library/lib.settings`, filters reads by the settings, clusters UMI and UEI sequences, builds consensus pairings, and writes inference files under the library directory. UEI libraries produce `uei_grp*/` directories that contain the graph used by GSE. cDNA libraries produce label and assignment files used for gene or sequence annotation.

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

## Outputs to check

After a successful run, start with `statuslog.csv` for progress and errors. In a UEI group directory, `link_assoc_reindexed.npz`, `GSEoutput.txt`, `cluster_labels.npy`, and `final.h5ad` are the most useful downstream files. In registration mode, inspect `final_coarsening/` and its `match_result_*` subdirectory.

## Notes

GSE coordinates are determined up to translation, rotation, and reflection. Compare reconstructions by aligning them to a reference or to ground truth. The `lib` stage is designed to resume from existing outputs; remove an output directory before a fully fresh rerun.
