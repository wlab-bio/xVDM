#!/usr/bin/env python3
# PATCH_VERSION: ann12_cell_slice_camera_framing_fix_2026-05-25
"""Focused Leiden + PyVista arc/surface pipeline.

Accepted command forms::

    python -u ann12_new3a.py --file-paths STR_1 ... STR_N --slice-paths spatial_sixtime_slice_stereoseq.h5ad --coarsen-align-arc-arc-alpha FLOAT --coarsen-align-arc-line-width FLOAT --coarsen-align-arc-min-opacity FLOAT --coarsen-align-arc-frontal-view-bias FLOAT --fig-dir arcs_envelopes/

    python -u ann12_new3a.py \
      --fig-dir STR \
      --replot-existing-surfaces-only \
      --replot-sample-k 2 \
      --coarsened-graph-dir arcs_envelopes/sample_cell_connectivity_h5ad

The normal route aggregates hub-level H5AD files into component-split cells,
performs QC, downsampling, PCA/UMAP, Leiden clustering, marker-module mapping,
Leiden-colored PyVista rendering, cell-coarsened graph export, and coarsen-align
cell-to-slice arc rendering.  The second route only rebuilds/renders existing
cell-surface outputs for one manifest sample and updates the matching
sample_cell_connectivity_h5ad surface metrics.
"""

from __future__ import annotations

import argparse
import atexit
import gc
import hashlib
import json
import os
import shutil
import subprocess
import textwrap
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import anndata as ad
import h5py
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.sparse.csgraph import connected_components as sparse_connected_components
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from scipy.spatial.distance import pdist

# Optional 3D rendering (PyVista/VTK)
try:
    import pyvista as pv
except Exception:  # pragma: no cover
    pv = None

_XVFB_PROC = None

# Filled by render_gse_pyvista_snapshots_for_all_samples and consumed by the
# cell-surface renderer. This removes the old RNG-lockstep replay path.
GSE_LEIDEN_FOREGROUND_CLUSTER_IDS_BY_SAMPLE: Dict[str, Dict[str, object]] = {}


def _log(msg: str) -> None:
    print(str(msg), flush=True)


# =============================================================================
# Configuration retained for the FigureS2 + pyvista_gse_scatter_leiden routes
# =============================================================================


FIG_DIR = "figures_multipanel_downsampled_"
TABLE_DIR = os.path.join(FIG_DIR, "tables")

MIN_READS = 1
CLUSTER_KEY = "cluster_infomap"
DROP_CLUSTER_VALUES = {"-1", "", "nan", "None", "NA", "N/A"}
RANDOM_SEED = 0

# Targeted cell-tabulation patches.  Hub clusters are first split
# into disconnected components using link_assoc_reindexed.npz.  Each component
# becomes a candidate cell and then passes through the existing ann12 QC gates
# (QC_MIN_COUNTS / QC_MIN_GENES / QC_MIN_CELLS_PER_GENE).  A sidecar maps
# original hub rows to refined component labels so PyVista routes can color and
# surface split components rather than the original unsplit infomap label.
SPLIT_DISCONNECTED_COMPONENTS = True
CONNECTIVITY_NPZ_BASENAME = "link_assoc_reindexed.npz"
WRITE_REFINED_HUB_LABEL_SIDECAR = True
REFINED_HUB_LABEL_SIDECAR = "hub_refined_labels.tsv"

# Targeted rRNA pseudo-feature patch.  Raw rRNA/Mt-rRNA features
# are collapsed into binary per-hub pseudo-features and removed from the ordinary
# expression feature set before duplicate gene IDs are collapsed.
RRNA_PSEUDO_FEATURES_ENABLE = True
RRNA_SUM_FEATURE = "rRNA_sum"
MT_RRNA_SUM_FEATURE = "Mt_rRNA_sum"

QC_MIN_COUNTS = 100
QC_MIN_GENES = 50
QC_MIN_CELLS_PER_GENE = 3
# Rarefy the exact count quantity used by the QC_MIN_COUNTS gate: qualifying
# sub-consensus gene-call events, i.e. hub x gene entries with X >= MIN_READS
# after hub-to-cell aggregation. This removes per-cell event depth as a
# confounder before HVG/PCA/Leiden/UMAP and downstream DE.
DOWNSAMPLE_SUBCONSENSUS_GENE_CALLS = True
DOWNSAMPLE_TARGET_COUNTS_PER_CELL = QC_MIN_COUNTS
DOWNSAMPLE_SUMMARY_TABLE = "subconsensus_gene_call_downsampling_summary.tsv"
RAW_SUBCONSENSUS_COUNTS_LAYER = "subconsensus_counts_raw"
ANALYSIS_COUNTS_LAYER = "counts"
GLOBAL_N_NEIGHBORS = 30
GLOBAL_N_PCS = 30

LEIDEN_ENABLE = True
LEIDEN_KEY = "leiden_pca"
LEIDEN_RESOLUTION = 0.2
LEIDEN_RANDOM_STATE = RANDOM_SEED
LEIDEN_AUTO_RESOLUTION = False
LEIDEN_MAX_DISTINCT_COLORS = 50
LEIDEN_MIN_CLUSTERS = 8
LEIDEN_MIN_CLUSTER_SIZE = 100
LEIDEN_RESOLUTION_GRID = np.round(np.arange(0.1, 2.05, 0.1), 2)
LEIDEN_RESOLUTION_SEARCH_TABLE = "leiden_resolution_search.tsv"

DE_ENABLE = True
DE_TOP_N_CLUSTERS = 0  # 0 means all clusters
DE_CELLS_PER_SAMPLE = 10
DE_MIN_SAMPLES = 2
DE_TARGET_COUNTS_PER_CELL = QC_MIN_COUNTS
DE_PSEUDOCOUNT = 0.5
# Balanced-DE p-values are inferential only when there are multiple biological
# samples/specimens.  For a one-sample run whose goal is simply to identify the
# marker-expression module associated with each Leiden cluster, use a descriptive
# cell-level cluster-vs-rest summary for the marker map instead of suppressing it.
MARKER_MAP_SINGLE_SAMPLE_DESCRIPTIVE = True
MARKER_MAP_DESCRIPTIVE_STATUS = "ok_marker_map_descriptive_cellwise"
MARKER_MAP_DESCRIPTIVE_FDR_WEIGHT = 1.0
DE_FDR_ALPHA = 0.05
DE_TABLE_ALL = "leiden_DE_balanced_all_genes.tsv"
DE_TABLE_SUMMARY = "leiden_DE_balanced_sampling_summary.tsv"
MARKER_MAP_FIG = "FigureS2_leiden_expression_cluster_map.pdf"
MARKER_MAP_TABLE_SELECTED_GENES = "leiden_marker_expression_selected_genes.tsv"
MARKER_MAP_TABLE_MODULES = "leiden_expression_gene_modules.tsv"
MARKER_MAP_TABLE_MODULE_BY_LEIDEN = "leiden_expression_gene_module_by_leiden.tsv"
PCA_FIG = "FigureS2_leiden_PCA_colored.pdf"
PCA_FIG_PNG = "FigureS2_leiden_PCA_colored.png"
UMAP_FIG = "FigureS2_leiden_UMAP_colored.pdf"
UMAP_FIG_PNG = "FigureS2_leiden_UMAP_colored.png"
UMAP_AGE_FIG = "FigureS2_leiden_UMAP_by_age.pdf"
UMAP_AGE_FIG_PNG = "FigureS2_leiden_UMAP_by_age.png"
UMAP_SUBCONS_FIG = "FigureS2_leiden_UMAP_by_subconsensus_count.pdf"
UMAP_SUBCONS_FIG_PNG = "FigureS2_leiden_UMAP_by_subconsensus_count.png"
ANALYSIS_MANIFEST = "leiden_de_analysis_manifest.json"

# Main DE figure: map Leiden clusters to expression modules instead of
# avoiding dense all-label displays.
MARKER_MAP_ENABLE = True
MARKER_MAP_TOP_GENES_PER_LEIDEN = 12
MARKER_MAP_MAX_GENES = 160
MARKER_MAP_MIN_LOG2FC = 0.75
MARKER_MAP_MIN_PCT_IN = 0.05
MARKER_MAP_MIN_PCT_DELTA = 0.03
MARKER_MAP_FDR_ALPHA = DE_FDR_ALPHA
MARKER_MAP_FALLBACK_FDR = 0.25
MARKER_MAP_FALLBACK_MIN_LOG2FC = 0.25
MARKER_MAP_FALLBACK_MIN_PCT_IN = 0.02
MARKER_MAP_MAX_MODULES = 12
MARKER_MAP_MIN_MODULES = 4
MARKER_MAP_TARGET_GENES_PER_MODULE = 12
MARKER_MAP_TOP_GENES_IN_MODULE_LABEL = 4
MARKER_MAP_Z_CLIP = 2.5
MARKER_MAP_DOT_MIN_SIZE = 18.0
MARKER_MAP_DOT_MAX_SIZE = 260.0
MARKER_MAP_MIN_FIG_WIDTH = 7.6
MARKER_MAP_MAX_FIG_WIDTH = 10.6
MARKER_MAP_CLUSTER_WIDTH = 0.40
MARKER_MAP_ROW_HEIGHT = 0.82
MARKER_MAP_BASE_HEIGHT = 2.3
MARKER_MAP_LABEL_WRAP = 28
MARKER_MAP_XTICK_FONTSIZE = 8.0
MARKER_MAP_YTICK_FONTSIZE = 8.2
MARKER_MAP_EXCLUDE_FEATURES = {"rRNA_sum", "Mt_rRNA_sum"}

# PCA/UMAP figure settings.
PCA_POINT_SIZE = 11.0
PCA_ALPHA = 0.72
PCA_LABEL_FONTSIZE = 13.0
PCA_SHOW_CENTROID_LABELS = True
UMAP_POINT_SIZE = PCA_POINT_SIZE
UMAP_ALPHA = PCA_ALPHA
UMAP_LABEL_FONTSIZE = PCA_LABEL_FONTSIZE
UMAP_SHOW_CENTROID_LABELS = True
UMAP_RANDOM_STATE = RANDOM_SEED
AGE_BY_SAMPLE = {
    "zf1": "12hpf",
    "zf2": "12hpf",
    "zf3": "18hpf",
    "zf4": "18hpf",
    "zf5": "18hpf",
    "zf6": "18hpf",
    "zf7": "24hpf",
    "zf8": "24hpf",
}
AGE_ORDER = ["12hpf", "18hpf", "24hpf"]
AGE_PALETTE = {
    "12hpf": "#3b82f6",
    "18hpf": "#10b981",
    "24hpf": "#f59e0b",
}
UMAP_CONTINUOUS_CMAP = "viridis"

# Palette caching. The same palette object is passed to marker-map and PyVista outputs.
PALETTE_FORCE_REGENERATE = True

# =============================================================================
# Cell-filtering diagnostic settings
# =============================================================================
# These control the diagnostic that exposes WHY data is rejected as it moves
# from raw hubs through to QC-passed cells. The pipeline rejects data at six
# distinct gates:
#   (G1) invalid features (gene_id) dropped from var
#   (G2) hubs with NaN cluster or cluster in DROP_CLUSTER_VALUES dropped
#   (G3) hubs with zero per-entry calls >= MIN_READS contribute nothing
#   (G4) cells with total_counts < QC_MIN_COUNTS dropped
#   (G5) cells with n_genes_by_counts < QC_MIN_GENES dropped
#   (G6) genes detected in < QC_MIN_CELLS_PER_GENE cells dropped
# The diagnostic produces a per-stage summary TSV plus rank-order plots that
# make the "knee" of each filter, and the fraction it rejects, immediately
# visible.
DIAG_ENABLE = True
DIAG_DIR_NAME = "filtering_diagnostics"
DIAG_TABLE_PER_STAGE = "filtering_per_stage_summary.tsv"
DIAG_TABLE_PER_CELL = "filtering_per_cell_metrics.tsv"
DIAG_TABLE_PER_GENE = "filtering_per_gene_metrics.tsv"
DIAG_TABLE_PER_HUB = "filtering_per_hub_metrics.tsv"
DIAG_FIG_RANK_HUB_CALLS = "filtering_rank_hub_calls_per_hub.pdf"
DIAG_FIG_RANK_CELL_HUBS = "filtering_rank_hubs_per_cell.pdf"
DIAG_FIG_RANK_CELL_COUNTS = "filtering_rank_total_counts_per_cell.pdf"
DIAG_FIG_RANK_CELL_GENES = "filtering_rank_n_genes_per_cell.pdf"
DIAG_FIG_RANK_GENE_CELLS = "filtering_rank_n_cells_per_gene.pdf"
DIAG_FIG_FUNNEL = "filtering_waterfall_funnel.pdf"
DIAG_FIG_JOINT = "filtering_joint_counts_vs_genes.pdf"

# PyVista/VTK 3D scatter settings.
GSE_RENDER_ENABLE = True
GSE_COORD_KEYS = ("GSE_1", "GSE_2", "GSE_3")
GSE_PNG_DIR = os.path.join(FIG_DIR, "pyvista_gse_scatter")
GSE_WINDOW_SIZE = (1600, 1200)
GSE_POINT_SIZE = 3.0
GSE_BACKGROUND = "#000000"
GSE_STATS_SAMPLE_N = 200_000
GSE_INFRAME_FRACTION = 0.99
GSE_INFRAME_NEAR_QUANTILE = 0.01
GSE_FRAMING_USE_ASSIGNED_POINTS = True
GSE_FRAMING_MIN_ASSIGNED_POINTS = 5_000
GSE_RENDER_ONLY_ASSIGNED = False
GSE_CAMERA_PARALLEL_PROJECTION = False
GSE_CAMERA_VIEW_ANGLE_DEG = 25.0
GSE_BACKGROUND_TOP = None
GSE_RENDER_POINTS_AS_SPHERES = False
GSE_MIN_POINT_SIZE_FOR_SPHERES = 2.0
GSE_ENABLE_LIGHTKIT = False
GSE_LIGHTING = False
GSE_MATERIAL_AMBIENT = 0.15
GSE_MATERIAL_DIFFUSE = 0.85
GSE_MATERIAL_SPECULAR = 0.25
GSE_MATERIAL_SPECULAR_POWER = 30.0
GSE_ENABLE_EYE_DOME_LIGHTING = True
GSE_EDL_STRENGTH = 0.35
GSE_EDL_RADIUS = 0.7
GSE_ENABLE_SSAO = False
GSE_SSAO_RADIUS = 0.35
GSE_SSAO_BIAS = 0.02
GSE_SSAO_KERNEL_SIZE = 32
GSE_ANTIALIASING = "ssaa"
GSE_SCREENSHOT_SCALE = 2
GSE_AUTO_OPACITY = True
GSE_ASSIGNED_OPACITY_AT_1M = 0.25
GSE_OPACITY_SCALING_EXPONENT = 0.5
GSE_ASSIGNED_OPACITY_MIN = 0.020
GSE_ASSIGNED_OPACITY_MAX = 0.450
GSE_UNASSIGNED_OPACITY_FRACTION = 0.015
GSE_UNASSIGNED_OPACITY_MIN = 0.0
GSE_UNASSIGNED_OPACITY_MAX = 0.050
GSE_ASSIGNED_OPACITY = 0.25
GSE_UNASSIGNED_OPACITY = 0.004
GSE_UNASSIGNED_RGBA = (211, 211, 211, int(round(GSE_UNASSIGNED_OPACITY * 255)))


# Cluster-balanced PyVista rendering for very dense GSE clouds.  At ~1e7
# points, faithful rendering of every point is less informative than a
# visibility-balanced rendering.  For the Leiden-colored outputs in
# pyvista_gse_scatter_leiden, label L0 is reserved as the gray palette slot but
# is not plotted.  This keeps gray impossible in both the full-Leiden and top-5
# PNGs, while all visible Leiden labels share one brightness/point-size family.
GSE_CLUSTER_BALANCED_RENDER = True
GSE_EXCLUDED_LEIDEN_LABELS: Tuple[str, ...] = ("0",)
GSE_EXCLUDED_LABEL_SENTINEL = "__excluded_l0__"
GSE_BALANCED_TOTAL_MAX_POINTS = 3_000_000       # <=0 means no global cap
GSE_MINOR_CLUSTER_MAX_POINTS = 600_000          # cap per plotted non-L0 Leiden label; <=0 means all
GSE_UNASSIGNED_CONTEXT_MAX_POINTS = 0           # unassigned gray dust is not drawn
GSE_MINOR_CLUSTER_OPACITY = 1.0
GSE_MINOR_CLUSTER_MIN_OPACITY = 0.6
GSE_MINOR_CLUSTER_POINT_SIZE = 3.0
GSE_UNASSIGNED_CONTEXT_COLOR = "#5c6470"
GSE_UNASSIGNED_CONTEXT_OPACITY = 0.0
GSE_UNASSIGNED_CONTEXT_POINT_SIZE = 0.0

# Supplemental per-sample top-5 focus renders for pyvista_gse_scatter_leiden.
# L0 and unassigned hubs are excluded exactly as in the full-Leiden render.  The
# five selected current-sample plotted labels use the exact same color, opacity,
# and point size as the full-Leiden mode.  Non-top plotted labels are shown as a
# neutral gray backdrop that uses the same opacity and point size, but the L0
# reserved palette gray itself is still never rendered.
GSE_SAMPLE_TOP_PLOTTED_CLUSTER_RENDER_ENABLE = True
GSE_SAMPLE_TOP_PLOTTED_CLUSTER_N = 5
GSE_SAMPLE_TOP_PLOTTED_CLUSTER_FILE_TAG = "top5_sample_plotted"
GSE_SAMPLE_TOP_PLOTTED_HIGHLIGHT_OPACITY = GSE_MINOR_CLUSTER_OPACITY
GSE_SAMPLE_TOP_PLOTTED_HIGHLIGHT_POINT_SIZE = GSE_MINOR_CLUSTER_POINT_SIZE
GSE_SAMPLE_TOP_PLOTTED_REMAINDER_COLOR = "#555a63"  # neutral backdrop gray; distinct from reserved L0 gray
GSE_SAMPLE_TOP_PLOTTED_REMAINDER_OPACITY = GSE_MINOR_CLUSTER_OPACITY
GSE_SAMPLE_TOP_PLOTTED_REMAINDER_POINT_SIZE = GSE_MINOR_CLUSTER_POINT_SIZE
GSE_SAMPLE_TOP_PLOTTED_REMAINDER_MAX_POINTS = 1_250_000
GSE_SAMPLE_TOP_PLOTTED_UNASSIGNED_COLOR = GSE_UNASSIGNED_CONTEXT_COLOR
GSE_SAMPLE_TOP_PLOTTED_UNASSIGNED_OPACITY = 0.0
GSE_SAMPLE_TOP_PLOTTED_UNASSIGNED_POINT_SIZE = 0.0
GSE_SAMPLE_TOP_PLOTTED_UNASSIGNED_MAX_POINTS = 0
GSE_SAMPLE_TOP_PLOTTED_SUMMARY_TABLE = "pyvista_gse_scatter_leiden_top_plotted_clusters.tsv"
GSE_ARC_OPACITY_FLOOR = 0.003


GSE_RENDER_RARE_CLUSTERS_LAST = True
GSE_VIEW_ANGLES_DEG = [
    (0.0, 20.0),
    (45.0, 20.0),
    (90.0, 20.0),
    (135.0, 20.0),
    (180.0, 20.0),
    (225.0, 20.0),
    (270.0, 20.0),
    (315.0, 20.0),
    (0.0, 60.0),
    (90.0, 60.0),
    (180.0, 60.0),
    (270.0, 60.0),
]

# Camera-space scale bar for the Leiden-colored PyVista renders. This is the
GSE_SCALE_BAR_ENABLE = True
GSE_SCALE_BAR_LENGTH = 25.0
GSE_SCALE_BAR_COLOR = "white"
GSE_SCALE_BAR_LINE_WIDTH = 4.0
GSE_SCALE_BAR_MARGIN_FRAC = (0.10, 0.10)
GSE_SCALE_BAR_ACTOR_NAME = "__gse_scale_bar__"

# Additional PyVista cell-level surface renderer.  Random-color
# infomap-cell scatter is omitted; surfaces are the cell-level
# visualization path.
# One merged colored
# surface mesh per sample, with one surface component per rendered infomap-
# aggregated cell. The renderer first restricts to post-QC/downsampled cells and
# then, by default, further restricts to the same infomap cells that appear as
# colored foreground in the Leiden PyVista balanced render. In the default
# balanced Leiden render, L0 is excluded from the scatter route, and all
# non-excluded Leiden labels are considered "colored" here.
#
# Geometry is deliberately not ellipsoidal. The default route now builds an
# adaptive PCA-normalized alpha surface from sampled per-cell hubs. This is much
# less convex than the prior support-hull approximation and can preserve lobes,
# indentations, and local roughness. Support hulls and radial witness shells are
# retained only as robust fallbacks for sparse/degenerate cells.
GSE_CELL_SURFACE_RENDER_ENABLE = True
GSE_CELL_SURFACE_PNG_DIR_NAME = "pyvista_gse_infomap_cell_surfaces"
GSE_CELL_SURFACE_CACHE_ENABLE = True
GSE_CELL_SURFACE_CACHE_DIR_NAME = "_geometry"
GSE_CELL_SURFACE_FORCE_REBUILD = False
GSE_CELL_SURFACE_SAVE_GEOMETRY = True
GSE_CELL_SURFACE_METHOD = "adaptive_alpha_surface"  # adaptive_alpha_surface | support_hull | pca_radial_witness_shell
GSE_CELL_SURFACE_CLUSTER_ID_COL = "cluster_id"
GSE_CELL_SURFACE_REQUIRE_POSTFILTER_CELLS = True
GSE_CELL_SURFACE_FILTER_TO_LEIDEN_COLORED_CELLS = True
GSE_CELL_SURFACE_LEIDEN_FILTER_LABEL_COL = LEIDEN_KEY
GSE_CELL_SURFACE_LEIDEN_FILTER_TABLE = "infomap_cell_surface_leiden_colored_cells_filter.tsv"
GSE_CELL_SURFACE_DIRECTION_COUNT = 256
GSE_CELL_SURFACE_MIN_POINTS = 1
GSE_CELL_SURFACE_MAX_POINTS_PER_CELL = 1536
GSE_CELL_SURFACE_CANDIDATE_MULTIPLIER = 6
GSE_CELL_SURFACE_BOUNDARY_SAMPLE_FRACTION = 0.55
# Adaptive alpha-surface controls. Alpha surfaces are built in a robust
# PCA-normalized local coordinate system from true sampled hub coordinates, not
# from an ellipsoid template. The radius is inferred from kNN spacing and relaxed
# only if the first alpha value produces too few boundary faces.
GSE_CELL_SURFACE_ALPHA_MIN_POINTS = 16
GSE_CELL_SURFACE_ALPHA_MAX_POINTS = 1024
GSE_CELL_SURFACE_ALPHA_KNN_K = 8
GSE_CELL_SURFACE_ALPHA_KNN_QUANTILE = 0.65
GSE_CELL_SURFACE_ALPHA_RADIUS_MULTIPLIER = 2.20
GSE_CELL_SURFACE_ALPHA_MIN_RELATIVE_RADIUS = 0.035
GSE_CELL_SURFACE_ALPHA_MAX_RELATIVE_RADIUS = 0.60
GSE_CELL_SURFACE_ALPHA_RELAXATION_FACTORS = (1.0, 1.35, 1.8, 2.4, 3.2)
GSE_CELL_SURFACE_ALPHA_MIN_BOUNDARY_FACES = 12
GSE_CELL_SURFACE_ALPHA_TRIM_RADIAL_QUANTILE = 0.998
GSE_CELL_SURFACE_ALPHA_EXPANSION = 1.010
GSE_CELL_SURFACE_ALPHA_QHULL_OPTIONS = "QJ Qbb Qc Q12"
# Support-hull fallback controls. Hulls use a bounded witness set, not all hubs,
# so dense cells remain tractable while retaining non-ellipsoid shape if an alpha
# surface is numerically degenerate.
GSE_CELL_SURFACE_HULL_MIN_POINTS = 8
GSE_CELL_SURFACE_HULL_SUPPORT_TOP_K = 3
GSE_CELL_SURFACE_HULL_MAX_WITNESS_POINTS = 384
GSE_CELL_SURFACE_HULL_RADIAL_WITNESS_FRACTION = 0.45
GSE_CELL_SURFACE_HULL_TRIM_RADIAL_QUANTILE = 0.998
GSE_CELL_SURFACE_HULL_EXPANSION = 1.015
GSE_CELL_SURFACE_HULL_QHULL_OPTIONS = "QJ Pp"
# Fallback radial shell controls for low-support or degenerate cells.
GSE_CELL_SURFACE_RADIUS_QUANTILE = 0.97
GSE_CELL_SURFACE_RADIUS_EXPANSION = 1.05
GSE_CELL_SURFACE_ANGULAR_NEIGHBOR_FRACTION = 0.14
GSE_CELL_SURFACE_ANGULAR_NEIGHBOR_MIN = 16
GSE_CELL_SURFACE_TANGENTIAL_TOP_FRACTION = 0.20
GSE_CELL_SURFACE_TANGENTIAL_DRIFT_FRACTION = 0.28
GSE_CELL_SURFACE_MAX_TANGENTIAL_DRIFT_FRACTION = 0.42
GSE_CELL_SURFACE_SMOOTH_ITERATIONS = 0
GSE_CELL_SURFACE_SMOOTH_WEIGHT = 0.20
GSE_CELL_SURFACE_AXIS_FLOOR_FRACTION = 0.035
GSE_CELL_SURFACE_DEGENERATE_RADIUS_FRACTION = 0.0025
GSE_CELL_SURFACE_OPACITY = 1.0
GSE_CELL_SURFACE_LIGHTING = True
GSE_CELL_SURFACE_SMOOTH_SHADING = True
GSE_CELL_SURFACE_MATERIAL_AMBIENT = 0.14
GSE_CELL_SURFACE_MATERIAL_DIFFUSE = 0.88
GSE_CELL_SURFACE_MATERIAL_SPECULAR = 0.28
GSE_CELL_SURFACE_MATERIAL_SPECULAR_POWER = 34.0
GSE_CELL_SURFACE_COLOR_PREFERENCE = "point"
GSE_CELL_SURFACE_ENABLE_LIGHTKIT = True
GSE_CELL_SURFACE_ENABLE_CUSTOM_LIGHTS = True
GSE_CELL_SURFACE_KEY_LIGHT_INTENSITY = 0.95
GSE_CELL_SURFACE_FILL_LIGHT_INTENSITY = 0.35
GSE_CELL_SURFACE_RIM_LIGHT_INTENSITY = 0.45
# Ambient occlusion is required for the surface renderer. The route calls SSAO
# unconditionally and aborts the surface render if PyVista/VTK cannot enable it.
GSE_CELL_SURFACE_SSAO_RADIUS = 1.15
GSE_CELL_SURFACE_SSAO_BIAS = 0.006
GSE_CELL_SURFACE_SSAO_KERNEL_SIZE = 192
# Geometry build parallelism.  Only CPU-side, per-cell NumPy/SciPy geometry is
# threaded; VTK/PyVista mesh creation and GPU rendering stay single-threaded.
GSE_CELL_SURFACE_WORKERS = int(os.environ.get("ANN12_CELL_SURFACE_WORKERS", str(max(1, min(8, (os.cpu_count() or 1) // 2 or 1)))))
GSE_CELL_SURFACE_PARALLEL_MIN_CELLS = int(os.environ.get("ANN12_CELL_SURFACE_PARALLEL_MIN_CELLS", "64"))

# Black-background-safe Leiden palette, tuned for ~5–7 clusters at very small
# point sizes on a black background. The first five entries hit maximally
# separated hues (red / yellow / green / cyan / magenta) at high saturation
# AND high value, so single-pixel dots remain readable on black. Slot 6 fills
# the cyan→magenta gap with lavender; slot 7 adds orange. Slots 8–12 are
# brighter fallbacks if the cluster count ever exceeds 7. Colors are assigned
# to the largest Leiden clusters first.
LEIDEN_PALETTE_BACKGROUND = "#000000"
LEIDEN_RESERVED_GRAY_LABELS: Tuple[str, ...] = GSE_EXCLUDED_LEIDEN_LABELS
LEIDEN_RESERVED_GRAY_COLOR = "#7A7A7A"
DARK_BG_TONAL_PRIORITY_PALETTE: Tuple[str, ...] = (
    LEIDEN_RESERVED_GRAY_COLOR,  # reserved exclusively for L0; never assigned to plotted non-L0 labels
    "#FF3B30",  # vivid red          (h ~3°,   L* ~55)
    "#FFE03B",  # vivid yellow       (h ~52°,  L* ~88)
    "#22E04A",  # vivid green        (h ~131°, L* ~78)
    "#00D9F0",  # vivid cyan         (h ~186°, L* ~79)
    "#FF6FD8",  # hot magenta/pink   (h ~313°, L* ~65)
    "#9C7BFF",  # bright lavender    (h ~255°, L* ~60) — fills cyan↔magenta
    "#FF9A1F",  # bright orange      (h ~34°,  L* ~70)
    "#5FC9FF",  # sky blue
    "#B5FF4A",  # lime
    "#FF5C7C",  # coral
    "#7AFFC1",  # mint
    "#FFB347",  # amber
)


# =============================================================================
# Figure styling
# =============================================================================

def setup_figure_style() -> None:
    """Configure matplotlib for high-quality vector output."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.unicode_minus": False,
        "pdf.use14corefonts": True,
        "ps.useafm": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "legend.frameon": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    })


setup_figure_style()
np.random.seed(RANDOM_SEED)


# =============================================================================
# General helpers needed by aggregation, Leiden, DE, marker maps, and PyVista
# =============================================================================

def _ensure_counts_layer(adata: ad.AnnData, layer: str = "counts") -> None:
    if layer not in adata.layers:
        adata.layers[layer] = adata.X.copy()


def _ensure_raw_subconsensus_counts_layer(adata: ad.AnnData) -> None:
    """Preserve the post-QC, pre-rarefaction cell x gene subconsensus counts."""
    if RAW_SUBCONSENSUS_COUNTS_LAYER in adata.layers:
        return
    source = ANALYSIS_COUNTS_LAYER if ANALYSIS_COUNTS_LAYER in adata.layers else "X"
    X = adata.layers[source].copy() if source in adata.layers else adata.X.copy()
    if sparse.issparse(X):
        X = X.tocsr(copy=True)
        if X.dtype.kind == "f":
            X.data = np.rint(X.data).astype(np.int32, copy=False)
        elif X.dtype.itemsize > 4 or X.dtype.kind not in {"i", "u"}:
            X.data = X.data.astype(np.int32, copy=False)
        X.eliminate_zeros()
    else:
        X = np.rint(np.asarray(X)).astype(np.int32, copy=False)
    adata.layers[RAW_SUBCONSENSUS_COUNTS_LAYER] = X


def _collapse_duplicate_varnames_sum(adata: ad.AnnData) -> ad.AnnData:
    """Collapse duplicate genes by summing columns, preserving layers."""
    if adata.n_vars == 0 or adata.var_names.is_unique:
        return adata
    codes, uniq = pd.factorize(adata.var_names.astype(str), sort=False)
    n_vars = adata.n_vars
    n_uniq = len(uniq)
    G = sparse.coo_matrix(
        (np.ones(n_vars, dtype=np.float32), (np.arange(n_vars, dtype=np.int32), codes.astype(np.int32))),
        shape=(n_vars, n_uniq),
    ).tocsr()
    X_new = adata.X @ G
    layers_new = {k: adata.layers[k] @ G for k in list(adata.layers.keys())}
    first_idx = pd.Series(codes).drop_duplicates(keep="first").index.to_numpy()
    var_new = adata.var.iloc[first_idx].copy()
    var_new.index = pd.Index(uniq, name=adata.var.index.name)
    adata_new = ad.AnnData(X=X_new, obs=adata.obs.copy(), var=var_new)
    for k, v in layers_new.items():
        adata_new.layers[k] = v
    for key in adata.obsm_keys():
        adata_new.obsm[key] = adata.obsm[key]
    for key in adata.uns.keys():
        adata_new.uns[key] = adata.uns[key]
    return adata_new


def _natural_sort_key(s: str) -> Tuple:
    """Natural sort key for strings like cluster labels ('0', '1', '10')."""
    try:
        return (0, int(s))
    except Exception:
        return (1, str(s))


# =============================================================================
# Distinct categorical palettes shared exactly by marker-map and PyVista outputs
# =============================================================================

def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=float)
    a = 0.055
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + a) / (1.0 + a)) ** 2.4)


def _linear_rgb_to_lab(rgb_lin: np.ndarray) -> np.ndarray:
    rgb_lin = np.asarray(rgb_lin, dtype=float)
    M = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=float,
    )
    xyz = rgb_lin @ M.T
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    xyz = xyz / np.array([Xn, Yn, Zn], dtype=float)

    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    L = 116.0 * f[:, 1] - 16.0
    a = 500.0 * (f[:, 0] - f[:, 1])
    b = 200.0 * (f[:, 1] - f[:, 2])
    return np.stack([L, a, b], axis=1)


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    return _linear_rgb_to_lab(_srgb_to_linear(np.asarray(rgb, dtype=float)))


def _generate_distinct_colors(
    n: int,
    *,
    seed: int = 0,
    background: str = "#ffffff",
    n_candidates: Optional[int] = None,
    sat_range: Tuple[float, float] = (0.75, 0.98),
    val_range: Tuple[float, float] = (0.75, 0.98),
    min_luminance: float = 0.30,
    max_luminance: float = 0.90,
) -> List[str]:
    """Generate highly distinguishable categorical colors using CIELAB farthest-point sampling."""
    n = int(n)
    if n <= 0:
        return []

    rng = np.random.default_rng(int(seed))
    if n_candidates is None:
        n_candidates = max(2000, int(300 * n))
    n_candidates = int(n_candidates)

    h = rng.random(n_candidates)
    s = rng.uniform(float(sat_range[0]), float(sat_range[1]), n_candidates)
    v = rng.uniform(float(val_range[0]), float(val_range[1]), n_candidates)

    i = np.floor(h * 6.0).astype(int)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6

    rgb = np.zeros((n_candidates, 3), dtype=float)
    mask = i == 0
    rgb[mask] = np.stack([v[mask], t[mask], p[mask]], axis=1)
    mask = i == 1
    rgb[mask] = np.stack([q[mask], v[mask], p[mask]], axis=1)
    mask = i == 2
    rgb[mask] = np.stack([p[mask], v[mask], t[mask]], axis=1)
    mask = i == 3
    rgb[mask] = np.stack([p[mask], q[mask], v[mask]], axis=1)
    mask = i == 4
    rgb[mask] = np.stack([t[mask], p[mask], v[mask]], axis=1)
    mask = i == 5
    rgb[mask] = np.stack([v[mask], p[mask], q[mask]], axis=1)

    lum = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
    keep = (lum >= float(min_luminance)) & (lum <= float(max_luminance))
    rgb_filtered = rgb[keep]
    if rgb_filtered.shape[0] >= n:
        rgb = rgb_filtered

    lab = _srgb_to_lab(rgb)
    bg_rgb = np.array([mpl.colors.to_rgb(background)], dtype=float)
    bg_lab = _srgb_to_lab(bg_rgb)[0]

    dist_bg = np.linalg.norm(lab - bg_lab[None, :], axis=1)
    first = int(np.argmax(dist_bg))
    selected = [first]
    min_dist = np.linalg.norm(lab - lab[first][None, :], axis=1)

    for _ in range(1, n):
        idx = int(np.argmax(min_dist))
        selected.append(idx)
        d = np.linalg.norm(lab - lab[idx][None, :], axis=1)
        min_dist = np.minimum(min_dist, d)

    return [mpl.colors.to_hex(tuple(c)) for c in rgb[np.array(selected, dtype=int)]]


def _extend_priority_palette(
    n: int,
    *,
    seed: int = RANDOM_SEED,
    background: str = LEIDEN_PALETTE_BACKGROUND,
) -> List[str]:
    """Return n dark-background-safe colors without maxing out brightness.

    The previous bright-first palette solved contrast against black by pushing
    value/luminance very high. In dense PyVista clouds that made different
    clusters read as similarly neon. This version starts with lower-chroma,
    tonal colors and uses generated fallbacks only in a bounded, non-neon
    luminance range.
    """
    n = int(n)
    if n <= 0:
        return []
    cols = [str(c) for c in DARK_BG_TONAL_PRIORITY_PALETTE[:n]]
    if len(cols) >= n:
        return cols[:n]

    generated = _generate_distinct_colors(
        max(n * 8, n + len(DARK_BG_TONAL_PRIORITY_PALETTE), 128),
        seed=int(seed),
        background=str(background),
        sat_range=(0.85, 1.00),
        val_range=(0.85, 1.00),
        min_luminance=0.55,
        max_luminance=0.95,
    )
    existing = {c.lower() for c in cols}
    existing_lab = (
        _srgb_to_lab(np.array([mpl.colors.to_rgb(c) for c in cols], dtype=float))
        if cols
        else np.empty((0, 3), dtype=float)
    )
    deferred: List[str] = []
    for cand in generated:
        cand = str(cand)
        cand_l = cand.lower()
        if cand_l in existing:
            continue
        cand_lab = _srgb_to_lab(np.array([mpl.colors.to_rgb(cand)], dtype=float))[0]
        min_dist = (
            float(np.min(np.linalg.norm(existing_lab - cand_lab[None, :], axis=1)))
            if existing_lab.shape[0] > 0
            else float("inf")
        )
        if min_dist >= 22.0:
            cols.append(cand)
            existing.add(cand_l)
            existing_lab = np.vstack([existing_lab, cand_lab[None, :]])
            if len(cols) >= n:
                break
        else:
            deferred.append(cand)
    for cand in deferred:
        cand_l = str(cand).lower()
        if cand_l not in existing:
            cols.append(str(cand))
            existing.add(cand_l)
            if len(cols) >= n:
                break
    return cols[:n]


def make_distinct_palette(categories: Sequence, *, seed: int = RANDOM_SEED, background: str = "#ffffff") -> Dict[str, str]:
    cats = [str(c) for c in pd.Index(categories).astype(str).tolist()]
    cats = sorted(dict.fromkeys(cats), key=_natural_sort_key)
    cols = _extend_priority_palette(len(cats), seed=int(seed), background=background)
    return {c: col for c, col in zip(cats, cols)}


def _nonreserved_leiden_palette_colors(n: int, *, seed: int = RANDOM_SEED) -> List[str]:
    """Return n Leiden colors excluding the reserved L0 gray slot."""
    n = int(n)
    if n <= 0:
        return []
    cols = [str(c) for c in _extend_priority_palette(n + 1, seed=int(seed), background=str(LEIDEN_PALETTE_BACKGROUND))]
    reserved = {str(LEIDEN_RESERVED_GRAY_COLOR).lower()}
    out = [c for c in cols if str(c).lower() not in reserved]
    if len(out) < n:
        extra = _generate_distinct_colors(
            max(128, 8 * n),
            seed=int(seed) + 1009,
            background=str(LEIDEN_PALETTE_BACKGROUND),
            sat_range=(0.85, 1.00),
            val_range=(0.85, 1.00),
            min_luminance=0.55,
            max_luminance=0.95,
        )
        seen = {x.lower() for x in out} | reserved
        for c in extra:
            if str(c).lower() not in seen:
                out.append(str(c))
                seen.add(str(c).lower())
                if len(out) >= n:
                    break
    return out[:n]


def _muted_non_gray_hex(hex_color: str, *, fraction: float = 0.46) -> str:
    """Darken a palette color without converting it to the reserved gray channel."""
    try:
        rgb = np.asarray(mpl.colors.to_rgb(str(hex_color)), dtype=float)
        frac = float(np.clip(fraction, 0.18, 0.85))
        muted = np.clip(rgb * frac, 0.0, 1.0)
        if float(np.max(muted) - np.min(muted)) < 0.08:
            j = int(np.argmax(rgb))
            muted[j] = min(1.0, muted[j] + 0.12)
        return mpl.colors.to_hex(tuple(muted))
    except Exception:
        return "#334155"


def _leiden_render_color_for_label(label: str, palette_hex: Dict[str, str]) -> str:
    """Return a non-gray PyVista color for a plotted non-L0 Leiden label.

    This is a final guard against stale palette TSVs in cached runs:
    even if an older palette mapped a non-L0 label to the reserved gray swatch,
    the rendered full-Leiden/top-5 foreground actor is reassigned a deterministic
    nonreserved palette color.
    """
    lab = str(label)
    color = str((palette_hex or {}).get(lab, "#ffffff"))
    reserved = {str(x) for x in LEIDEN_RESERVED_GRAY_LABELS}
    try:
        is_reserved_gray = str(mpl.colors.to_hex(color)).lower() == str(mpl.colors.to_hex(LEIDEN_RESERVED_GRAY_COLOR)).lower()
    except Exception:
        is_reserved_gray = str(color).lower() == str(LEIDEN_RESERVED_GRAY_COLOR).lower()
    if lab not in reserved and is_reserved_gray:
        colors = _nonreserved_leiden_palette_colors(64, seed=int(RANDOM_SEED))
        if colors:
            h = int(hashlib.blake2b(lab.encode("utf-8"), digest_size=2).hexdigest(), 16)
            return str(colors[h % len(colors)])
        return "#ffffff"
    return color


def leiden_palette_from_labels(labels: Sequence, *, seed: int = RANDOM_SEED) -> Dict[str, str]:
    """Assign Leiden colors while reserving gray exclusively for L0."""
    label_s = pd.Series(labels, dtype=object).astype(str)
    if label_s.shape[0] == 0:
        return {}
    counts = label_s.value_counts().to_dict()
    cats = sorted(counts.keys(), key=lambda c: (-int(counts.get(c, 0)), _natural_sort_key(str(c))))
    reserved_labels = {str(x) for x in LEIDEN_RESERVED_GRAY_LABELS}
    plotted_cats = [str(c) for c in cats if str(c) not in reserved_labels]
    cols = _nonreserved_leiden_palette_colors(len(plotted_cats), seed=int(seed))
    palette = {str(c): col for c, col in zip(plotted_cats, cols)}
    for lab in cats:
        if str(lab) in reserved_labels:
            palette[str(lab)] = str(LEIDEN_RESERVED_GRAY_COLOR)
    return {str(c): palette[str(c)] for c in cats if str(c) in palette}

def leiden_palette_from_adata(adata: ad.AnnData, leiden_key: str = LEIDEN_KEY) -> Dict[str, str]:
    if leiden_key not in adata.obs.columns:
        raise KeyError(f"Missing obs['{leiden_key}']")
    return leiden_palette_from_labels(adata.obs[leiden_key].astype(str).values, seed=int(RANDOM_SEED))


def _hex_to_rgba_u8(hex_color: str, *, alpha: int = 255) -> np.ndarray:
    r, g, b, _ = mpl.colors.to_rgba(hex_color)
    return np.array([int(round(r * 255)), int(round(g * 255)), int(round(b * 255)), int(alpha)], dtype=np.uint8)


def save_palette_tsvs(palette_hex: Dict[str, str], *, key: str, out_dirs: Sequence[str]) -> None:
    """Save the exact label->color mapping consumed by figure and PyVista routes."""
    df = pd.DataFrame({"label": list(palette_hex.keys()), "hex": list(palette_hex.values())})
    for out_dir in out_dirs:
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, f"{key}_palette.tsv"), sep="\t", index=False)
        df.to_csv(os.path.join(out_dir, f"{key}_palette_tab20.tsv"), sep="\t", index=False)


def save_categorical_palette_legend(
    palette_hex: Dict[str, str],
    *,
    out_pdf: str,
    out_png: Optional[str] = None,
    title: str = "",
    subtitle: str = "",
    max_cols: int = 3,
    fontsize: float = 8.0,
) -> None:
    """Save a simple categorical color key as PDF/PNG."""
    if not isinstance(palette_hex, dict) or len(palette_hex) == 0:
        return

    labels = sorted([str(x) for x in palette_hex.keys()], key=_natural_sort_key)
    n = len(labels)
    ncols = int(max(1, min(int(max_cols), n)))
    nrows = int(np.ceil(n / ncols))
    fig_w = 6.5 if ncols <= 2 else 8.0
    fig_h = max(1.6, 0.36 * nrows + (0.55 if (title or subtitle) else 0.25))

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    y = 0.98
    if title:
        ax.text(0.02, y, str(title), ha="left", va="top", fontsize=10, fontweight="bold")
        y -= 0.06
    if subtitle:
        ax.text(0.02, y, str(subtitle), ha="left", va="top", fontsize=8, color="#444444")
        y -= 0.05

    left, right, top, bottom = 0.02, 0.98, y, 0.04
    row_h = (top - bottom) / max(nrows, 1)
    col_w = (right - left) / max(ncols, 1)

    for i, lab in enumerate(labels):
        r = i % nrows
        c = i // nrows
        x0 = left + c * col_w
        y0 = top - (r + 1) * row_h
        ax.add_patch(
            mpatches.Rectangle(
                (x0, y0 + 0.5 * row_h - 0.009),
                0.035,
                0.018,
                transform=ax.transAxes,
                facecolor=palette_hex.get(str(lab), "#888888"),
                edgecolor="none",
            )
        )
        ax.text(
            x0 + 0.045,
            y0 + 0.5 * row_h,
            str(lab),
            ha="left",
            va="center",
            fontsize=float(fontsize),
            color="#111111",
            transform=ax.transAxes,
        )

    fig.savefig(out_pdf, format="pdf", dpi=300, bbox_inches="tight")
    if out_png:
        fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Hub -> cell aggregation, QC, PCA-neighbor graph, and Leiden
# =============================================================================

def _infer_sample_name_from_filepath(filepath: str) -> str:
    """Infer sample name from a hub-level H5AD path."""
    try:
        parts = [p for p in os.path.normpath(filepath).split(os.sep) if p]
        for part in parts:
            if part.endswith(".UEI"):
                token = part.split("-")[0].strip()
                if token:
                    return token
        for i, part in enumerate(parts):
            if part.startswith("uei_grp") and i > 0:
                token = parts[i - 1].split("-")[0].strip()
                if token:
                    return token
        for part in parts:
            if "-" in part and not part.endswith(".h5ad"):
                token = part.split("-")[0].strip()
                if token:
                    return token
    except Exception:
        pass
    return "unknown"


def _map_sample_to_age(sample_name: str) -> str:
    sample = str(sample_name)
    return str(AGE_BY_SAMPLE.get(sample, "unknown"))


def annotate_age_from_sample(adata: ad.AnnData, *, sample_col: str = "sample", age_col: str = "age_hpf") -> None:
    if sample_col not in adata.obs.columns:
        return
    adata.obs[age_col] = adata.obs[sample_col].astype(str).map(_map_sample_to_age).astype(str)




def _binary_threshold_matrix(X, *, min_reads: int = MIN_READS) -> sparse.csr_matrix:
    """Return a CSR matrix with one event per hub x feature call passing MIN_READS."""
    X_csr = X.tocsr(copy=True) if sparse.issparse(X) else sparse.csr_matrix(np.asarray(X))
    if X_csr.nnz:
        X_csr.data = (X_csr.data >= int(min_reads)).astype(np.int8, copy=False)
        X_csr.eliminate_zeros()
    return X_csr.tocsr()


def _feature_type_normalized(var: pd.DataFrame) -> pd.Series:
    """Normalize feature_type values for robust rRNA/Mt-rRNA detection."""
    if "feature_type" not in var.columns:
        return pd.Series([""] * int(var.shape[0]), index=var.index, dtype="string")
    s = var["feature_type"].astype("string").fillna("").astype(str)
    return s.str.strip().str.replace("-", "_", regex=False).str.replace(" ", "_", regex=False).str.lower()


def _gene_id_series(var: pd.DataFrame, *, gene_id_key: str = "gene_id") -> pd.Series:
    """Return feature gene IDs, falling back to var_names for missing IDs."""
    if gene_id_key in var.columns:
        s = var[gene_id_key].astype("string").fillna("").astype(str).str.strip()
    elif "gene_id" in var.columns:
        s = var["gene_id"].astype("string").fillna("").astype(str).str.strip()
    else:
        s = pd.Series(var.index.astype(str), index=var.index).astype("string").fillna("").astype(str).str.strip()
    missing = s.eq("") | s.str.lower().isin(["nan", "none", "<na>"])
    if bool(missing.any()):
        fallback = pd.Series(var.index.astype(str), index=var.index).astype(str)
        s.loc[missing] = fallback.loc[missing]
    return s.astype(str)


def _valid_gene_id_mask(gene_ids: pd.Series) -> np.ndarray:
    """Ann12-compatible valid-gene gate, used after pseudo-feature construction."""
    s = gene_ids.astype("string").fillna("").astype(str).str.strip()
    bad_tokens = {"", "nan", "none", "<na>", "intergenic", "intronic", "unknown", "unmapped"}
    ok = ~s.str.lower().isin(bad_tokens)
    ok &= ~s.str.startswith("__genome__", na=False)
    return ok.to_numpy(dtype=bool)


def _append_pseudo_feature(
    X: sparse.csr_matrix,
    var: pd.DataFrame,
    *,
    feature_name: str,
    feature_type: str,
    values: np.ndarray,
) -> Tuple[sparse.csr_matrix, pd.DataFrame]:
    """Append one binary pseudo-feature column to a hub x feature matrix."""
    values = np.asarray(values).reshape(-1, 1).astype(np.int8, copy=False)
    X_new = sparse.hstack([X.tocsr(), sparse.csr_matrix(values)], format="csr")
    row_data = {col: "" for col in var.columns}
    row_data["gene_id"] = str(feature_name)
    row_data["feature_type"] = str(feature_type)
    row = pd.DataFrame([row_data], index=[str(feature_name)])
    var_new = pd.concat([var.copy(), row], axis=0)
    return X_new, var_new


def _collapse_csr_columns_by_names(
    X: sparse.csr_matrix,
    var: pd.DataFrame,
    names: Sequence[str],
) -> Tuple[sparse.csr_matrix, pd.DataFrame]:
    """Collapse duplicate feature names by summing CSR columns."""
    names_index = pd.Index(pd.Series(names, dtype="string").fillna("").astype(str).values)
    if names_index.is_unique:
        var_new = var.copy()
        var_new.index = names_index
        var_new["gene_id"] = names_index.astype(str)
        return X.tocsr(), var_new

    codes, uniq = pd.factorize(names_index.astype(str), sort=False)
    n_vars = int(len(names_index))
    n_uniq = int(len(uniq))
    G = sparse.coo_matrix(
        (np.ones(n_vars, dtype=np.float32), (np.arange(n_vars, dtype=np.int32), codes.astype(np.int32))),
        shape=(n_vars, n_uniq),
    ).tocsr()
    X_new = (X.tocsr() @ G).tocsr()
    first_idx = pd.Series(codes).drop_duplicates(keep="first").index.to_numpy()
    var_new = var.iloc[first_idx].copy()
    var_new.index = pd.Index(uniq.astype(str), name=var.index.name)
    var_new["gene_id"] = var_new.index.astype(str)
    return X_new, var_new


def build_augmented_gene_matrix(
    adata_node: ad.AnnData,
    *,
    min_reads: int = MIN_READS,
    gene_id_key: str = "gene_id",
) -> Tuple[sparse.csr_matrix, pd.DataFrame, Dict[str, int]]:
    """Build a hub x feature binary matrix with rRNA/Mt-rRNA pseudo-features.

    This mirrors the targeted earlier behavior: create rRNA_sum and Mt_rRNA_sum
    from raw rRNA-like feature calls, remove the individual raw rRNA/Mt-rRNA
    columns, then collapse duplicate gene IDs by summing columns.
    """
    X_bin = _binary_threshold_matrix(adata_node.X, min_reads=int(min_reads))
    var = adata_node.var.copy()
    diag: Dict[str, int] = {
        "n_features_initial": int(var.shape[0]),
        "n_raw_rrna_features": 0,
        "n_raw_mt_rrna_features": 0,
        "n_pseudo_features_added": 0,
        "n_pseudo_rrna_positive_hubs": 0,
        "n_pseudo_mt_rrna_positive_hubs": 0,
        "n_features_dropped_invalid": 0,
        "n_features_dropped_raw_rrna": 0,
        "n_features_kept_after_rrna_pseudo": 0,
    }

    if bool(RRNA_PSEUDO_FEATURES_ENABLE):
        ft_norm = _feature_type_normalized(var)
        rrna_like = ft_norm.str.contains("rrna", regex=False).to_numpy(dtype=bool)
        mt_rrna_mask = (ft_norm.str.contains("rrna", regex=False) & ft_norm.str.contains("mt", regex=False)).to_numpy(dtype=bool)
        # Keep the two pseudo-feature classes disjoint, as in earlier.
        rrna_mask = rrna_like & ~mt_rrna_mask
        diag["n_raw_rrna_features"] = int(np.sum(rrna_mask))
        diag["n_raw_mt_rrna_features"] = int(np.sum(mt_rrna_mask))

        rrna_has = (
            np.asarray(X_bin[:, rrna_mask].sum(axis=1)).ravel() > 0
            if bool(rrna_mask.any())
            else np.zeros(X_bin.shape[0], dtype=bool)
        )
        mt_rrna_has = (
            np.asarray(X_bin[:, mt_rrna_mask].sum(axis=1)).ravel() > 0
            if bool(mt_rrna_mask.any())
            else np.zeros(X_bin.shape[0], dtype=bool)
        )
        diag["n_pseudo_rrna_positive_hubs"] = int(np.sum(rrna_has))
        diag["n_pseudo_mt_rrna_positive_hubs"] = int(np.sum(mt_rrna_has))

        X_aug, var_aug = _append_pseudo_feature(
            X_bin,
            var,
            feature_name=str(RRNA_SUM_FEATURE),
            feature_type="aggregated_rRNA",
            values=rrna_has.astype(np.int8, copy=False),
        )
        X_aug, var_aug = _append_pseudo_feature(
            X_aug,
            var_aug,
            feature_name=str(MT_RRNA_SUM_FEATURE),
            feature_type="aggregated_Mt_rRNA",
            values=mt_rrna_has.astype(np.int8, copy=False),
        )
        diag["n_pseudo_features_added"] = 2
    else:
        X_aug, var_aug = X_bin, var

    gene_ids = _gene_id_series(var_aug, gene_id_key=str(gene_id_key))
    valid_gene = _valid_gene_id_mask(gene_ids)
    diag["n_features_dropped_invalid"] = int(np.sum(~valid_gene))

    if bool(RRNA_PSEUDO_FEATURES_ENABLE) and "feature_type" in var_aug.columns:
        ft_aug = _feature_type_normalized(var_aug)
        is_pseudo_rrna = ft_aug.eq("aggregated_rrna") | ft_aug.eq("aggregated_mt_rrna")
        raw_rrna_like = ft_aug.str.contains("rrna", regex=False) & ~is_pseudo_rrna
        keep_feature_type = (~raw_rrna_like).to_numpy(dtype=bool)
        diag["n_features_dropped_raw_rrna"] = int(np.sum(raw_rrna_like.to_numpy(dtype=bool) & valid_gene))
    else:
        keep_feature_type = np.ones(var_aug.shape[0], dtype=bool)

    keep = valid_gene & keep_feature_type
    if int(np.sum(keep)) == 0:
        raise ValueError("No valid features after gene-ID filtering and rRNA/Mt-rRNA pseudo-feature aggregation.")

    X_keep = X_aug[:, keep].tocsr()
    var_keep = var_aug.iloc[np.flatnonzero(keep)].copy()
    gene_keep = gene_ids.iloc[np.flatnonzero(keep)].astype(str).values
    X_collapsed, var_collapsed = _collapse_csr_columns_by_names(X_keep, var_keep, gene_keep)
    diag["n_features_kept_after_rrna_pseudo"] = int(X_collapsed.shape[1])
    return X_collapsed.tocsr(), var_collapsed, diag


def _load_kept_symmetric_connectivity(filepath: str, kept_original_indices: np.ndarray) -> Optional[sparse.csr_matrix]:
    """Load link_assoc_reindexed.npz and return a kept-hub symmetric subgraph."""
    conn_path = os.path.join(os.path.dirname(filepath), str(CONNECTIVITY_NPZ_BASENAME))
    if not os.path.isfile(conn_path):
        print(f"   [ComponentSplit] {CONNECTIVITY_NPZ_BASENAME} missing; falling back to unsplit infomap clusters.")
        return None
    try:
        conn_raw = sparse.load_npz(conn_path).tocsr()
    except Exception as e:
        print(f"   [ComponentSplit] Failed reading {conn_path} ({e}); falling back to unsplit infomap clusters.")
        return None
    if conn_raw.shape[0] != conn_raw.shape[1]:
        print(f"   [ComponentSplit] Connectivity matrix is not square ({conn_raw.shape}); falling back to unsplit clusters.")
        return None
    kept_original_indices = np.asarray(kept_original_indices, dtype=np.int64)
    if kept_original_indices.size and int(kept_original_indices.max()) >= int(conn_raw.shape[0]):
        print("   [ComponentSplit] Connectivity matrix is smaller than the max kept hub index; falling back to unsplit clusters.")
        return None
    C = conn_raw[kept_original_indices, :][:, kept_original_indices].tocsr()
    C = (C + C.T).tocsr()
    C.setdiag(0)
    C.eliminate_zeros()
    return C


def _enumerate_disconnected_components(
    X_hub: sparse.csr_matrix,
    clusters: np.ndarray,
    kept_original_indices: np.ndarray,
    *,
    filepath: str,
) -> List[Dict[str, object]]:
    """Enumerate disconnected components within each infomap cell."""
    conn = _load_kept_symmetric_connectivity(filepath, kept_original_indices) if bool(SPLIT_DISCONNECTED_COMPONENTS) else None
    clusters_s = pd.Series(clusters).astype(str).to_numpy(dtype=object)
    order = np.argsort(clusters_s.astype(str), kind="mergesort")
    sorted_clusters = clusters_s[order].astype(str)
    unique_clusters, starts = np.unique(sorted_clusters, return_index=True)
    ends = np.r_[starts[1:], len(order)]

    components: List[Dict[str, object]] = []
    for raw_cluster, start, end in zip(unique_clusters, starts, ends):
        hub_locals_all = order[int(start):int(end)].astype(np.int64, copy=False)
        if hub_locals_all.size == 0:
            continue

        if conn is None or hub_locals_all.size == 1:
            cc_labels = np.zeros(hub_locals_all.size, dtype=np.int32)
            n_cc = 1
        else:
            sub_adj = conn[hub_locals_all, :][:, hub_locals_all].tocsr()
            sub_adj.eliminate_zeros()
            n_cc, cc_labels = sparse_connected_components(sub_adj, directed=False)

        for cc_id in range(int(n_cc)):
            cc_local_positions = np.flatnonzero(cc_labels == int(cc_id)).astype(np.int64, copy=False)
            hub_locals = hub_locals_all[cc_local_positions].astype(np.int64, copy=False)
            if hub_locals.size == 0:
                continue
            X_comp = X_hub[hub_locals, :].tocsr()
            vec = X_comp.sum(axis=0)
            if sparse.issparse(vec):
                vec_csr = vec.tocsr()
                subconsensus = int(np.asarray(vec_csr.sum()).ravel()[0])
                distinct_genes = int(vec_csr.getnnz())
            else:
                arr = np.asarray(vec).ravel()
                subconsensus = int(arr.sum())
                distinct_genes = int(np.count_nonzero(arr))
            hub_call_counts = np.asarray(X_comp.sum(axis=1)).ravel().astype(np.int64)
            n_hubs_with_call = int(np.sum(hub_call_counts > 0))
            n_multigene_hubs = int(np.sum(hub_call_counts > 1))
            components.append({
                "raw_cluster": str(raw_cluster),
                "component_id": int(cc_id),
                "hub_local_indices": hub_locals,
                "subconsensus_count": int(subconsensus),
                "distinct_gene_count": int(distinct_genes),
                "n_hubs": int(hub_locals.size),
                "n_hubs_with_call": int(n_hubs_with_call),
                "n_multigene_hubs": int(n_multigene_hubs),
                "n_components_in_raw_cluster": int(n_cc),
            })

    if conn is not None:
        del conn
    gc.collect()
    return components


def _make_component_label(comp: Dict[str, object]) -> str:
    """Make a stable refined cell label from a raw infomap label and component ID."""
    raw = str(comp.get("raw_cluster", ""))
    raw = "_".join(raw.split()).replace(os.sep, "_")
    if raw == "":
        raw = "cluster"
    return f"{raw}_cc{int(comp.get('component_id', 0))}"


def _write_refined_hub_label_sidecar(
    *,
    filepath: str,
    n_hubs_original: int,
    kept_original_indices: np.ndarray,
    selected_components: Sequence[Dict[str, object]],
    cell_labels: Sequence[str],
) -> None:
    """Write original-hub -> refined component-cell labels for downstream PyVista routes."""
    if not bool(WRITE_REFINED_HUB_LABEL_SIDECAR):
        return
    try:
        sidecar_labels = np.full(int(n_hubs_original), "", dtype=object)
        kept_original_indices = np.asarray(kept_original_indices, dtype=np.int64)
        for label, comp in zip(cell_labels, selected_components):
            local_idx = np.asarray(comp.get("hub_local_indices", []), dtype=np.int64)
            if local_idx.size == 0:
                continue
            original_idx = kept_original_indices[local_idx]
            sidecar_labels[original_idx] = str(label)
        sidecar_path = os.path.join(os.path.dirname(filepath), str(REFINED_HUB_LABEL_SIDECAR))
        pd.DataFrame({
            "hub_original_index": np.arange(int(n_hubs_original), dtype=np.int64),
            "refined_cell_label": sidecar_labels,
        }).to_csv(sidecar_path, sep="\t", index=False)
        print(f"   [ComponentSplit] Wrote refined hub-label sidecar: {sidecar_path}")
    except Exception as e:
        print(f"   [ComponentSplit] WARNING: failed to write refined hub-label sidecar ({e}).")

def aggregate_nodes_to_cells(
    filepath: str,
    *,
    min_reads: int = MIN_READS,
    cluster_key: str = CLUSTER_KEY,
    drop_cluster_values: Optional[set] = DROP_CLUSTER_VALUES,
    gene_id_key: str = "gene_id",
    diagnostics_out: Optional[Dict[str, object]] = None,
) -> Optional[ad.AnnData]:
    """Aggregate hub-level calls into a component-split cell x gene matrix.

    Targeted changes applied here:
      1. Raw rRNA/Mt-rRNA features are replaced by rRNA_sum and Mt_rRNA_sum
         pseudo-features before gene-ID collapsing.
      2. Each raw ``cluster_key`` value is split into disconnected components
         from ``link_assoc_reindexed.npz``; each nonempty component becomes a
         candidate cell and then flows through the existing ann12 QC gates.
    """
    sample_name = _infer_sample_name_from_filepath(filepath)
    print(f"\n=== Loading {sample_name} from {filepath} ===")

    if not os.path.exists(filepath):
        print(f"!! File not found: {filepath}. Skipping.")
        return None

    adata_node = sc.read_h5ad(filepath)

    n_hubs_initial = int(adata_node.n_obs)
    n_features_initial = int(adata_node.n_vars)
    n_hubs_dropped_nan_cluster = 0
    n_hubs_dropped_drop_value = 0
    drop_value_breakdown: Dict[str, int] = {}

    try:
        X_aug, var_aug, feature_diag = build_augmented_gene_matrix(
            adata_node,
            min_reads=int(min_reads),
            gene_id_key=str(gene_id_key),
        )
    except Exception as e:
        print(f"!! Failed to build augmented gene matrix for {filepath}: {e}")
        if diagnostics_out is not None:
            diagnostics_out.update({
                "sample": sample_name,
                "filepath": filepath,
                "n_hubs_initial": n_hubs_initial,
                "n_features_initial": n_features_initial,
                "n_features_dropped_invalid": 0,
                "n_features_kept": 0,
                "n_hubs_dropped_nan_cluster": 0,
                "n_hubs_dropped_drop_value": 0,
                "n_hubs_after_cluster_filter": 0,
                "n_hubs_with_zero_calls": 0,
                "n_hubs_with_at_least_one_call": 0,
                "n_cells_pre_qc": 0,
                "drop_value_breakdown": {},
                "per_hub_calls": np.array([], dtype=np.int64),
                "aborted": True,
                "abort_reason": f"augmented_gene_matrix_failed:{type(e).__name__}",
            })
        return None

    n_features_dropped_invalid = int(feature_diag.get("n_features_dropped_invalid", 0))
    n_features_kept = int(X_aug.shape[1])
    print(
        f"   [Features] Retained {n_features_kept:,}/{n_features_initial:,} features "
        f"after valid-gene filtering and rRNA/Mt-rRNA pseudo-feature aggregation "
        f"(raw rRNA dropped={int(feature_diag.get('n_features_dropped_raw_rrna', 0)):,}, "
        f"pseudo added={int(feature_diag.get('n_pseudo_features_added', 0))})."
    )

    if X_aug.shape[1] == 0 or adata_node.n_obs == 0:
        if diagnostics_out is not None:
            diagnostics_out.update({
                "sample": sample_name,
                "filepath": filepath,
                "n_hubs_initial": n_hubs_initial,
                "n_features_initial": n_features_initial,
                "n_features_dropped_invalid": n_features_dropped_invalid,
                "n_features_kept": int(X_aug.shape[1]),
                "n_hubs_dropped_nan_cluster": 0,
                "n_hubs_dropped_drop_value": 0,
                "n_hubs_after_cluster_filter": 0,
                "n_hubs_with_zero_calls": 0,
                "n_hubs_with_at_least_one_call": 0,
                "n_cells_pre_qc": 0,
                "drop_value_breakdown": {},
                "per_hub_calls": np.array([], dtype=np.int64),
                "aborted": True,
                **feature_diag,
            })
        return None

    if cluster_key not in adata_node.obs.columns:
        print(f"!! Missing obs['{cluster_key}'] in {filepath}. Skipping.")
        if diagnostics_out is not None:
            diagnostics_out.update({
                "sample": sample_name,
                "filepath": filepath,
                "n_hubs_initial": n_hubs_initial,
                "n_features_initial": n_features_initial,
                "n_features_dropped_invalid": n_features_dropped_invalid,
                "n_features_kept": int(X_aug.shape[1]),
                "n_hubs_dropped_nan_cluster": int(n_hubs_initial),
                "n_hubs_dropped_drop_value": 0,
                "n_hubs_after_cluster_filter": 0,
                "n_hubs_with_zero_calls": 0,
                "n_hubs_with_at_least_one_call": 0,
                "n_cells_pre_qc": 0,
                "drop_value_breakdown": {},
                "per_hub_calls": np.array([], dtype=np.int64),
                "aborted": True,
                **feature_diag,
            })
        return None

    cl = adata_node.obs[cluster_key]
    cl_str = cl.astype("string").fillna("").astype(str).str.strip()
    nan_mask = ~cl.notna().to_numpy(dtype=bool)
    n_hubs_dropped_nan_cluster = int(nan_mask.sum())
    drop_mask = np.zeros(adata_node.n_obs, dtype=bool)
    if drop_cluster_values:
        for dv in drop_cluster_values:
            this_mask = (cl_str == str(dv)).to_numpy(dtype=bool) & ~nan_mask
            drop_value_breakdown[str(dv)] = int(this_mask.sum())
            drop_mask |= this_mask
    n_hubs_dropped_drop_value = int(drop_mask.sum())
    keep_hubs = ~(nan_mask | drop_mask)
    kept_original_indices = np.flatnonzero(keep_hubs).astype(np.int64)
    n_hubs_after_cluster_filter = int(kept_original_indices.size)

    if n_hubs_after_cluster_filter == 0:
        if diagnostics_out is not None:
            diagnostics_out.update({
                "sample": sample_name,
                "filepath": filepath,
                "n_hubs_initial": n_hubs_initial,
                "n_features_initial": n_features_initial,
                "n_features_dropped_invalid": n_features_dropped_invalid,
                "n_features_kept": int(X_aug.shape[1]),
                "n_hubs_dropped_nan_cluster": n_hubs_dropped_nan_cluster,
                "n_hubs_dropped_drop_value": n_hubs_dropped_drop_value,
                "n_hubs_after_cluster_filter": 0,
                "n_hubs_with_zero_calls": 0,
                "n_hubs_with_at_least_one_call": 0,
                "n_cells_pre_qc": 0,
                "drop_value_breakdown": drop_value_breakdown,
                "per_hub_calls": np.array([], dtype=np.int64),
                "aborted": True,
                **feature_diag,
            })
        return None

    X_hub = X_aug[kept_original_indices, :].tocsr()
    clusters = cl_str.iloc[kept_original_indices].astype(str).values
    hub_calls_full = np.asarray(X_hub.sum(axis=1)).ravel().astype(np.int64)
    n_hubs_with_zero_calls = int(np.sum(hub_calls_full == 0))
    n_hubs_with_at_least_one_call = int(np.sum(hub_calls_full > 0))

    if n_hubs_with_at_least_one_call == 0:
        if diagnostics_out is not None:
            diagnostics_out.update({
                "sample": sample_name,
                "filepath": filepath,
                "n_hubs_initial": n_hubs_initial,
                "n_features_initial": n_features_initial,
                "n_features_dropped_invalid": n_features_dropped_invalid,
                "n_features_kept": int(X_aug.shape[1]),
                "n_hubs_dropped_nan_cluster": n_hubs_dropped_nan_cluster,
                "n_hubs_dropped_drop_value": n_hubs_dropped_drop_value,
                "n_hubs_after_cluster_filter": n_hubs_after_cluster_filter,
                "n_hubs_with_zero_calls": int(n_hubs_after_cluster_filter),
                "n_hubs_with_at_least_one_call": 0,
                "n_cells_pre_qc": 0,
                "drop_value_breakdown": drop_value_breakdown,
                "per_hub_calls": hub_calls_full,
                "aborted": True,
                **feature_diag,
            })
        return None

    components_all = _enumerate_disconnected_components(
        X_hub,
        clusters,
        kept_original_indices,
        filepath=filepath,
    )
    selected_components = [
        c for c in components_all
        if int(c.get("subconsensus_count", 0)) > 0 and int(c.get("n_hubs_with_call", 0)) > 0
    ]
    n_raw_clusters = int(pd.Series(clusters).astype(str).nunique())
    n_split_raw_clusters = int(sum(
        1 for _raw, grp in pd.DataFrame({
            "raw": [str(c.get("raw_cluster", "")) for c in components_all],
            "ncc": [int(c.get("n_components_in_raw_cluster", 1)) for c in components_all],
        }).groupby("raw") if int(grp["ncc"].max()) > 1
    )) if components_all else 0
    print(
        f"   [ComponentSplit] Raw clusters={n_raw_clusters:,}; components enumerated={len(components_all):,}; "
        f"candidate nonempty components={len(selected_components):,}; split raw clusters={n_split_raw_clusters:,}."
    )

    if len(selected_components) == 0:
        if diagnostics_out is not None:
            diagnostics_out.update({
                "sample": sample_name,
                "filepath": filepath,
                "n_hubs_initial": n_hubs_initial,
                "n_features_initial": n_features_initial,
                "n_features_dropped_invalid": n_features_dropped_invalid,
                "n_features_kept": int(X_aug.shape[1]),
                "n_hubs_dropped_nan_cluster": n_hubs_dropped_nan_cluster,
                "n_hubs_dropped_drop_value": n_hubs_dropped_drop_value,
                "n_hubs_after_cluster_filter": n_hubs_after_cluster_filter,
                "n_hubs_with_zero_calls": n_hubs_with_zero_calls,
                "n_hubs_with_at_least_one_call": n_hubs_with_at_least_one_call,
                "n_cells_pre_qc": 0,
                "drop_value_breakdown": drop_value_breakdown,
                "per_hub_calls": hub_calls_full,
                "n_raw_clusters_pre_component_split": n_raw_clusters,
                "n_components_enumerated": int(len(components_all)),
                "n_components_with_calls": 0,
                "n_raw_clusters_split": n_split_raw_clusters,
                "aborted": True,
                **feature_diag,
            })
        return None

    cell_labels = [_make_component_label(c) for c in selected_components]
    row_parts: List[np.ndarray] = []
    col_parts: List[np.ndarray] = []
    for j, comp in enumerate(selected_components):
        hubs = np.asarray(comp["hub_local_indices"], dtype=np.int64)
        row_parts.append(hubs)
        col_parts.append(np.full(hubs.shape[0], int(j), dtype=np.int64))
    rows = np.concatenate(row_parts).astype(np.int64, copy=False)
    cols = np.concatenate(col_parts).astype(np.int64, copy=False)
    A = sparse.csr_matrix(
        (np.ones(rows.shape[0], dtype=np.int8), (rows, cols)),
        shape=(X_hub.shape[0], len(selected_components)),
    )
    cell_gene = (A.T @ X_hub).tocsr()

    adata_cell = ad.AnnData(X=cell_gene, var=var_aug.copy())
    adata_cell.obs_names = pd.Index(cell_labels, dtype=str)
    adata_cell.var_names = var_aug.index.astype(str)
    adata_cell.var["gene_id"] = adata_cell.var_names.astype(str)
    adata_cell.obs["sample"] = sample_name
    adata_cell.obs["age_hpf"] = _map_sample_to_age(sample_name)
    adata_cell.obs["cluster_id"] = adata_cell.obs_names.astype(str)
    adata_cell.obs["cluster_id_raw"] = [str(c["raw_cluster"]) for c in selected_components]
    adata_cell.obs["component_id"] = [int(c["component_id"]) for c in selected_components]
    adata_cell.obs["n_components_in_raw_cluster"] = [int(c["n_components_in_raw_cluster"]) for c in selected_components]
    adata_cell.obs["component_split_from_raw_cluster"] = [int(c["n_components_in_raw_cluster"]) > 1 for c in selected_components]
    adata_cell.obs["n_hubs"] = [int(c["n_hubs"]) for c in selected_components]
    adata_cell.obs["n_hubs_with_call"] = [int(c["n_hubs_with_call"]) for c in selected_components]
    adata_cell.obs["n_distinct_subconsensuses"] = adata_cell.obs["n_hubs_with_call"].astype(int).values
    adata_cell.obs["n_multigene_hubs"] = [int(c["n_multigene_hubs"]) for c in selected_components]
    adata_cell.obs["frac_multigene_hubs"] = np.divide(
        adata_cell.obs["n_multigene_hubs"].astype(float).values,
        np.maximum(adata_cell.obs["n_hubs_with_call"].astype(float).values, 1.0),
    )
    adata_cell.obs["component_subconsensus_gene_call_count"] = [int(c["subconsensus_count"]) for c in selected_components]
    adata_cell.obs["component_distinct_gene_count"] = [int(c["distinct_gene_count"]) for c in selected_components]
    adata_cell.obs["cell_subconsensus_count"] = np.asarray(cell_gene.sum(axis=1)).ravel().astype(np.int64)
    adata_cell.obs["cell_distinct_gene_count"] = np.asarray(cell_gene.getnnz(axis=1)).ravel().astype(np.int64)
    adata_cell.obs["cell_calling_cluster_key"] = str(cluster_key)

    _ensure_counts_layer(adata_cell, ANALYSIS_COUNTS_LAYER)
    if RAW_SUBCONSENSUS_COUNTS_LAYER not in adata_cell.layers:
        adata_cell.layers[RAW_SUBCONSENSUS_COUNTS_LAYER] = adata_cell.layers[ANALYSIS_COUNTS_LAYER].copy()

    _write_refined_hub_label_sidecar(
        filepath=filepath,
        n_hubs_original=n_hubs_initial,
        kept_original_indices=kept_original_indices,
        selected_components=selected_components,
        cell_labels=cell_labels,
    )

    if diagnostics_out is not None:
        diagnostics_out.update({
            "sample": sample_name,
            "filepath": filepath,
            "n_hubs_initial": n_hubs_initial,
            "n_features_initial": n_features_initial,
            "n_features_dropped_invalid": n_features_dropped_invalid,
            "n_features_kept": int(adata_cell.n_vars),
            "n_hubs_dropped_nan_cluster": n_hubs_dropped_nan_cluster,
            "n_hubs_dropped_drop_value": n_hubs_dropped_drop_value,
            "n_hubs_after_cluster_filter": n_hubs_after_cluster_filter,
            "n_hubs_with_zero_calls": n_hubs_with_zero_calls,
            "n_hubs_with_at_least_one_call": n_hubs_with_at_least_one_call,
            "n_cells_pre_qc": int(adata_cell.n_obs),
            "drop_value_breakdown": drop_value_breakdown,
            "per_hub_calls": hub_calls_full,
            "n_raw_clusters_pre_component_split": n_raw_clusters,
            "n_components_enumerated": int(len(components_all)),
            "n_components_with_calls": int(len(selected_components)),
            "n_raw_clusters_split": n_split_raw_clusters,
            "aborted": False,
            **feature_diag,
        })

    del adata_node, X_aug, X_hub, A, cell_gene
    gc.collect()
    return adata_cell

def apply_qc_gates(
    adata: ad.AnnData,
    *,
    diagnostics_out: Optional[Dict[str, object]] = None,
) -> ad.AnnData:
    """Apply per-cell and per-gene QC filters.

    If ``diagnostics_out`` is provided, the dictionary is populated with
    per-cell and per-gene metrics taken BEFORE any filter, plus the count of
    cells/genes rejected at each gate. This makes the cell-loss funnel
    auditable.
    """
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

    # Snapshot pre-filter per-cell and per-gene metrics.
    n_cells_pre_qc = int(adata.n_obs)
    n_genes_pre_qc = int(adata.n_vars)
    pre_total_counts = np.asarray(adata.obs["total_counts"].values, dtype=np.float64).copy()
    pre_n_genes_by_counts = np.asarray(adata.obs["n_genes_by_counts"].values, dtype=np.int64).copy()
    pre_cell_names = adata.obs_names.astype(str).to_numpy().copy()
    pre_cell_sample = (
        adata.obs["sample"].astype(str).to_numpy().copy()
        if "sample" in adata.obs.columns
        else np.array(["unknown"] * n_cells_pre_qc)
    )
    pre_n_hubs_with_call = (
        np.asarray(adata.obs["n_hubs_with_call"].values, dtype=np.int64).copy()
        if "n_hubs_with_call" in adata.obs.columns
        else np.full(n_cells_pre_qc, -1, dtype=np.int64)
    )
    pre_n_cells_by_counts_per_gene = np.asarray(
        adata.var["n_cells_by_counts"].values, dtype=np.int64
    ).copy()
    pre_gene_names = adata.var_names.astype(str).to_numpy().copy()

    # Gate G4: min_counts per cell.
    pass_min_counts_mask = pre_total_counts >= float(QC_MIN_COUNTS)
    n_cells_dropped_min_counts = int(np.sum(~pass_min_counts_mask))
    sc.pp.filter_cells(adata, min_counts=QC_MIN_COUNTS)

    # Gate G5: min_genes per cell. Only applies to cells that already passed
    # the min_counts gate, so we count rejections among the survivors of G4.
    pre_genes_for_survivors = pre_n_genes_by_counts[pass_min_counts_mask]
    pass_min_genes_mask_among_survivors = pre_genes_for_survivors >= int(QC_MIN_GENES)
    n_cells_dropped_min_genes = int(np.sum(~pass_min_genes_mask_among_survivors))
    sc.pp.filter_cells(adata, min_genes=QC_MIN_GENES)

    # Gate G6: min_cells per gene.
    pass_min_cells_mask = pre_n_cells_by_counts_per_gene >= int(QC_MIN_CELLS_PER_GENE)
    n_genes_dropped_min_cells = int(np.sum(~pass_min_cells_mask))
    sc.pp.filter_genes(adata, min_cells=QC_MIN_CELLS_PER_GENE)
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

    if diagnostics_out is not None:
        # Reason of rejection per cell. Cells that fail min_counts are
        # tagged "fail_min_counts"; cells that pass min_counts but fail
        # min_genes are tagged "fail_min_genes"; survivors are "pass".
        reason = np.full(n_cells_pre_qc, "pass", dtype=object)
        reason[~pass_min_counts_mask] = "fail_min_counts"
        passed_g4_idx = np.flatnonzero(pass_min_counts_mask)
        fail_g5_local = pre_n_genes_by_counts[pass_min_counts_mask] < int(QC_MIN_GENES)
        reason[passed_g4_idx[fail_g5_local]] = "fail_min_genes"

        diagnostics_out.update({
            "n_cells_pre_qc": n_cells_pre_qc,
            "n_genes_pre_qc": n_genes_pre_qc,
            "n_cells_dropped_min_counts": n_cells_dropped_min_counts,
            "n_cells_dropped_min_genes": n_cells_dropped_min_genes,
            "n_genes_dropped_min_cells": n_genes_dropped_min_cells,
            "n_cells_post_qc": int(adata.n_obs),
            "n_genes_post_qc": int(adata.n_vars),
            "qc_min_counts": int(QC_MIN_COUNTS),
            "qc_min_genes": int(QC_MIN_GENES),
            "qc_min_cells_per_gene": int(QC_MIN_CELLS_PER_GENE),
            "pre_cell_names": pre_cell_names,
            "pre_cell_sample": pre_cell_sample,
            "pre_total_counts": pre_total_counts,
            "pre_n_genes_by_counts": pre_n_genes_by_counts,
            "pre_n_hubs_with_call": pre_n_hubs_with_call,
            "pre_cell_reason": reason,
            "pre_gene_names": pre_gene_names,
            "pre_n_cells_by_counts_per_gene": pre_n_cells_by_counts_per_gene,
        })

    return adata


def _downsample_csr_rows_without_replacement(
    X: sparse.csr_matrix,
    *,
    target_counts: int,
    rng: np.random.Generator,
) -> sparse.csr_matrix:
    """Rarefy every CSR row to exactly ``target_counts`` integer events.

    Each nonzero count is treated as that many discrete sub-consensus gene-call
    events. Sampling is without replacement from the finite event multiset in
    each row, so a gene cannot receive more sampled events than it had before
    rarefaction.
    """
    X = X.tocsr(copy=True)
    if X.dtype.kind in {"f"}:
        X.data = np.rint(X.data).astype(np.int64, copy=False)
    else:
        X.data = X.data.astype(np.int64, copy=False)
    X.eliminate_zeros()

    t = int(target_counts)
    if t <= 0:
        raise ValueError(f"target_counts must be positive, got {target_counts}")

    new_indptr = np.zeros(X.shape[0] + 1, dtype=np.int64)
    new_indices: List[np.ndarray] = []
    new_data: List[np.ndarray] = []

    for i in range(X.shape[0]):
        a = int(X.indptr[i])
        b = int(X.indptr[i + 1])
        cols = X.indices[a:b]
        vals = X.data[a:b].astype(np.int64, copy=False)
        s = int(vals.sum())
        if s < t:
            raise ValueError(
                f"Cannot downsample row {i}: total count {s} is below target {t}. "
                "Filter/drop below-target cells before calling this function."
            )
        if s == t:
            out_cols = cols.astype(np.int32, copy=False)
            out_vals = vals.astype(np.int32, copy=False)
        else:
            # Same finite-multiset, without-replacement semantics as the prior
            # np.repeat(...)+rng.choice(..., replace=False) implementation, but
            # avoids materializing one event per count and runs inside NumPy.
            draw = rng.multivariate_hypergeometric(colors=vals, nsample=t)
            nz = draw > 0
            out_cols = cols[nz].astype(np.int32, copy=False)
            out_vals = draw[nz].astype(np.int32, copy=False)
        new_indices.append(out_cols)
        new_data.append(out_vals)
        new_indptr[i + 1] = new_indptr[i] + int(out_cols.size)

    if new_indices:
        indices = np.concatenate(new_indices).astype(np.int32, copy=False)
        data = np.concatenate(new_data).astype(np.int32, copy=False)
    else:
        indices = np.array([], dtype=np.int32)
        data = np.array([], dtype=np.int32)
    return sparse.csr_matrix((data, indices, new_indptr), shape=X.shape, dtype=np.int32)


def downsample_subconsensus_gene_calls_to_target(
    adata: ad.AnnData,
    *,
    target_counts: int = QC_MIN_COUNTS,
    layer: str = "counts",
    random_seed: int = RANDOM_SEED,
    reapply_min_genes: bool = True,
) -> pd.DataFrame:
    """Rarefy QC-passed cells to a fixed sub-consensus gene-call depth.

    ``aggregate_nodes_to_cells`` constructs the count matrix from one event per
    qualifying hub/sub-consensus x gene call (X >= MIN_READS). Therefore row
    sums in ``layer`` are exactly the quantity filtered by QC_MIN_COUNTS. This
    function equalizes that row-sum before normalized/HVG/PCA/Leiden/UMAP.

    Cells whose post-gene-filter total is below ``target_counts`` cannot be
    downsampled to the target and are dropped. This can happen if a cell passed
    QC_MIN_COUNTS before G6 but many of its calls belonged to genes later
    removed by QC_MIN_CELLS_PER_GENE.
    """
    if adata.n_obs == 0:
        return pd.DataFrame()

    target = int(target_counts)
    X = adata.layers[layer] if layer in adata.layers else adata.X
    X = X.tocsr() if sparse.issparse(X) else sparse.csr_matrix(np.asarray(X))
    if X.dtype.kind in {"f"}:
        X = X.copy()
        X.data = np.rint(X.data).astype(np.int64, copy=False)
    else:
        X = X.copy()
        X.data = X.data.astype(np.int64, copy=False)
    X.eliminate_zeros()

    cell_names_before = adata.obs_names.astype(str).to_numpy().copy()
    totals_before = np.asarray(X.sum(axis=1)).ravel().astype(np.int64)
    genes_before = np.asarray(X.getnnz(axis=1)).ravel().astype(np.int64)
    eligible = totals_before >= target

    dropped_df = pd.DataFrame()
    if not np.all(eligible):
        dropped_df = pd.DataFrame({
            "cell": cell_names_before[~eligible],
            "total_counts_before_downsample": totals_before[~eligible],
            "n_genes_before_downsample": genes_before[~eligible],
            "downsample_target_counts": target,
            "downsample_status": "dropped_below_target_after_gene_filter",
        })
        print(
            f"[Downsample] Dropping {int((~eligible).sum()):,} cells with "
            f"post-gene-filter total_counts < target ({target})."
        )
        adata._inplace_subset_obs(eligible)
        X = X[eligible, :].tocsr()
        cell_names_before = adata.obs_names.astype(str).to_numpy().copy()
        totals_before = totals_before[eligible]
        genes_before = genes_before[eligible]

    # Preserve the true pre-rarefaction cell x gene count matrix. These entries
    # quantify the number of qualifying subconsensus gene-call events per
    # retained cell-gene pair and are exported in sample_cell_connectivity_h5ad.
    if RAW_SUBCONSENSUS_COUNTS_LAYER not in adata.layers:
        adata.layers[RAW_SUBCONSENSUS_COUNTS_LAYER] = X.copy().astype(np.int32, copy=False)

    rng = np.random.default_rng(int(random_seed))
    X_ds = _downsample_csr_rows_without_replacement(X, target_counts=target, rng=rng)
    adata.layers[layer] = X_ds
    adata.X = X_ds.copy()
    adata.obs["total_counts_pre_downsample"] = totals_before
    adata.obs["n_genes_pre_downsample"] = genes_before
    adata.obs["subconsensus_gene_calls_downsample_target"] = target

    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    passes_min_genes = np.asarray(adata.obs["n_genes_by_counts"].values, dtype=np.int64) >= int(QC_MIN_GENES)

    kept_summary = pd.DataFrame({
        "cell": cell_names_before,
        "total_counts_before_downsample": totals_before,
        "n_genes_before_downsample": genes_before,
        "total_counts_after_downsample": np.asarray(adata.obs["total_counts"].values, dtype=np.int64),
        "n_genes_after_downsample": np.asarray(adata.obs["n_genes_by_counts"].values, dtype=np.int64),
        "downsample_target_counts": target,
        "passes_min_genes_after_downsample": passes_min_genes,
        "downsample_status": "kept_downsampled",
    })

    if reapply_min_genes and np.any(~passes_min_genes):
        print(
            f"[Downsample] Dropping {int((~passes_min_genes).sum()):,} cells with "
            f"n_genes_by_counts < QC_MIN_GENES ({QC_MIN_GENES}) after downsampling."
        )
        kept_summary.loc[~passes_min_genes, "downsample_status"] = "dropped_min_genes_after_downsample"
        sc.pp.filter_cells(adata, min_genes=QC_MIN_GENES)
        sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

    if not dropped_df.empty:
        kept_summary = pd.concat([kept_summary, dropped_df], ignore_index=True)
    return kept_summary


# =============================================================================
# Cell-filtering diagnostic: per-stage funnel and rank-order plots
# =============================================================================

def _rank_plot_with_threshold(
    values: np.ndarray,
    *,
    ax: "mpl.axes.Axes",
    threshold: Optional[float] = None,
    title: str = "",
    xlabel: str = "rank",
    ylabel: str = "value",
    log_y: bool = True,
    color_pass: str = "#2563eb",
    color_fail: str = "#dc2626",
    threshold_label: Optional[str] = None,
    annotate: bool = True,
) -> Dict[str, float]:
    """Draw a rank-ordered curve with a horizontal cutoff line.

    Returns a small stats dict so the caller can include numbers in a table.
    Values are plotted descending; the threshold line splits pass/fail and
    the rejected portion is shaded.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n_total = int(v.size)
    if n_total == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#666666")
        ax.set_title(title)
        return {"n_total": 0, "n_pass": 0, "n_fail": 0, "frac_fail": 0.0}

    order = np.argsort(-v)
    sorted_v = v[order]
    ranks = np.arange(1, n_total + 1)

    if threshold is None:
        ax.plot(ranks, sorted_v, color=color_pass, lw=1.4)
        n_pass = n_total
        n_fail = 0
    else:
        pass_mask = sorted_v >= float(threshold)
        n_pass = int(np.sum(pass_mask))
        n_fail = int(n_total - n_pass)
        # Plot pass and fail segments separately so the eye finds the knee.
        if n_pass > 0:
            ax.plot(ranks[:n_pass], sorted_v[:n_pass], color=color_pass, lw=1.4,
                    label=f"pass (n={n_pass:,})")
        if n_fail > 0:
            ax.plot(ranks[n_pass:], sorted_v[n_pass:], color=color_fail, lw=1.4,
                    label=f"fail (n={n_fail:,})")
            # Shade the rejected region.
            ax.axvspan(ranks[n_pass - 1] if n_pass > 0 else 0, ranks[-1],
                       color=color_fail, alpha=0.06, lw=0)
        ax.axhline(float(threshold), color="#111111", lw=0.8, ls="--",
                   label=(threshold_label or f"threshold = {threshold:g}"))

    if log_y:
        # Avoid log of zero by setting a small floor.
        floor = max(1e-3, float(np.min(sorted_v[sorted_v > 0])) * 0.5) if np.any(sorted_v > 0) else 1e-3
        ax.set_yscale("log")
        ax.set_ylim(bottom=floor)
    ax.set_xscale("log")
    ax.set_xlim(left=1, right=max(n_total, 2))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", lw=0.3, alpha=0.35)
    if annotate and threshold is not None:
        frac_fail = (n_fail / n_total) if n_total > 0 else 0.0
        ax.text(
            0.98, 0.96,
            f"n total = {n_total:,}\nn fail = {n_fail:,} ({100.0 * frac_fail:.1f}%)",
            ha="right", va="top", transform=ax.transAxes,
            fontsize=8, color="#111111",
            bbox=dict(facecolor="white", edgecolor="#888888", lw=0.4, pad=2.0, alpha=0.92),
        )
    if threshold is not None:
        ax.legend(loc="lower left", fontsize=7, frameon=False)
    return {
        "n_total": n_total,
        "n_pass": int(n_pass),
        "n_fail": int(n_fail),
        "frac_fail": (n_fail / n_total) if n_total > 0 else 0.0,
    }


def _save_diag_fig(fig: "mpl.figure.Figure", outpath: str) -> None:
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, format="pdf", dpi=300, bbox_inches="tight")
    png_path = os.path.splitext(outpath)[0] + ".png"
    fig.savefig(png_path, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_cell_filtering_diagnostics(
    per_file_diags: Sequence[Dict[str, object]],
    qc_diag: Dict[str, object],
    *,
    out_dir: str,
) -> pd.DataFrame:
    """Write the per-stage funnel TSV and the rank-order diagnostic plots.

    The diagnostic answers: at which gate does most of the data get rejected
    on the way to becoming a QC-passed cell? Stages tracked:

        S0 hubs_initial                  raw rows in input h5ad files
        S1 hubs_after_invalid_features   (informational; gene-level filter)
        S2 hubs_after_cluster_filter     drop NaN/-1 cluster hubs
        S3 hubs_with_at_least_one_call   hubs whose entries cleared MIN_READS
        S4 cells_pre_qc                  unique cluster ids (pre-aggregation)
        S5 cells_after_min_counts        pass QC_MIN_COUNTS
        S6 cells_after_min_genes         pass QC_MIN_GENES
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---- Per-stage summary table -------------------------------------------
    n_hubs_initial = int(sum(int(d.get("n_hubs_initial", 0)) for d in per_file_diags))
    n_features_dropped_invalid = int(sum(int(d.get("n_features_dropped_invalid", 0)) for d in per_file_diags))
    n_hubs_dropped_nan_cluster = int(sum(int(d.get("n_hubs_dropped_nan_cluster", 0)) for d in per_file_diags))
    n_hubs_dropped_drop_value = int(sum(int(d.get("n_hubs_dropped_drop_value", 0)) for d in per_file_diags))
    n_hubs_after_cluster_filter = int(sum(int(d.get("n_hubs_after_cluster_filter", 0)) for d in per_file_diags))
    n_hubs_with_zero_calls = int(sum(int(d.get("n_hubs_with_zero_calls", 0)) for d in per_file_diags))
    n_hubs_with_at_least_one_call = int(sum(int(d.get("n_hubs_with_at_least_one_call", 0)) for d in per_file_diags))
    n_cells_pre_qc_per_file = int(sum(int(d.get("n_cells_pre_qc", 0)) for d in per_file_diags))

    # The QC stage may operate on a different (concatenated) cell count if any
    # cluster ids collide across samples (the script makes obs_names unique
    # post-concat), so prefer qc_diag's pre-QC count when available.
    n_cells_pre_qc = int(qc_diag.get("n_cells_pre_qc", n_cells_pre_qc_per_file))
    n_cells_dropped_min_counts = int(qc_diag.get("n_cells_dropped_min_counts", 0))
    n_cells_dropped_min_genes = int(qc_diag.get("n_cells_dropped_min_genes", 0))
    n_cells_post_qc = int(qc_diag.get("n_cells_post_qc", 0))
    n_genes_pre_qc = int(qc_diag.get("n_genes_pre_qc", 0))
    n_genes_dropped_min_cells = int(qc_diag.get("n_genes_dropped_min_cells", 0))
    n_genes_post_qc = int(qc_diag.get("n_genes_post_qc", 0))

    cells_after_min_counts = max(n_cells_pre_qc - n_cells_dropped_min_counts, 0)
    cells_after_min_genes = max(cells_after_min_counts - n_cells_dropped_min_genes, 0)

    summary_rows: List[Dict[str, object]] = [
        {
            "stage_id": "S0",
            "stage": "hubs_initial",
            "level": "hub",
            "n_input": n_hubs_initial,
            "n_kept": n_hubs_initial,
            "n_rejected": 0,
            "frac_rejected_of_input": 0.0,
            "frac_rejected_of_S0_hubs": 0.0,
            "filter": "(none)",
            "threshold": "",
        },
        {
            "stage_id": "G1",
            "stage": "features_invalid_dropped",
            "level": "gene",
            "n_input": "",
            "n_kept": "",
            "n_rejected": n_features_dropped_invalid,
            "frac_rejected_of_input": "",
            "frac_rejected_of_S0_hubs": "",
            "filter": "drop gene_id in {NaN, '', INTERGENIC, INTRONIC, UNKNOWN, UNMAPPED}",
            "threshold": "",
        },
        {
            "stage_id": "G2",
            "stage": "hubs_cluster_filter",
            "level": "hub",
            "n_input": n_hubs_initial,
            "n_kept": n_hubs_after_cluster_filter,
            "n_rejected": int(n_hubs_dropped_nan_cluster + n_hubs_dropped_drop_value),
            "frac_rejected_of_input": (
                (n_hubs_dropped_nan_cluster + n_hubs_dropped_drop_value) / n_hubs_initial
                if n_hubs_initial > 0 else 0.0
            ),
            "frac_rejected_of_S0_hubs": (
                (n_hubs_dropped_nan_cluster + n_hubs_dropped_drop_value) / n_hubs_initial
                if n_hubs_initial > 0 else 0.0
            ),
            "filter": f"drop hubs with NaN cluster or cluster in {sorted(DROP_CLUSTER_VALUES)}",
            "threshold": "",
        },
        {
            "stage_id": "G3",
            "stage": "hubs_with_at_least_one_call",
            "level": "hub",
            "n_input": n_hubs_after_cluster_filter,
            "n_kept": n_hubs_with_at_least_one_call,
            "n_rejected": n_hubs_with_zero_calls,
            "frac_rejected_of_input": (
                n_hubs_with_zero_calls / n_hubs_after_cluster_filter
                if n_hubs_after_cluster_filter > 0 else 0.0
            ),
            "frac_rejected_of_S0_hubs": (
                n_hubs_with_zero_calls / n_hubs_initial
                if n_hubs_initial > 0 else 0.0
            ),
            "filter": "hub contributes nothing if no entry has X >= MIN_READS",
            "threshold": f"MIN_READS = {MIN_READS}",
        },
        {
            "stage_id": "S4",
            "stage": "cells_pre_qc",
            "level": "cell",
            "n_input": n_hubs_with_at_least_one_call,
            "n_kept": n_cells_pre_qc,
            "n_rejected": "",
            "frac_rejected_of_input": "",
            "frac_rejected_of_S0_hubs": "",
            "filter": "aggregate hubs by cluster_id and disconnected component (one candidate cell per nonempty component)",
            "threshold": "",
        },
        {
            "stage_id": "G4",
            "stage": "cells_after_min_counts",
            "level": "cell",
            "n_input": n_cells_pre_qc,
            "n_kept": cells_after_min_counts,
            "n_rejected": n_cells_dropped_min_counts,
            "frac_rejected_of_input": (
                n_cells_dropped_min_counts / n_cells_pre_qc if n_cells_pre_qc > 0 else 0.0
            ),
            "frac_rejected_of_S0_hubs": "",
            "filter": "drop cell if total_counts < QC_MIN_COUNTS",
            "threshold": f"QC_MIN_COUNTS = {QC_MIN_COUNTS}",
        },
        {
            "stage_id": "G5",
            "stage": "cells_after_min_genes",
            "level": "cell",
            "n_input": cells_after_min_counts,
            "n_kept": cells_after_min_genes,
            "n_rejected": n_cells_dropped_min_genes,
            "frac_rejected_of_input": (
                n_cells_dropped_min_genes / cells_after_min_counts
                if cells_after_min_counts > 0 else 0.0
            ),
            "frac_rejected_of_S0_hubs": "",
            "filter": "drop cell if n_genes_by_counts < QC_MIN_GENES",
            "threshold": f"QC_MIN_GENES = {QC_MIN_GENES}",
        },
        {
            "stage_id": "S6",
            "stage": "cells_post_qc",
            "level": "cell",
            "n_input": cells_after_min_genes,
            "n_kept": n_cells_post_qc,
            "n_rejected": int(max(cells_after_min_genes - n_cells_post_qc, 0)),
            "frac_rejected_of_input": "",
            "frac_rejected_of_S0_hubs": "",
            "filter": "(reconciliation: should equal cells_after_min_genes)",
            "threshold": "",
        },
        {
            "stage_id": "G6",
            "stage": "genes_after_min_cells",
            "level": "gene",
            "n_input": n_genes_pre_qc,
            "n_kept": n_genes_post_qc,
            "n_rejected": n_genes_dropped_min_cells,
            "frac_rejected_of_input": (
                n_genes_dropped_min_cells / n_genes_pre_qc if n_genes_pre_qc > 0 else 0.0
            ),
            "frac_rejected_of_S0_hubs": "",
            "filter": "drop gene if n_cells_by_counts < QC_MIN_CELLS_PER_GENE",
            "threshold": f"QC_MIN_CELLS_PER_GENE = {QC_MIN_CELLS_PER_GENE}",
        },
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, DIAG_TABLE_PER_STAGE), sep="\t", index=False)

    # ---- Per-cell metrics table -------------------------------------------
    pre_cell_names = np.asarray(qc_diag.get("pre_cell_names", np.array([])))
    if pre_cell_names.size > 0:
        per_cell_df = pd.DataFrame({
            "cell": pre_cell_names,
            "sample": np.asarray(qc_diag.get("pre_cell_sample", [])),
            "total_counts": np.asarray(qc_diag.get("pre_total_counts", [])),
            "n_genes_by_counts": np.asarray(qc_diag.get("pre_n_genes_by_counts", [])),
            "n_hubs_with_call": np.asarray(qc_diag.get("pre_n_hubs_with_call", [])),
            "qc_status": np.asarray(qc_diag.get("pre_cell_reason", [])),
        })
        per_cell_df.to_csv(os.path.join(out_dir, DIAG_TABLE_PER_CELL), sep="\t", index=False)
    else:
        per_cell_df = pd.DataFrame()

    # ---- Per-gene metrics table -------------------------------------------
    pre_gene_names = np.asarray(qc_diag.get("pre_gene_names", np.array([])))
    if pre_gene_names.size > 0:
        per_gene_df = pd.DataFrame({
            "gene": pre_gene_names,
            "n_cells_by_counts_pre_qc": np.asarray(qc_diag.get("pre_n_cells_by_counts_per_gene", [])),
            "passes_min_cells": (
                np.asarray(qc_diag.get("pre_n_cells_by_counts_per_gene", []), dtype=np.int64)
                >= int(QC_MIN_CELLS_PER_GENE)
            ),
        })
        per_gene_df.to_csv(os.path.join(out_dir, DIAG_TABLE_PER_GENE), sep="\t", index=False)
    else:
        per_gene_df = pd.DataFrame()

    # ---- Per-hub metrics table (across all samples) -----------------------
    per_hub_rows: List[pd.DataFrame] = []
    for d in per_file_diags:
        per_hub = np.asarray(d.get("per_hub_calls", np.array([])), dtype=np.int64)
        if per_hub.size == 0:
            continue
        per_hub_rows.append(pd.DataFrame({
            "sample": str(d.get("sample", "unknown")),
            "hub_index": np.arange(per_hub.size, dtype=np.int64),
            "n_calls_in_hub": per_hub,
            "has_any_call": per_hub > 0,
        }))
    if per_hub_rows:
        per_hub_df = pd.concat(per_hub_rows, ignore_index=True)
        per_hub_df.to_csv(os.path.join(out_dir, DIAG_TABLE_PER_HUB), sep="\t", index=False)
    else:
        per_hub_df = pd.DataFrame()

    # ---- Console summary ---------------------------------------------------
    print("\n" + "-" * 72)
    print("[Diag] Cell-filtering funnel (cumulative across all input files)")
    print("-" * 72)
    fmt = "{:<6} {:<32} {:<6} {:>14} {:>14} {:>14} {:>10}"
    print(fmt.format("stage", "name", "level", "n_input", "n_kept", "n_rejected", "%reject"))
    for r in summary_rows:
        n_in = r["n_input"] if r["n_input"] != "" else "-"
        n_k = r["n_kept"] if r["n_kept"] != "" else "-"
        n_r = r["n_rejected"] if r["n_rejected"] != "" else "-"
        if r["frac_rejected_of_input"] == "" or r["frac_rejected_of_input"] is None:
            pct = "-"
        else:
            pct = f"{100.0 * float(r['frac_rejected_of_input']):.1f}%"
        n_in_s = f"{int(n_in):,}" if isinstance(n_in, (int, np.integer)) else str(n_in)
        n_k_s = f"{int(n_k):,}" if isinstance(n_k, (int, np.integer)) else str(n_k)
        n_r_s = f"{int(n_r):,}" if isinstance(n_r, (int, np.integer)) else str(n_r)
        print(fmt.format(str(r["stage_id"]), str(r["stage"])[:32], str(r["level"]),
                         n_in_s, n_k_s, n_r_s, pct))
    print("-" * 72)

    # ---- Rank-order plots --------------------------------------------------
    # 1) Per-hub: rank-ordered total calls per hub, threshold at >= 1 call.
    if not per_hub_df.empty:
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        # Combined across all samples.
        v = per_hub_df["n_calls_in_hub"].to_numpy(dtype=float)
        _rank_plot_with_threshold(
            v, ax=ax, threshold=1.0,
            title=f"Per-hub call count (rank-ordered, all samples; MIN_READS={MIN_READS})",
            xlabel="hub rank (descending by call count)",
            ylabel="number of gene calls in hub (>= MIN_READS)",
            log_y=True,
            threshold_label="threshold = 1 call (G3)",
        )
        # Overlay per-sample curves in faded color so any one sample dominating is visible.
        for sample, grp in per_hub_df.groupby("sample"):
            sv = grp["n_calls_in_hub"].to_numpy(dtype=float)
            sv = np.sort(sv)[::-1]
            sv = np.where(sv > 0, sv, 0.5)  # avoid log(0)
            ax.plot(np.arange(1, sv.size + 1), sv, color="#666666", lw=0.4, alpha=0.5,
                    label=f"sample {sample}")
        ax.legend(loc="upper right", fontsize=6, frameon=False, ncol=2)
        _save_diag_fig(fig, os.path.join(out_dir, DIAG_FIG_RANK_HUB_CALLS))

    # 2) Per-cell: rank-ordered hubs-with-call per cell.
    if not per_cell_df.empty and "n_hubs_with_call" in per_cell_df.columns:
        v = per_cell_df["n_hubs_with_call"].to_numpy(dtype=float)
        v = v[v >= 0]  # drop sentinel
        if v.size > 0:
            fig, ax = plt.subplots(figsize=(7.0, 4.4))
            _rank_plot_with_threshold(
                v, ax=ax, threshold=None,
                title="Per-cell sub-consensus depth (rank-ordered)",
                xlabel="cell rank (descending)",
                ylabel="n hubs with call per cell",
                log_y=True,
            )
            _save_diag_fig(fig, os.path.join(out_dir, DIAG_FIG_RANK_CELL_HUBS))

    # 3) Per-cell: rank-ordered total_counts with QC_MIN_COUNTS threshold.
    if not per_cell_df.empty and "total_counts" in per_cell_df.columns:
        v = per_cell_df["total_counts"].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        _rank_plot_with_threshold(
            v, ax=ax, threshold=float(QC_MIN_COUNTS),
            title=f"Per-cell total_counts (rank-ordered)  —  gate G4: QC_MIN_COUNTS={QC_MIN_COUNTS}",
            xlabel="cell rank (descending by total_counts)",
            ylabel="total_counts per cell (binary calls summed)",
            log_y=True,
            threshold_label=f"QC_MIN_COUNTS = {QC_MIN_COUNTS}",
        )
        _save_diag_fig(fig, os.path.join(out_dir, DIAG_FIG_RANK_CELL_COUNTS))

    # 4) Per-cell: rank-ordered n_genes_by_counts with QC_MIN_GENES threshold.
    if not per_cell_df.empty and "n_genes_by_counts" in per_cell_df.columns:
        v = per_cell_df["n_genes_by_counts"].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        _rank_plot_with_threshold(
            v, ax=ax, threshold=float(QC_MIN_GENES),
            title=f"Per-cell n_genes_by_counts (rank-ordered)  —  gate G5: QC_MIN_GENES={QC_MIN_GENES}",
            xlabel="cell rank (descending by n_genes_by_counts)",
            ylabel="n_genes_by_counts per cell",
            log_y=True,
            threshold_label=f"QC_MIN_GENES = {QC_MIN_GENES}",
        )
        _save_diag_fig(fig, os.path.join(out_dir, DIAG_FIG_RANK_CELL_GENES))

    # 5) Per-gene: rank-ordered n_cells_by_counts with QC_MIN_CELLS_PER_GENE.
    if not per_gene_df.empty:
        v = per_gene_df["n_cells_by_counts_pre_qc"].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        _rank_plot_with_threshold(
            v, ax=ax, threshold=float(QC_MIN_CELLS_PER_GENE),
            title=f"Per-gene cell support (rank-ordered)  —  gate G6: QC_MIN_CELLS_PER_GENE={QC_MIN_CELLS_PER_GENE}",
            xlabel="gene rank (descending by n_cells_by_counts)",
            ylabel="n cells in which gene is detected",
            log_y=True,
            threshold_label=f"QC_MIN_CELLS_PER_GENE = {QC_MIN_CELLS_PER_GENE}",
        )
        _save_diag_fig(fig, os.path.join(out_dir, DIAG_FIG_RANK_GENE_CELLS))

    # 6) Joint scatter: total_counts vs n_genes_by_counts, colored by qc_status.
    if not per_cell_df.empty:
        fig, ax = plt.subplots(figsize=(6.4, 5.6))
        status_colors = {
            "fail_min_counts": "#dc2626",
            "fail_min_genes": "#f59e0b",
            "pass": "#2563eb",
        }
        z_order = {"fail_min_counts": 1, "fail_min_genes": 2, "pass": 3}
        for status, grp in per_cell_df.groupby("qc_status"):
            ax.scatter(
                np.maximum(grp["total_counts"].to_numpy(dtype=float), 0.5),
                np.maximum(grp["n_genes_by_counts"].to_numpy(dtype=float), 0.5),
                s=4.0,
                c=status_colors.get(str(status), "#888888"),
                alpha=0.45,
                lw=0,
                label=f"{status} (n={len(grp):,})",
                zorder=z_order.get(str(status), 0),
            )
        ax.axvline(float(QC_MIN_COUNTS), color="#111111", lw=0.7, ls="--",
                   label=f"QC_MIN_COUNTS = {QC_MIN_COUNTS}")
        ax.axhline(float(QC_MIN_GENES), color="#111111", lw=0.7, ls=":",
                   label=f"QC_MIN_GENES = {QC_MIN_GENES}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("total_counts per cell")
        ax.set_ylabel("n_genes_by_counts per cell")
        ax.set_title("Joint per-cell QC: total_counts vs detected genes\n(red=fails G4; orange=fails G5; blue=passes both)")
        ax.legend(loc="lower right", fontsize=7, frameon=False)
        ax.grid(True, which="both", lw=0.3, alpha=0.35)
        _save_diag_fig(fig, os.path.join(out_dir, DIAG_FIG_JOINT))

    # 7) Funnel/waterfall bar plot showing the cell-creation pipeline.
    funnel_labels = [
        f"S0 hubs_initial",
        f"G2 cluster filter (drop NaN/{','.join(sorted(DROP_CLUSTER_VALUES)) or '-'})",
        f"G3 >=1 call (MIN_READS={MIN_READS})",
        f"S4 cells (unique cluster_ids)",
        f"G4 pass min_counts ({QC_MIN_COUNTS})",
        f"G5 pass min_genes ({QC_MIN_GENES})",
    ]
    funnel_values = [
        n_hubs_initial,
        n_hubs_after_cluster_filter,
        n_hubs_with_at_least_one_call,
        n_cells_pre_qc,
        cells_after_min_counts,
        cells_after_min_genes,
    ]
    funnel_levels = ["hub", "hub", "hub", "cell", "cell", "cell"]

    fig, ax = plt.subplots(figsize=(8.4, max(3.6, 0.55 * len(funnel_labels) + 1.0)))
    ypos = np.arange(len(funnel_labels))[::-1]
    bar_colors = ["#475569" if lv == "hub" else "#2563eb" for lv in funnel_levels]
    ax.barh(ypos, funnel_values, color=bar_colors, edgecolor="#111111", lw=0.4, height=0.62)
    for y, v, lv in zip(ypos, funnel_values, funnel_levels):
        anchor = n_hubs_initial if lv == "hub" else max(n_cells_pre_qc, 1)
        pct_anchor = (100.0 * v / anchor) if anchor > 0 else 0.0
        ax.text(v, y, f"  {v:,}  ({pct_anchor:.1f}% of {'S0 hubs' if lv == 'hub' else 'S4 cells'})",
                va="center", ha="left", fontsize=8, color="#111111")
    ax.set_yticks(ypos)
    ax.set_yticklabels(funnel_labels, fontsize=8)
    if max(funnel_values) > 0:
        ax.set_xlim(0, max(funnel_values) * 1.40)
    ax.set_xlabel("count")
    ax.set_title("Cell-filtering funnel: where does the data go?")
    ax.grid(True, axis="x", lw=0.3, alpha=0.35)
    legend_handles = [
        mpatches.Patch(facecolor="#475569", edgecolor="#111111", label="hub-level stage"),
        mpatches.Patch(facecolor="#2563eb", edgecolor="#111111", label="cell-level stage"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=7, frameon=False)
    _save_diag_fig(fig, os.path.join(out_dir, DIAG_FIG_FUNNEL))

    print(f"[Diag] Wrote diagnostic outputs under: {out_dir}")
    print(f"  - {DIAG_TABLE_PER_STAGE}")
    print(f"  - {DIAG_TABLE_PER_CELL}")
    print(f"  - {DIAG_TABLE_PER_GENE}")
    print(f"  - {DIAG_TABLE_PER_HUB}")
    print(f"  - {DIAG_FIG_FUNNEL}")
    print(f"  - {DIAG_FIG_RANK_HUB_CALLS}")
    print(f"  - {DIAG_FIG_RANK_CELL_HUBS}")
    print(f"  - {DIAG_FIG_RANK_CELL_COUNTS}")
    print(f"  - {DIAG_FIG_RANK_CELL_GENES}")
    print(f"  - {DIAG_FIG_RANK_GENE_CELLS}")
    print(f"  - {DIAG_FIG_JOINT}")

    return summary_df


def process_for_leiden(adata: ad.AnnData) -> bool:
    """Compute normalized/log expression, HVGs, PCA, neighbors, and UMAP.

    The UMAP is computed from the same PCA-neighbor graph used for Leiden, with
    standard Scanpy-style defaults appropriate for single-cell gene-expression
    analysis. This provides a nonlinear 2D companion to the PCA panel without
    altering the clustering graph construction itself.
    """
    if adata.n_obs < max(10, GLOBAL_N_NEIGHBORS + 1):
        return False
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if "sample" in adata.obs.columns and adata.obs["sample"].nunique() > 1:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3.0, min_disp=0.5, batch_key="sample")
    else:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3.0, min_disp=0.5)
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, svd_solver="arpack", use_highly_variable=True, random_state=RANDOM_SEED)
    sc.pp.neighbors(adata, n_neighbors=GLOBAL_N_NEIGHBORS, n_pcs=GLOBAL_N_PCS, random_state=RANDOM_SEED)
    try:
        sc.tl.umap(adata, min_dist=0.5, spread=1.0, random_state=int(UMAP_RANDOM_STATE))
    except TypeError:
        sc.tl.umap(adata)
    return True


def _run_leiden(
    adata: ad.AnnData,
    *,
    resolution: float,
    key_added: str,
    random_state: Optional[int] = None,
) -> None:
    """Run Scanpy Leiden with compatibility across Scanpy versions."""
    try:
        if random_state is not None:
            sc.tl.leiden(adata, resolution=float(resolution), key_added=str(key_added), random_state=int(random_state))
        else:
            sc.tl.leiden(adata, resolution=float(resolution), key_added=str(key_added))
    except TypeError:
        sc.tl.leiden(adata, resolution=float(resolution), key_added=str(key_added))


def auto_tune_leiden_resolution_for_palette(
    adata: ad.AnnData,
    *,
    key_added: str = LEIDEN_KEY,
    resolution_grid: Sequence[float] = LEIDEN_RESOLUTION_GRID,
    min_clusters: int = LEIDEN_MIN_CLUSTERS,
    max_clusters: int = LEIDEN_MAX_DISTINCT_COLORS,
    min_cluster_size: int = LEIDEN_MIN_CLUSTER_SIZE,
    random_state: int = LEIDEN_RANDOM_STATE,
    out_table_path: Optional[str] = None,
) -> float:
    """Choose the coarsest resolution yielding the largest cluster count within the color budget."""
    tmp_key = "__leiden_tmp__"
    rows: List[Dict[str, object]] = []
    for res in resolution_grid:
        res_f = float(res)
        try:
            _run_leiden(adata, resolution=res_f, key_added=tmp_key, random_state=int(random_state))
            labs = adata.obs[tmp_key].astype(str)
            n_clust = int(labs.nunique())
            vc = labs.value_counts()
            min_size = int(vc.min()) if vc.size else 0
            rows.append({"resolution": res_f, "n_clusters": n_clust, "min_cluster_size": min_size})
        except Exception as e:
            rows.append({"resolution": res_f, "n_clusters": np.nan, "min_cluster_size": np.nan, "error": str(e)})

    if tmp_key in adata.obs.columns:
        try:
            del adata.obs[tmp_key]
        except Exception:
            pass

    df = pd.DataFrame(rows).sort_values("resolution")
    if out_table_path:
        os.makedirs(os.path.dirname(out_table_path), exist_ok=True)
        df.to_csv(out_table_path, sep="\t", index=False)

    df_ok = df.loc[df["n_clusters"].notna()].copy()
    if df_ok.shape[0] == 0:
        print("[Leiden] Auto-tuning failed; using fallback resolution.")
        return float(LEIDEN_RESOLUTION)

    def _select(df_in: pd.DataFrame) -> Optional[pd.Series]:
        if df_in.shape[0] == 0:
            return None
        best_n = int(df_in["n_clusters"].max())
        return df_in.loc[df_in["n_clusters"] == best_n].sort_values("resolution").iloc[0]

    best = _select(df_ok.loc[
        (df_ok["n_clusters"] >= int(min_clusters))
        & (df_ok["n_clusters"] <= int(max_clusters))
        & (df_ok["min_cluster_size"] >= int(min_cluster_size))
    ])
    if best is None:
        best = _select(df_ok.loc[(df_ok["n_clusters"] >= int(min_clusters)) & (df_ok["n_clusters"] <= int(max_clusters))])
    if best is None:
        df_under = df_ok.loc[df_ok["n_clusters"] <= int(max_clusters)].copy()
        if df_under.shape[0] > 0:
            df_under["gap"] = int(max_clusters) - df_under["n_clusters"].astype(int)
            best = df_under.sort_values(["gap", "resolution"]).iloc[0]
    if best is None:
        best = df_ok.sort_values("resolution").iloc[0]
        print(
            f"[Leiden] Warning: smallest tested resolution produced {int(best['n_clusters'])} clusters, "
            f"exceeding max_clusters={max_clusters}."
        )

    best_res = float(best["resolution"])
    print(
        f"[Leiden] Auto-tuned resolution={best_res:.3g} -> n_clusters={int(best['n_clusters'])} "
        f"(min_cluster_size={int(best['min_cluster_size'])}); palette_budget={max_clusters}"
    )
    return best_res


# =============================================================================
# Balanced pseudobulk Leiden DE
# =============================================================================

def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = int(p.size)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * (n / (np.arange(1, n + 1, dtype=float)))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out


def _as_csr_counts_matrix(adata: ad.AnnData, layer: str = "counts") -> sparse.csr_matrix:
    X = adata.layers[layer] if layer in adata.layers else adata.X
    if sparse.issparse(X):
        X = X.tocsr()
    else:
        X = sparse.csr_matrix(np.asarray(X))
    if X.dtype.kind in {"f"}:
        X = X.copy()
        X.data = np.rint(X.data).astype(np.int64, copy=False)
    else:
        X.data = X.data.astype(np.int64, copy=False)
    return X


def _pseudobulk_sum_depth_rarefied(
    X: sparse.csr_matrix,
    cell_idx: np.ndarray,
    *,
    target_sum: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sum counts after rarefying each selected cell to target_sum."""
    cell_idx = np.asarray(cell_idx, dtype=np.int64)
    n_genes = int(X.shape[1])
    out = np.zeros(n_genes, dtype=np.int64)
    indptr, indices, data = X.indptr, X.indices, X.data
    t = int(target_sum)

    for i in cell_idx:
        a = int(indptr[i])
        b = int(indptr[i + 1])
        if b <= a:
            continue
        cols = indices[a:b]
        vals = data[a:b].astype(np.int64, copy=False)
        s = int(vals.sum())
        if s <= 0:
            continue
        if s == t:
            out[cols] += vals
        elif s > t:
            p = vals / float(s)
            draw = rng.multinomial(t, p)
            nz = draw > 0
            if np.any(nz):
                out[cols[nz]] += draw[nz]
        else:
            out[cols] += vals
    return out


def run_balanced_leiden_de(
    adata: ad.AnnData,
    *,
    leiden_key: str,
    sample_col: str = "sample",
    layer: str = "counts",
    top_n: int = 0,
    cells_per_sample: int = 80,
    target_sum: int = 100,
    min_samples: int = 2,
    pseudocount: float = 0.5,
    random_seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Balanced pseudobulk DE for Leiden cluster-vs-rest contrasts.

    Each eligible sample contributes paired pseudobulk replicates using the same
    number of cells in-cluster and out-of-cluster, with each cell rarefied to the
    same depth before summation.
    """
    if leiden_key not in adata.obs.columns:
        raise KeyError(f"Missing obs['{leiden_key}']")
    if sample_col not in adata.obs.columns:
        raise KeyError(f"Missing obs['{sample_col}']")

    clust = adata.obs[leiden_key].astype(str)
    sample = adata.obs[sample_col].astype(str)
    cluster_sizes = clust.value_counts().sort_values(ascending=False)
    top_n_int = int(top_n) if top_n is not None else 0
    if top_n_int <= 0 or top_n_int >= int(len(cluster_sizes)):
        top_clusters = cluster_sizes.index.astype(str).tolist()
    else:
        top_clusters = cluster_sizes.head(top_n_int).index.astype(str).tolist()
    if len(top_clusters) == 0:
        raise ValueError("No clusters available for DE")

    X = _as_csr_counts_matrix(adata, layer=layer)
    genes = adata.var_names.astype(str).to_numpy()
    n_cells_total = clust.value_counts().to_dict()
    uniq_samples = pd.unique(sample)

    from scipy.stats import t as tdist

    all_rows: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []

    for k, c in enumerate(top_clusters):
        mask_in = clust.values == str(c)
        mask_out = ~mask_in

        eligible_samples: List[str] = []
        for s in uniq_samples:
            sm = sample.values == str(s)
            n_in = int(np.sum(sm & mask_in))
            n_out = int(np.sum(sm & mask_out))
            if n_in >= int(cells_per_sample) and n_out >= int(cells_per_sample):
                eligible_samples.append(str(s))

        if len(eligible_samples) < int(min_samples):
            summary_rows.append({
                "cluster": str(c),
                "n_cells_total": int(n_cells_total.get(str(c), int(np.sum(mask_in)))),
                "n_samples_used": int(len(eligible_samples)),
                "cells_per_sample": int(cells_per_sample),
                "target_sum_per_cell": int(target_sum),
                "status": "skipped_insufficient_samples",
            })
            continue

        stable = int(np.abs(np.int64(np.frombuffer(str(c).encode("utf-8"), dtype=np.uint8).sum())))
        rng = np.random.default_rng(int(random_seed) + 1009 * k + stable)

        C_list: List[np.ndarray] = []
        R_list: List[np.ndarray] = []
        used_samples: List[str] = []
        for s in eligible_samples:
            sm = sample.values == str(s)
            in_idx = np.flatnonzero(sm & mask_in)
            out_idx = np.flatnonzero(sm & mask_out)
            sel_in = rng.choice(in_idx, size=int(cells_per_sample), replace=False)
            sel_out = rng.choice(out_idx, size=int(cells_per_sample), replace=False)
            C_list.append(_pseudobulk_sum_depth_rarefied(X, sel_in, target_sum=int(target_sum), rng=rng))
            R_list.append(_pseudobulk_sum_depth_rarefied(X, sel_out, target_sum=int(target_sum), rng=rng))
            used_samples.append(str(s))

        C = np.vstack(C_list).astype(np.float64, copy=False)
        R = np.vstack(R_list).astype(np.float64, copy=False)
        lib_C = C.sum(axis=1, keepdims=True)
        lib_R = R.sum(axis=1, keepdims=True)

        pc = float(pseudocount)
        cpm_C = (C + pc) / (lib_C + pc * C.shape[1]) * 1e6
        cpm_R = (R + pc) / (lib_R + pc * R.shape[1]) * 1e6
        log2fc = np.log2(cpm_C) - np.log2(cpm_R)
        mean_log2fc = np.mean(log2fc, axis=0)

        n_rep = int(log2fc.shape[0])
        if n_rep < 2:
            pvals = np.ones(log2fc.shape[1], dtype=float)
        else:
            sd = np.std(log2fc, axis=0, ddof=1)
            se = sd / np.sqrt(float(n_rep))
            tstat = np.divide(mean_log2fc, se, out=np.zeros_like(mean_log2fc), where=se > 0)
            pvals = 2.0 * tdist.sf(np.abs(tstat), df=n_rep - 1)
            pvals = np.asarray(pvals, dtype=float)
            pvals[~np.isfinite(pvals)] = 1.0
        fdr = _bh_fdr(pvals)

        X_in = X[mask_in, :]
        X_out = X[mask_out, :]
        pct_in = (np.asarray(X_in.getnnz(axis=0)).ravel() / max(int(X_in.shape[0]), 1)).astype(np.float32)
        pct_out = (np.asarray(X_out.getnnz(axis=0)).ravel() / max(int(X_out.shape[0]), 1)).astype(np.float32)
        mean_logcpm_in = np.mean(np.log10(cpm_C + 1.0), axis=0)
        mean_logcpm_out = np.mean(np.log10(cpm_R + 1.0), axis=0)

        all_rows.append(pd.DataFrame({
            "cluster": str(c),
            "gene": genes,
            "mean_log2fc": mean_log2fc.astype(np.float32),
            "pval": pvals.astype(np.float64),
            "fdr": fdr.astype(np.float64),
            "pct_in": pct_in,
            "pct_out": pct_out,
            "mean_log10_cpm_in": mean_logcpm_in.astype(np.float32),
            "mean_log10_cpm_out": mean_logcpm_out.astype(np.float32),
            "n_samples_used": int(n_rep),
            "cells_per_sample": int(cells_per_sample),
            "target_sum_per_cell": int(target_sum),
        }))

        summary_rows.append({
            "cluster": str(c),
            "n_cells_total": int(n_cells_total.get(str(c), int(np.sum(mask_in)))),
            "n_samples_used": int(n_rep),
            "cells_per_sample": int(cells_per_sample),
            "target_sum_per_cell": int(target_sum),
            "pseudobulk_libsize_cluster_mean": float(np.mean(lib_C)) if lib_C.size else float("nan"),
            "pseudobulk_libsize_rest_mean": float(np.mean(lib_R)) if lib_R.size else float("nan"),
            "eligible_samples": ",".join(used_samples),
            "status": "ok",
        })

    de_all = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(columns=["cluster", "gene", "mean_log2fc", "pval", "fdr"])
    de_summary = pd.DataFrame(summary_rows)
    return de_all, de_summary


def run_descriptive_leiden_marker_map_de(
    adata: ad.AnnData,
    *,
    leiden_key: str,
    sample_col: str = "sample",
    layer: str = "counts",
    top_n: int = 0,
    target_sum: int = 100,
    pseudocount: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cell-level descriptive Leiden cluster-vs-rest marker summaries.

    This is intentionally not inferential DE. It exists for one-specimen runs
    where the figure's purpose is to assign Leiden clusters to marker-gene
    modules. Rows are shaped like the balanced-DE table so the existing marker
    map can be reused, but p-values/FDR are placeholders and should not be used
    as evidence across biological replicates.
    """
    if leiden_key not in adata.obs.columns:
        raise KeyError(f"Missing obs['{leiden_key}']")

    clust = adata.obs[leiden_key].astype(str)
    sample_vals = (
        adata.obs[sample_col].astype(str)
        if sample_col in adata.obs.columns
        else pd.Series(["single_sample"] * adata.n_obs, index=adata.obs_names)
    )
    cluster_sizes = clust.value_counts().sort_values(ascending=False)
    top_n_int = int(top_n) if top_n is not None else 0
    if top_n_int <= 0 or top_n_int >= int(len(cluster_sizes)):
        top_clusters = cluster_sizes.index.astype(str).tolist()
    else:
        top_clusters = cluster_sizes.head(top_n_int).index.astype(str).tolist()
    if len(top_clusters) == 0:
        raise ValueError("No clusters available for descriptive marker-map summaries")

    X = _as_csr_counts_matrix(adata, layer=layer)
    genes = adata.var_names.astype(str).to_numpy()
    n_genes = int(X.shape[1])
    pc = float(pseudocount)
    samples_used = sorted(pd.unique(sample_vals.astype(str)).tolist())

    all_rows: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []

    for c in top_clusters:
        mask_in = clust.values == str(c)
        mask_out = ~mask_in
        n_in = int(np.sum(mask_in))
        n_out = int(np.sum(mask_out))
        if n_in <= 0 or n_out <= 0:
            summary_rows.append({
                "cluster": str(c),
                "n_cells_total": int(n_in),
                "n_cells_rest": int(n_out),
                "n_samples_used": int(len(samples_used)),
                "cells_per_sample": int(n_in),
                "target_sum_per_cell": int(target_sum),
                "eligible_samples": ",".join(samples_used),
                "status": "skipped_no_cluster_or_rest_cells",
            })
            continue

        X_in = X[mask_in, :]
        X_out = X[mask_out, :]
        mean_in = np.asarray(X_in.mean(axis=0)).ravel().astype(np.float64)
        mean_out = np.asarray(X_out.mean(axis=0)).ravel().astype(np.float64)
        lib_in = float(np.mean(np.asarray(X_in.sum(axis=1)).ravel())) if n_in > 0 else float(target_sum)
        lib_out = float(np.mean(np.asarray(X_out.sum(axis=1)).ravel())) if n_out > 0 else float(target_sum)
        if not np.isfinite(lib_in) or lib_in <= 0:
            lib_in = float(target_sum)
        if not np.isfinite(lib_out) or lib_out <= 0:
            lib_out = float(target_sum)

        cpm_in = (mean_in + pc) / (lib_in + pc * n_genes) * 1e6
        cpm_out = (mean_out + pc) / (lib_out + pc * n_genes) * 1e6
        mean_log2fc = np.log2(cpm_in) - np.log2(cpm_out)
        pct_in = (np.asarray(X_in.getnnz(axis=0)).ravel() / max(n_in, 1)).astype(np.float32)
        pct_out = (np.asarray(X_out.getnnz(axis=0)).ravel() / max(n_out, 1)).astype(np.float32)

        all_rows.append(pd.DataFrame({
            "cluster": str(c),
            "gene": genes,
            "mean_log2fc": mean_log2fc.astype(np.float32),
            "pval": np.ones(n_genes, dtype=np.float64),
            "fdr": np.ones(n_genes, dtype=np.float64),
            "pct_in": pct_in,
            "pct_out": pct_out,
            "mean_log10_cpm_in": np.log10(cpm_in + 1.0).astype(np.float32),
            "mean_log10_cpm_out": np.log10(cpm_out + 1.0).astype(np.float32),
            "n_samples_used": int(len(samples_used)),
            "cells_per_sample": int(n_in),
            "target_sum_per_cell": int(target_sum),
            "marker_map_descriptive_cellwise": True,
            "marker_map_score_basis": "log2fc_x_detection_no_inferential_pvalue",
        }))
        summary_rows.append({
            "cluster": str(c),
            "n_cells_total": int(n_in),
            "n_cells_rest": int(n_out),
            "n_samples_used": int(len(samples_used)),
            "cells_per_sample": int(n_in),
            "target_sum_per_cell": int(target_sum),
            "pseudobulk_libsize_cluster_mean": float("nan"),
            "pseudobulk_libsize_rest_mean": float("nan"),
            "eligible_samples": ",".join(samples_used),
            "status": str(MARKER_MAP_DESCRIPTIVE_STATUS),
            "inferential_pvalues_valid": False,
        })

    de_all = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(columns=["cluster", "gene", "mean_log2fc", "pval", "fdr"])
    de_summary = pd.DataFrame(summary_rows)
    return de_all, de_summary


# =============================================================================
# Leiden marker-expression module map
# =============================================================================

def _numeric_de_columns(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _prepare_marker_map_de_frame(
    de_summary: pd.DataFrame,
    de_all: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """Return finite DE rows for balanced-DE-passed Leiden clusters."""
    if de_summary.shape[0] == 0 or de_all.shape[0] == 0:
        return pd.DataFrame(), []
    if "status" not in de_summary.columns or "cluster" not in de_summary.columns:
        return pd.DataFrame(), []

    status_s = de_summary["status"].astype(str)
    ok_mask = status_s == "ok"
    if bool(MARKER_MAP_SINGLE_SAMPLE_DESCRIPTIVE):
        ok_mask = ok_mask | (status_s == str(MARKER_MAP_DESCRIPTIVE_STATUS))
    ok_clusters = sorted(
        de_summary.loc[ok_mask, "cluster"].astype(str).tolist(),
        key=_natural_sort_key,
    )
    if not ok_clusters:
        return pd.DataFrame(), []

    df = de_all.loc[de_all["cluster"].astype(str).isin(ok_clusters)].copy()
    if df.shape[0] == 0:
        return pd.DataFrame(), []

    df["cluster"] = df["cluster"].astype(str)
    df["gene"] = df["gene"].astype(str)
    df = _numeric_de_columns(
        df,
        [
            "mean_log2fc",
            "pval",
            "fdr",
            "pct_in",
            "pct_out",
            "mean_log10_cpm_in",
            "mean_log10_cpm_out",
        ],
    )
    df = df.loc[
        np.isfinite(df["mean_log2fc"].values)
        & np.isfinite(df["fdr"].values)
        & np.isfinite(df["pct_in"].values)
    ].copy()
    if MARKER_MAP_EXCLUDE_FEATURES:
        exclude = {str(x) for x in MARKER_MAP_EXCLUDE_FEATURES}
        df = df.loc[~df["gene"].astype(str).isin(exclude)].copy()
    if df.shape[0] == 0:
        return pd.DataFrame(), []

    df["neglog10_fdr"] = _safe_neglog10_fdr(df["fdr"].values)
    df["pct_delta"] = df["pct_in"].astype(float).values - df.get("pct_out", 0.0).astype(float).values
    descriptive = np.zeros(df.shape[0], dtype=bool)
    for desc_col in ("marker_map_descriptive_cellwise", "marker_map_descriptive_single_sample"):
        if desc_col in df.columns:
            descriptive |= pd.Series(df[desc_col]).astype(str).str.lower().isin({"1", "true", "yes", "y"}).to_numpy()
    fdr_weight = np.minimum(df["neglog10_fdr"].astype(float).values, 20.0)
    fdr_weight = np.where(descriptive, float(MARKER_MAP_DESCRIPTIVE_FDR_WEIGHT), fdr_weight)
    df["marker_score"] = (
        np.maximum(df["mean_log2fc"].astype(float).values, 0.0)
        * fdr_weight
        * np.sqrt(np.clip(df["pct_in"].astype(float).values, 0.0, 1.0) + 1e-6)
    )
    return df, ok_clusters


def _select_marker_map_genes(df: pd.DataFrame, ok_clusters: Sequence[str]) -> pd.DataFrame:
    """Select a capped, high-information marker set for the expression map."""
    if df.shape[0] == 0:
        return pd.DataFrame()

    strict = df.loc[
        (df["fdr"].astype(float).values <= float(MARKER_MAP_FDR_ALPHA))
        & (df["mean_log2fc"].astype(float).values >= float(MARKER_MAP_MIN_LOG2FC))
        & (df["pct_in"].astype(float).values >= float(MARKER_MAP_MIN_PCT_IN))
        & (df["pct_delta"].astype(float).values >= float(MARKER_MAP_MIN_PCT_DELTA))
    ].copy()

    selected_parts: List[pd.DataFrame] = []
    for cl in sorted([str(c) for c in ok_clusters], key=_natural_sort_key):
        sub = strict.loc[strict["cluster"].astype(str) == str(cl)].copy()
        if sub.shape[0] == 0:
            sub = df.loc[
                (df["cluster"].astype(str).values == str(cl))
                & (df["fdr"].astype(float).values <= float(MARKER_MAP_FALLBACK_FDR))
                & (df["mean_log2fc"].astype(float).values >= float(MARKER_MAP_FALLBACK_MIN_LOG2FC))
                & (df["pct_in"].astype(float).values >= float(MARKER_MAP_FALLBACK_MIN_PCT_IN))
            ].copy()
        if sub.shape[0] == 0:
            sub = df.loc[df["cluster"].astype(str).values == str(cl)].copy()
        if sub.shape[0] == 0:
            continue
        sub = sub.sort_values(
            ["marker_score", "mean_log2fc", "neglog10_fdr", "pct_in", "gene"],
            ascending=[False, False, False, False, True],
        ).head(int(MARKER_MAP_TOP_GENES_PER_LEIDEN))
        selected_parts.append(sub)

    if not selected_parts:
        return pd.DataFrame()

    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.sort_values(
        ["marker_score", "mean_log2fc", "neglog10_fdr", "pct_in", "gene"],
        ascending=[False, False, False, False, True],
    )
    # Keep one primary row per gene: the Leiden cluster where it is most marker-like.
    selected = selected.drop_duplicates("gene", keep="first").copy()
    if int(MARKER_MAP_MAX_GENES) > 0 and selected.shape[0] > int(MARKER_MAP_MAX_GENES):
        selected = selected.head(int(MARKER_MAP_MAX_GENES)).copy()
    selected = selected.reset_index(drop=True)
    selected["selected_marker_rank"] = np.arange(1, selected.shape[0] + 1, dtype=int)
    return selected


def _pivot_gene_by_cluster(
    df: pd.DataFrame,
    genes: Sequence[str],
    clusters: Sequence[str],
    *,
    value_col: str,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    tab = df.loc[df["gene"].astype(str).isin([str(g) for g in genes])].pivot_table(
        index="gene",
        columns="cluster",
        values=value_col,
        aggfunc="mean",
    )
    tab = tab.reindex(index=[str(g) for g in genes], columns=[str(c) for c in clusters])
    return tab.fillna(float(fill_value))


def _row_center_zscore(values: np.ndarray) -> np.ndarray:
    X = np.asarray(values, dtype=float)
    mu = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, keepdims=True)
    Z = np.divide(X - mu, sd, out=np.zeros_like(X, dtype=float), where=sd > 1e-9)
    Z[~np.isfinite(Z)] = 0.0
    return np.clip(Z, -float(MARKER_MAP_Z_CLIP), float(MARKER_MAP_Z_CLIP))


def _cluster_marker_genes_into_modules(
    z_gene_cluster: np.ndarray,
    genes: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ordered row indices and raw module labels for marker genes."""
    n_genes = int(len(genes))
    if n_genes <= 1:
        return np.arange(n_genes, dtype=int), np.ones(n_genes, dtype=int)

    target_modules = int(np.ceil(n_genes / max(float(MARKER_MAP_TARGET_GENES_PER_MODULE), 1.0)))
    target_modules = int(np.clip(target_modules, int(MARKER_MAP_MIN_MODULES), int(MARKER_MAP_MAX_MODULES)))
    target_modules = int(min(target_modules, n_genes))

    D = pdist(np.asarray(z_gene_cluster, dtype=float), metric="correlation")
    if D.size == 0 or (not np.all(np.isfinite(D))) or float(np.nanmax(D)) <= 1e-12:
        D = pdist(np.asarray(z_gene_cluster, dtype=float), metric="euclidean")
    if D.size == 0 or (not np.all(np.isfinite(D))):
        order = np.arange(n_genes, dtype=int)
        return order, np.ones(n_genes, dtype=int)

    Z = linkage(D, method="average")
    order = leaves_list(Z).astype(int)
    raw_modules = fcluster(Z, t=target_modules, criterion="maxclust").astype(int)
    return order, raw_modules


def _format_marker_map_ylabel(module_id: str, dominant_cluster: str, n_genes: int, top_genes: Sequence[str]) -> str:
    genes_txt = ", ".join([str(g) for g in top_genes if str(g)])
    wrapped = textwrap.fill(
        genes_txt,
        width=int(MARKER_MAP_LABEL_WRAP),
        break_long_words=False,
        break_on_hyphens=False,
    ) if genes_txt else ""
    base = f"{module_id} → L{dominant_cluster} (n={int(n_genes)})"
    return base if not wrapped else f"{base}\n{wrapped}"


def create_figure_leiden_expression_cluster_map(
    de_summary: pd.DataFrame,
    de_all: pd.DataFrame,
    *,
    outpath: str,
    leiden_palette: Dict[str, str],
) -> None:
    """Create a compact Leiden-cluster to gene-expression-module map.

    Rows are gene-expression modules learned by clustering the selected marker
    genes' cluster-level expression profiles. Columns are Leiden clusters. Dot
    color is the module's mean gene-centered expression z-score in that Leiden
    cluster; dot size is the mean detection fraction for genes in the module.
    """
    df, ok_clusters = _prepare_marker_map_de_frame(de_summary, de_all)
    if df.shape[0] == 0 or not ok_clusters:
        print("[MarkerMap] No balanced DE rows available; skipping marker-expression map.")
        return

    selected = _select_marker_map_genes(df, ok_clusters)
    if selected.shape[0] == 0:
        print("[MarkerMap] No marker genes selected; skipping marker-expression map.")
        return

    genes = selected["gene"].astype(str).tolist()
    metric_col = "mean_log10_cpm_in" if "mean_log10_cpm_in" in df.columns else "mean_log2fc"
    expr = _pivot_gene_by_cluster(df, genes, ok_clusters, value_col=metric_col, fill_value=0.0)
    pct = _pivot_gene_by_cluster(df, genes, ok_clusters, value_col="pct_in", fill_value=0.0)
    lfc = _pivot_gene_by_cluster(df, genes, ok_clusters, value_col="mean_log2fc", fill_value=0.0)
    z = _row_center_zscore(expr.to_numpy(dtype=float))

    gene_order, raw_modules = _cluster_marker_genes_into_modules(z, genes)
    gene_to_best = selected.set_index("gene", drop=False)

    module_rows: List[Dict[str, object]] = []
    module_long_rows: List[Dict[str, object]] = []
    gene_rows: List[pd.DataFrame] = []

    # Summarize raw modules, then sort them by their dominant Leiden cluster.
    raw_module_summaries = []
    for raw_m in sorted(np.unique(raw_modules).astype(int).tolist()):
        idx = np.flatnonzero(raw_modules == int(raw_m)).astype(int)
        if idx.size == 0:
            continue
        mean_z = np.nanmean(z[idx, :], axis=0)
        mean_pct = np.nanmean(pct.to_numpy(dtype=float)[idx, :], axis=0)
        mean_lfc = np.nanmean(lfc.to_numpy(dtype=float)[idx, :], axis=0)
        dom_i = int(np.nanargmax(mean_z)) if mean_z.size else 0
        members = [genes[i] for i in idx]
        score_series = gene_to_best.reindex(members)["marker_score"].fillna(0.0)
        top_members = score_series.sort_values(ascending=False).index.astype(str).tolist()
        raw_module_summaries.append({
            "raw_module": int(raw_m),
            "idx": idx,
            "members": members,
            "top_members": top_members,
            "mean_z": mean_z,
            "mean_pct": mean_pct,
            "mean_lfc": mean_lfc,
            "dominant_cluster": str(ok_clusters[dom_i]),
            "dominant_cluster_index": int(dom_i),
            "dominant_z": float(mean_z[dom_i]) if mean_z.size else float("nan"),
        })

    raw_module_summaries = sorted(
        raw_module_summaries,
        key=lambda r: (_natural_sort_key(str(r["dominant_cluster"])), -float(r["dominant_z"]), int(r["raw_module"])),
    )
    if not raw_module_summaries:
        print("[MarkerMap] Marker genes could not be assigned to modules; skipping.")
        return

    module_z = []
    module_pct = []
    ylabels = []
    raw_to_display: Dict[int, str] = {}
    for display_i, rec in enumerate(raw_module_summaries, start=1):
        module_id = f"G{display_i}"
        raw_to_display[int(rec["raw_module"])] = module_id
        top_label_genes = [str(g) for g in rec["top_members"][: int(MARKER_MAP_TOP_GENES_IN_MODULE_LABEL)]]
        ylabels.append(_format_marker_map_ylabel(module_id, str(rec["dominant_cluster"]), len(rec["members"]), top_label_genes))
        module_z.append(np.asarray(rec["mean_z"], dtype=float))
        module_pct.append(np.asarray(rec["mean_pct"], dtype=float))
        module_rows.append({
            "gene_expression_module": module_id,
            "raw_module": int(rec["raw_module"]),
            "dominant_leiden_cluster": str(rec["dominant_cluster"]),
            "dominant_mean_z": float(rec["dominant_z"]),
            "n_genes": int(len(rec["members"])),
            "top_genes": ",".join(top_label_genes),
            "all_genes": ",".join([str(g) for g in rec["members"]]),
        })
        for j, cl in enumerate(ok_clusters):
            module_long_rows.append({
                "gene_expression_module": module_id,
                "leiden_cluster": str(cl),
                "mean_expression_z": float(rec["mean_z"][j]),
                "mean_detection_fraction": float(rec["mean_pct"][j]),
                "mean_log2fc_vs_rest": float(rec["mean_lfc"][j]),
                "n_genes": int(len(rec["members"])),
            })
        gene_sub = selected.loc[selected["gene"].astype(str).isin([str(g) for g in rec["members"]])].copy()
        gene_sub["gene_expression_module"] = module_id
        gene_sub["module_dominant_leiden_cluster"] = str(rec["dominant_cluster"])
        gene_rows.append(gene_sub)

    module_z_arr = np.vstack(module_z).astype(float)
    module_pct_arr = np.vstack(module_pct).astype(float)
    n_modules, n_clusters = module_z_arr.shape

    # Save tables before plotting so failed rendering still leaves useful output.
    os.makedirs(TABLE_DIR, exist_ok=True)
    selected_out = pd.concat(gene_rows, ignore_index=True) if gene_rows else selected.copy()
    selected_out.to_csv(os.path.join(TABLE_DIR, str(MARKER_MAP_TABLE_SELECTED_GENES)), sep="\t", index=False)
    pd.DataFrame(module_rows).to_csv(os.path.join(TABLE_DIR, str(MARKER_MAP_TABLE_MODULES)), sep="\t", index=False)
    pd.DataFrame(module_long_rows).to_csv(os.path.join(TABLE_DIR, str(MARKER_MAP_TABLE_MODULE_BY_LEIDEN)), sep="\t", index=False)

    fig_w = min(
        float(MARKER_MAP_MAX_FIG_WIDTH),
        max(float(MARKER_MAP_MIN_FIG_WIDTH), 4.8 + float(MARKER_MAP_CLUSTER_WIDTH) * float(n_clusters)),
    )
    fig_h = max(5.4, float(MARKER_MAP_BASE_HEIGHT) + float(MARKER_MAP_ROW_HEIGHT) * float(n_modules))
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.18, 1.0], hspace=0.03)
    ax_strip = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])

    x = np.tile(np.arange(n_clusters), n_modules)
    y = np.repeat(np.arange(n_modules), n_clusters)
    colors = module_z_arr.reshape(-1)
    sizes = float(MARKER_MAP_DOT_MIN_SIZE) + (
        float(MARKER_MAP_DOT_MAX_SIZE) - float(MARKER_MAP_DOT_MIN_SIZE)
    ) * np.clip(module_pct_arr.reshape(-1), 0.0, 1.0)

    sca = ax.scatter(
        x,
        y,
        c=colors,
        s=sizes,
        cmap="RdBu_r",
        vmin=-float(MARKER_MAP_Z_CLIP),
        vmax=float(MARKER_MAP_Z_CLIP),
        edgecolors="white",
        linewidths=0.55,
        zorder=3,
    )

    # Outline each module's dominant Leiden cluster to make the mapping explicit.
    for i in range(n_modules):
        j = int(np.nanargmax(module_z_arr[i, :]))
        ax.scatter(
            [j],
            [i],
            s=[float(MARKER_MAP_DOT_MAX_SIZE) * 1.08],
            facecolors="none",
            edgecolors="black",
            linewidths=1.0,
            zorder=4,
        )

    ax.set_xlim(-0.55, n_clusters - 0.45)
    ax.set_ylim(n_modules - 0.45, -0.55)
    ax.set_xticks(np.arange(n_clusters))
    cluster_sizes = de_summary.set_index(de_summary["cluster"].astype(str))["n_cells_total"].to_dict() if "n_cells_total" in de_summary.columns else {}
    if n_clusters <= 16 and cluster_sizes:
        xlabels = [f"L{c}\n{int(cluster_sizes.get(str(c), 0)):,} cells" for c in ok_clusters]
    else:
        xlabels = [f"L{c}" for c in ok_clusters]
    ax.set_xticklabels(
        xlabels,
        rotation=45 if n_clusters > 10 else 0,
        ha="right" if n_clusters > 10 else "center",
        fontsize=float(MARKER_MAP_XTICK_FONTSIZE),
    )
    ax.set_yticks(np.arange(n_modules))
    ax.set_yticklabels(ylabels, fontsize=float(MARKER_MAP_YTICK_FONTSIZE))
    ax.set_xlabel("Leiden cluster")
    ax.set_ylabel("Gene-expression module")
    ax.set_title("Leiden clusters mapped to marker-gene expression modules", pad=8)
    ax.set_axisbelow(True)
    ax.grid(True, axis="both", color="#e5e5e5", linewidth=0.7)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(sca, ax=ax, orientation="horizontal", fraction=0.06, pad=0.12, aspect=35)
    cbar.set_label("Module mean gene-centered expression z-score")
    cbar.ax.tick_params(labelsize=8)

    # Dot-size legend for detection fraction.
    legend_fracs = [0.05, 0.25, 0.50]
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="#bdbdbd",
            markeredgecolor="white",
            markeredgewidth=0.55,
            markersize=np.sqrt(
                float(MARKER_MAP_DOT_MIN_SIZE)
                + (float(MARKER_MAP_DOT_MAX_SIZE) - float(MARKER_MAP_DOT_MIN_SIZE)) * f
            ),
            label=f"{int(round(100 * f))}%",
        )
        for f in legend_fracs
    ]
    ax.legend(
        handles=handles,
        title="Mean detection",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.33),
        ncol=len(handles),
        frameon=False,
        borderaxespad=0,
        fontsize=8,
        title_fontsize=9,
        handletextpad=0.6,
        columnspacing=1.2,
    )

    # Leiden palette strip above the matrix.
    ax_strip.set_xlim(-0.55, n_clusters - 0.45)
    ax_strip.set_ylim(0, 1)
    for j, cl in enumerate(ok_clusters):
        ax_strip.add_patch(
            mpatches.Rectangle(
                (j - 0.48, 0.22),
                0.96,
                0.44,
                facecolor=leiden_palette.get(str(cl), "#888888"),
                edgecolor="none",
            )
        )
        ax_strip.text(j, 0.78, f"L{cl}", ha="center", va="bottom", fontsize=8)
    ax_strip.axis("off")

    fig.text(
        0.01,
        0.01,
        "Rows are clustered marker genes selected from balanced Leiden-vs-rest DE; outlined dots mark each module's dominant Leiden cluster.",
        ha="left",
        va="bottom",
        fontsize=7,
        color="#444444",
    )
    fig.subplots_adjust(left=0.34, right=0.98, top=0.94, bottom=0.20)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, format="pdf", dpi=300, bbox_inches="tight")
    png_path = os.path.splitext(outpath)[0] + ".png"
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")
    print(f"Saved: {png_path}")
    print(
        f"[MarkerMap] Wrote tables: {MARKER_MAP_TABLE_SELECTED_GENES}, "
        f"{MARKER_MAP_TABLE_MODULES}, {MARKER_MAP_TABLE_MODULE_BY_LEIDEN}"
    )

# =============================================================================
# =============================================================================

def _safe_neglog10_fdr(fdr: np.ndarray) -> np.ndarray:
    vals = np.asarray(fdr, dtype=float)
    vals = np.where(np.isfinite(vals), vals, 1.0)
    vals = np.clip(vals, 1e-300, 1.0)
    return -np.log10(vals)




def _pca_axis_label(adata: ad.AnnData, pc_index_zero_based: int) -> str:
    label = f"PC{pc_index_zero_based + 1}"
    try:
        vr = np.asarray(adata.uns.get("pca", {}).get("variance_ratio", []), dtype=float)
        if vr.size > pc_index_zero_based and np.isfinite(vr[pc_index_zero_based]):
            return f"{label} ({100.0 * float(vr[pc_index_zero_based]):.1f}% variance)"
    except Exception:
        pass
    return label


def create_figure_leiden_pca(
    adata: ad.AnnData,
    *,
    outpath: str,
    leiden_palette: Dict[str, str],
    leiden_key: str = LEIDEN_KEY,
) -> None:
    """Create a high-quality PC1/PC2 scatter colored by Leiden cluster.

    This plots the exact `adata.obsm["X_pca"]` coordinates that were used to
    build the PCA-neighbor graph for Leiden clustering.
    """
    if "X_pca" not in adata.obsm:
        raise RuntimeError("PCA figure requested, but adata.obsm['X_pca'] is missing.")
    X = np.asarray(adata.obsm["X_pca"])
    if X.ndim != 2 or X.shape[1] < 2 or X.shape[0] == 0:
        raise RuntimeError(f"PCA figure requested, but X_pca has invalid shape: {X.shape}.")
    if leiden_key not in adata.obs.columns:
        raise RuntimeError(f"PCA figure requested, but obs['{leiden_key}'] is missing.")

    plot_df = pd.DataFrame({
        "pc1": X[:, 0].astype(float, copy=False),
        "pc2": X[:, 1].astype(float, copy=False),
        "cluster": adata.obs[leiden_key].astype(str).values,
    })
    plot_df = plot_df.loc[np.isfinite(plot_df["pc1"].values) & np.isfinite(plot_df["pc2"].values)].copy()
    if plot_df.shape[0] == 0:
        raise RuntimeError("PCA figure requested, but there are no finite PC1/PC2 coordinates.")

    fig, ax = plt.subplots(figsize=(9.5, 8.4))
    ordered_clusters = sorted(plot_df["cluster"].unique().tolist(), key=_natural_sort_key)

    # Plot larger clusters first so smaller clusters remain visible on top.
    cluster_sizes = plot_df["cluster"].value_counts().to_dict()
    draw_order = sorted(ordered_clusters, key=lambda c: int(cluster_sizes.get(c, 0)), reverse=True)
    for cl in draw_order:
        sub = plot_df.loc[plot_df["cluster"].astype(str) == str(cl)]
        ax.scatter(
            sub["pc1"].to_numpy(dtype=float),
            sub["pc2"].to_numpy(dtype=float),
            s=float(PCA_POINT_SIZE),
            c=leiden_palette.get(str(cl), "#888888"),
            alpha=float(PCA_ALPHA),
            linewidths=0,
            rasterized=True,
            zorder=2,
        )

    if PCA_SHOW_CENTROID_LABELS:
        centers = plot_df.groupby("cluster", sort=False)[["pc1", "pc2"]].median().reset_index()
        for row in centers.itertuples(index=False):
            cl = str(row.cluster)
            color = leiden_palette.get(cl, "#333333")
            txt = ax.text(
                float(row.pc1),
                float(row.pc2),
                f"L{cl}",
                ha="center",
                va="center",
                fontsize=float(PCA_LABEL_FONTSIZE),
                fontweight="bold",
                color=color,
                zorder=5,
            )
            txt.set_path_effects([pe.withStroke(linewidth=3.5, foreground="white")])

    ax.set_xlabel(_pca_axis_label(adata, 0), fontsize=16)
    ax.set_ylabel(_pca_axis_label(adata, 1), fontsize=16)
    ax.tick_params(axis="both", labelsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.grid(False)
    ax.margins(0.05)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=7.5,
               markerfacecolor=leiden_palette.get(str(cl), "#888888"), markeredgecolor="none", label=f"L{cl}")
        for cl in ordered_clusters
    ]
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            title="Leiden",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=min(8, len(legend_handles)),
            fontsize=11,
            title_fontsize=12,
            columnspacing=0.9,
            handletextpad=0.35,
            frameon=False,
        )

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, format="pdf", dpi=300, bbox_inches="tight")
    # Write a PNG companion so runs leave a visible preview in directory listings.
    png_path = os.path.splitext(outpath)[0] + ".png"
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")
    print(f"Saved: {png_path}")


def _save_figure_with_png(fig: mpl.figure.Figure, outpath: str) -> None:
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, format="pdf", dpi=300, bbox_inches="tight")
    png_path = os.path.splitext(outpath)[0] + ".png"
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")
    print(f"Saved: {png_path}")


def _plot_umap_categorical(
    plot_df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    category_col: str,
    outpath: str,
    palette: Dict[str, str],
    title: str,
    legend_title: str,
    order: Optional[Sequence[str]] = None,
    show_labels: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 8.4))
    categories = [str(x) for x in pd.unique(plot_df[category_col].astype(str))]
    ordered = [str(x) for x in order if str(x) in categories] if order is not None else []
    ordered += [c for c in categories if c not in ordered]
    sizes = plot_df[category_col].astype(str).value_counts().to_dict()
    draw_order = sorted(ordered, key=lambda c: int(sizes.get(c, 0)), reverse=True)
    for cat in draw_order:
        sub = plot_df.loc[plot_df[category_col].astype(str) == str(cat)]
        ax.scatter(
            sub[x_col].to_numpy(dtype=float),
            sub[y_col].to_numpy(dtype=float),
            s=float(UMAP_POINT_SIZE),
            c=palette.get(str(cat), "#888888"),
            alpha=float(UMAP_ALPHA),
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
    if show_labels:
        centers = plot_df.groupby(category_col, sort=False)[[x_col, y_col]].median().reset_index()
        for row in centers.itertuples(index=False):
            cat = str(getattr(row, category_col))
            txt = ax.text(
                float(getattr(row, x_col)),
                float(getattr(row, y_col)),
                cat,
                ha="center",
                va="center",
                fontsize=float(UMAP_LABEL_FONTSIZE),
                fontweight="bold",
                color=palette.get(cat, "#333333"),
                zorder=5,
            )
            txt.set_path_effects([pe.withStroke(linewidth=3.5, foreground="white")])
    ax.set_xlabel("UMAP1", fontsize=16)
    ax.set_ylabel("UMAP2", fontsize=16)
    ax.set_title(title, fontsize=16, pad=10)
    ax.tick_params(axis="both", labelsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.grid(False)
    ax.margins(0.05)
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=7.5,
               markerfacecolor=palette.get(str(cat), "#888888"), markeredgecolor="none", label=str(cat))
        for cat in ordered
    ]
    if handles:
        ax.legend(
            handles=handles,
            title=legend_title,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=min(8, len(handles)),
            fontsize=11,
            title_fontsize=12,
            columnspacing=0.9,
            handletextpad=0.35,
            frameon=False,
        )
    _save_figure_with_png(fig, outpath)


def _plot_umap_continuous(
    plot_df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    value_col: str,
    outpath: str,
    title: str,
    colorbar_label: str,
    cmap: str = UMAP_CONTINUOUS_CMAP,
) -> None:
    vals = pd.to_numeric(plot_df[value_col], errors="coerce")
    mask = np.isfinite(plot_df[x_col].to_numpy(dtype=float)) & np.isfinite(plot_df[y_col].to_numpy(dtype=float)) & np.isfinite(vals.to_numpy(dtype=float))
    plot_df = plot_df.loc[mask].copy()
    vals = pd.to_numeric(plot_df[value_col], errors="coerce").to_numpy(dtype=float)
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(9.5, 8.4))
    sca = ax.scatter(
        plot_df.iloc[order][x_col].to_numpy(dtype=float),
        plot_df.iloc[order][y_col].to_numpy(dtype=float),
        c=vals[order],
        s=float(UMAP_POINT_SIZE),
        cmap=cmap,
        alpha=float(UMAP_ALPHA),
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    ax.set_xlabel("UMAP1", fontsize=16)
    ax.set_ylabel("UMAP2", fontsize=16)
    ax.set_title(title, fontsize=16, pad=10)
    ax.tick_params(axis="both", labelsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.grid(False)
    ax.margins(0.05)
    cbar = fig.colorbar(sca, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label(colorbar_label, fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    _save_figure_with_png(fig, outpath)


def create_figure_leiden_umap(
    adata: ad.AnnData,
    *,
    outpath: str,
    leiden_palette: Dict[str, str],
    leiden_key: str = LEIDEN_KEY,
) -> None:
    """Create a high-quality UMAP scatter colored by Leiden cluster."""
    if "X_umap" not in adata.obsm:
        raise RuntimeError("UMAP figure requested, but adata.obsm['X_umap'] is missing.")
    X = np.asarray(adata.obsm["X_umap"])
    if X.ndim != 2 or X.shape[1] < 2 or X.shape[0] == 0:
        raise RuntimeError(f"UMAP figure requested, but X_umap has invalid shape: {X.shape}.")
    if leiden_key not in adata.obs.columns:
        raise RuntimeError(f"UMAP figure requested, but obs['{leiden_key}'] is missing.")

    plot_df = pd.DataFrame({
        "umap1": X[:, 0].astype(float, copy=False),
        "umap2": X[:, 1].astype(float, copy=False),
        "cluster": adata.obs[leiden_key].astype(str).values,
    })
    plot_df = plot_df.loc[np.isfinite(plot_df["umap1"].values) & np.isfinite(plot_df["umap2"].values)].copy()
    if plot_df.shape[0] == 0:
        raise RuntimeError("UMAP figure requested, but there are no finite UMAP1/UMAP2 coordinates.")

    fig, ax = plt.subplots(figsize=(9.5, 8.4))
    ordered_clusters = sorted(plot_df["cluster"].unique().tolist(), key=_natural_sort_key)
    cluster_sizes = plot_df["cluster"].value_counts().to_dict()
    draw_order = sorted(ordered_clusters, key=lambda c: int(cluster_sizes.get(c, 0)), reverse=True)
    for cl in draw_order:
        sub = plot_df.loc[plot_df["cluster"].astype(str) == str(cl)]
        ax.scatter(
            sub["umap1"].to_numpy(dtype=float),
            sub["umap2"].to_numpy(dtype=float),
            s=float(UMAP_POINT_SIZE),
            c=leiden_palette.get(str(cl), "#888888"),
            alpha=float(UMAP_ALPHA),
            linewidths=0,
            rasterized=True,
            zorder=2,
        )

    if UMAP_SHOW_CENTROID_LABELS:
        centers = plot_df.groupby("cluster", sort=False)[["umap1", "umap2"]].median().reset_index()
        for row in centers.itertuples(index=False):
            cl = str(row.cluster)
            txt = ax.text(
                float(row.umap1),
                float(row.umap2),
                f"L{cl}",
                ha="center",
                va="center",
                fontsize=float(UMAP_LABEL_FONTSIZE),
                fontweight="bold",
                color=leiden_palette.get(cl, "#333333"),
                zorder=5,
            )
            txt.set_path_effects([pe.withStroke(linewidth=3.5, foreground="white")])

    ax.set_xlabel("UMAP1", fontsize=16)
    ax.set_ylabel("UMAP2", fontsize=16)
    ax.tick_params(axis="both", labelsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.grid(False)
    ax.margins(0.05)

    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=7.5,
               markerfacecolor=leiden_palette.get(str(cl), "#888888"), markeredgecolor="none", label=f"L{cl}")
        for cl in ordered_clusters
    ]
    if handles:
        ax.legend(
            handles=handles,
            title="Leiden",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=min(8, len(handles)),
            fontsize=11,
            title_fontsize=12,
            columnspacing=0.9,
            handletextpad=0.35,
            frameon=False,
        )

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, format="pdf", dpi=300, bbox_inches="tight")
    png_path = os.path.splitext(outpath)[0] + ".png"
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")
    print(f"Saved: {png_path}")




def create_figure_umap_by_age(adata: ad.AnnData, *, outpath: str, age_col: str = "age_hpf") -> None:
    if "X_umap" not in adata.obsm:
        raise RuntimeError("UMAP-by-age requested, but adata.obsm['X_umap'] is missing.")
    if age_col not in adata.obs.columns:
        raise KeyError(f"Missing obs['{age_col}'] required for UMAP-by-age.")
    X = np.asarray(adata.obsm["X_umap"])
    plot_df = pd.DataFrame({
        "umap1": X[:, 0].astype(float, copy=False),
        "umap2": X[:, 1].astype(float, copy=False),
        age_col: adata.obs[age_col].astype(str).values,
    })
    plot_df = plot_df.loc[np.isfinite(plot_df["umap1"].values) & np.isfinite(plot_df["umap2"].values)].copy()
    _plot_umap_categorical(
        plot_df,
        x_col="umap1",
        y_col="umap2",
        category_col=age_col,
        outpath=outpath,
        palette=AGE_PALETTE,
        title="UMAP colored by developmental age",
        legend_title="Age",
        order=AGE_ORDER,
        show_labels=True,
    )


def create_figure_umap_by_subconsensus_count(adata: ad.AnnData, *, outpath: str, value_col: str = "n_distinct_subconsensuses") -> None:
    if "X_umap" not in adata.obsm:
        raise RuntimeError("UMAP-by-subconsensus-count requested, but adata.obsm['X_umap'] is missing.")
    if value_col not in adata.obs.columns:
        if "n_hubs_with_call" in adata.obs.columns:
            value_col = "n_hubs_with_call"
        else:
            raise KeyError(f"Missing obs['{value_col}'] required for UMAP-by-subconsensus-count.")
    X = np.asarray(adata.obsm["X_umap"])
    plot_df = pd.DataFrame({
        "umap1": X[:, 0].astype(float, copy=False),
        "umap2": X[:, 1].astype(float, copy=False),
        value_col: pd.to_numeric(adata.obs[value_col], errors="coerce").values,
    })
    plot_df = plot_df.loc[np.isfinite(plot_df["umap1"].values) & np.isfinite(plot_df["umap2"].values)].copy()
    _plot_umap_continuous(
        plot_df,
        x_col="umap1",
        y_col="umap2",
        value_col=value_col,
        outpath=outpath,
        title="UMAP colored by distinct sub-consensuses used per cell",
        colorbar_label="Distinct sub-consensuses counted per cell",
        cmap=UMAP_CONTINUOUS_CMAP,
    )










# =============================================================================
# PyVista helpers for Leiden-colored GSE clouds
# =============================================================================

def _terminate_xvfb_proc() -> None:
    global _XVFB_PROC
    proc = _XVFB_PROC
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
    except Exception:
        pass
    _XVFB_PROC = None


def _pv_start_virtual_framebuffer_if_needed() -> None:
    """Prepare off-screen PyVista without blocking in pv.start_xvfb()."""
    if pv is None:
        raise ImportError("pyvista is required for GSE_1/GSE_2/GSE_3 rendering.")
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    if os.environ.get("DISPLAY", ""):
        return
    if os.environ.get("PYVISTA_SKIP_XVFB", "").strip().lower() in {"1", "true", "yes", "on"}:
        _log("[PyVista] DISPLAY is unset and PYVISTA_SKIP_XVFB is set; continuing without Xvfb.")
        return
    xvfb = shutil.which("Xvfb")
    if xvfb is None:
        _log("[PyVista] DISPLAY is unset and Xvfb was not found; continuing without Xvfb.")
        return

    global _XVFB_PROC
    if _XVFB_PROC is not None and _XVFB_PROC.poll() is None:
        os.environ.setdefault("DISPLAY", os.environ.get("PYVISTA_XVFB_DISPLAY", ":99"))
        return

    base_display = os.environ.get("PYVISTA_XVFB_DISPLAY")
    display_nums = []
    if base_display:
        try:
            display_nums.append(int(str(base_display).lstrip(":")))
        except Exception:
            pass
    start = 100 + (os.getpid() % 4000)
    display_nums.extend(range(start, start + 8))
    display_nums.extend(range(99, 103))

    for display_num in display_nums:
        display = f":{int(display_num)}"
        cmd = [xvfb, display, "-screen", "0", "1600x1200x24", "-nolisten", "tcp"]
        _log(f"[PyVista] Starting Xvfb on DISPLAY={display} for off-screen rendering.")
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.35)
            if proc.poll() is None:
                _XVFB_PROC = proc
                os.environ["DISPLAY"] = display
                atexit.register(_terminate_xvfb_proc)
                return
            _log(f"[PyVista] Xvfb DISPLAY={display} exited immediately; trying another display.")
        except Exception as e:  # pragma: no cover
            _log(f"[PyVista] Warning: could not start Xvfb on DISPLAY={display}: {e}")

    _log("[PyVista] Warning: could not start Xvfb; continuing without DISPLAY.")


def _decode_bytes_to_str_array(arr: np.ndarray) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    if arr.dtype.kind == "S":
        return np.char.decode(arr, "utf-8")
    if arr.dtype.kind == "O":
        out = []
        for x in arr:
            if isinstance(x, (bytes, bytearray)):
                out.append(x.decode("utf-8"))
            else:
                out.append(str(x))
        return np.asarray(out, dtype="U")
    return arr.astype("U") if arr.dtype.kind != "U" else arr


def _read_obs_column_h5ad(obs_group: "h5py.Group", col: str) -> np.ndarray:
    """Read one obs column from an h5ad via h5py, including categorical encodings."""
    if col not in obs_group:
        raise KeyError(f"obs['{col}'] not found")
    node = obs_group[col]
    if isinstance(node, h5py.Dataset):
        arr = node[()]
        if isinstance(arr, np.ndarray) and arr.dtype.kind in {"S", "O"}:
            return _decode_bytes_to_str_array(arr)
        return np.asarray(arr)
    if isinstance(node, h5py.Group):
        if "codes" not in node or "categories" not in node:
            raise ValueError(f"Unsupported obs column encoding for '{col}'")
        codes = np.asarray(node["codes"][()], dtype=np.int64)
        cats_raw = np.asarray(node["categories"][()])
        if cats_raw.dtype.kind in {"S", "O", "U"}:
            cats_str = _decode_bytes_to_str_array(cats_raw)
            cats_num = pd.to_numeric(pd.Series(cats_str), errors="coerce")
            if float(cats_num.notna().mean()) >= 0.99:
                cats = cats_num.astype(np.int64).to_numpy()
            else:
                cats = np.asarray(cats_str)
        else:
            cats = np.asarray(cats_raw)
        out = np.empty(codes.shape[0], dtype=cats.dtype)
        valid = codes >= 0
        out[valid] = cats[codes[valid]]
        if (~valid).any():
            if np.issubdtype(out.dtype, np.number):
                out[~valid] = -1
            else:
                out[~valid] = ""
        return out
    raise TypeError(f"Unexpected HDF5 node type for obs['{col}']")


def _read_gse_coords_and_cluster_from_h5ad(
    filepath: str,
    *,
    coord_keys: Tuple[str, str, str] = GSE_COORD_KEYS,
    cluster_key: str = CLUSTER_KEY,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read GSE_1/GSE_2/GSE_3 and hub cluster IDs without loading X."""
    with h5py.File(filepath, "r") as f:
        if "obs" not in f:
            raise ValueError("Invalid h5ad: missing 'obs' group")
        obs = f["obs"]
        first = _read_obs_column_h5ad(obs, coord_keys[0])
        n = int(first.shape[0])
        coords = np.empty((n, 3), dtype=np.float32)
        coords[:, 0] = np.asarray(first, dtype=np.float32)
        for i, key in enumerate(coord_keys[1:], start=1):
            a = _read_obs_column_h5ad(obs, key)
            if a.shape[0] != n:
                raise ValueError(f"Length mismatch for {key} in {filepath}")
            coords[:, i] = np.asarray(a, dtype=np.float32)
        clusters = np.array(_read_obs_column_h5ad(obs, cluster_key), copy=True)
        if clusters.shape[0] != n:
            raise ValueError(f"Length mismatch for cluster_key='{cluster_key}' in {filepath}")
    return coords, clusters


def _cluster_value_to_token(x) -> str:
    """Normalize raw/refined cluster IDs to stable string tokens for matching."""
    if isinstance(x, (bytes, bytearray)):
        try:
            x = x.decode("utf-8")
        except Exception:
            x = str(x)
    try:
        if x is None or pd.isna(x):
            return ""
    except Exception:
        if x is None:
            return ""
    if isinstance(x, np.generic):
        try:
            x = x.item()
        except Exception:
            pass
    if isinstance(x, str):
        s = x.strip()
        if s.lower() in {"", "nan", "none", "<na>", "na", "n/a"}:
            return ""
        try:
            f = float(s)
            if np.isfinite(f) and abs(f - round(f)) < 1e-6:
                return str(int(round(f)))
            if np.isfinite(f):
                return format(float(f), ".12g")
        except Exception:
            pass
        return s
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    try:
        f = float(x)
        if np.isfinite(f):
            if abs(f - round(f)) < 1e-6:
                return str(int(round(f)))
            return format(float(f), ".12g")
    except Exception:
        pass
    return str(x).strip()


def _writable_object_array(values: object) -> np.ndarray:
    """Return a writable object ndarray even when pandas/h5py exposes a read-only view."""
    out = np.array(values, dtype=object, copy=True).reshape(-1)
    if not out.flags.writeable:
        out = out.copy()
    return out


_BAD_CLUSTER_TOKEN_STRINGS = np.asarray(["", "nan", "none", "<na>", "na", "n/a"], dtype="U5")


def _normalize_string_token_array(arr_u: np.ndarray) -> np.ndarray:
    """Normalize an already-string ndarray without pandas object-dtype round-trips."""
    arr_u = np.asarray(arr_u)
    if arr_u.size == 0:
        return np.asarray(arr_u, dtype="U")
    s = np.char.strip(arr_u.astype("U", copy=False))
    lower = np.char.lower(s)
    bad = np.isin(lower, _BAD_CLUSTER_TOKEN_STRINGS)
    if np.any(bad):
        s = s.copy()
        s[bad] = ""

    # Avoid parsing every token as a float.  Most cluster IDs are integer dtypes
    # upstream; for string arrays only strings with an explicit decimal or exponent
    # can need normalization like "3.0" -> "3" or "1e3" -> "1000".
    maybe_float = (~bad) & (
        (np.char.find(s, ".") >= 0)
        | (np.char.find(lower, "e") >= 0)
    )
    if np.any(maybe_float):
        idx = np.flatnonzero(maybe_float)
        vals = pd.to_numeric(pd.Series(s[idx], copy=False), errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        finite = np.isfinite(vals)
        if np.any(finite):
            s_obj = s.astype(object, copy=True)
            idx_f = idx[finite]
            vals_f = vals[finite]
            int_like = np.abs(vals_f - np.rint(vals_f)) < 1e-6
            if np.any(int_like):
                s_obj[idx_f[int_like]] = np.rint(vals_f[int_like]).astype(np.int64).astype(str)
            if np.any(~int_like):
                s_obj[idx_f[~int_like]] = np.asarray([format(float(v), ".12g") for v in vals_f[~int_like]], dtype=object)
            return np.asarray(s_obj)
    return s


def _cluster_array_to_tokens(values: np.ndarray) -> np.ndarray:
    """Normalize cluster IDs to comparable tokens with one cheap path per dtype.

    The PyVista route calls this on arrays with up to ~1e7 hubs.  Keep the hot
    integer and string paths out of pandas/object dtype: do not materialize a
    writable object copy, do not run Series.map(lambda ...), and do not parse
    every token as a float.
    """
    arr = np.asarray(values).reshape(-1)
    if arr.size == 0:
        return np.array([], dtype="U1")

    kind = arr.dtype.kind
    if kind in {"i", "u"}:
        return arr.astype("U", copy=False)

    if kind == "b":
        return np.where(arr, "True", "False").astype("U5", copy=False)

    if kind == "f":
        out = np.empty(arr.shape[0], dtype=object)
        finite = np.isfinite(arr)
        out[~finite] = ""
        if np.any(finite):
            vals = arr[finite].astype(np.float64, copy=False)
            finite_idx = np.flatnonzero(finite)
            int_like = np.abs(vals - np.rint(vals)) < 1e-6
            if np.any(int_like):
                out[finite_idx[int_like]] = np.rint(vals[int_like]).astype(np.int64).astype(str)
            if np.any(~int_like):
                out[finite_idx[~int_like]] = np.asarray([format(float(v), ".12g") for v in vals[~int_like]], dtype=object)
        return out

    if kind == "S":
        return _normalize_string_token_array(np.char.decode(arr, "utf-8", errors="ignore"))

    if kind == "U":
        return _normalize_string_token_array(arr)

    # Object arrays are common after the refined-hub sidecar.  They are usually
    # already Python strings; converting once to Unicode and using np.char avoids
    # the much slower pandas map/to_numeric/object-copy pipeline.  Only object
    # arrays with bytes get the scalar fallback.
    if kind == "O":
        sample = arr[: min(arr.size, 4096)]
        if any(isinstance(x, (bytes, bytearray)) for x in sample):
            return np.asarray([_cluster_value_to_token(v) for v in arr], dtype=object)
        try:
            return _normalize_string_token_array(arr.astype("U", copy=False))
        except Exception:
            return np.asarray([_cluster_value_to_token(v) for v in arr], dtype=object)

    return np.asarray([_cluster_value_to_token(v) for v in arr], dtype=object)


def _normalized_cluster_label_lookup(cluster_to_label: Dict) -> Dict[str, object]:
    lookup: Dict[str, object] = {}
    for key, lab in dict(cluster_to_label).items():
        token = _cluster_value_to_token(key)
        if token == "":
            continue
        lookup.setdefault(token, lab)
    return lookup


def _load_refined_hub_labels_for_file(filepath: str, n_expected: int) -> Optional[np.ndarray]:
    """Load the component-split hub-label sidecar written during aggregation."""
    sidecar_path = os.path.join(os.path.dirname(filepath), str(REFINED_HUB_LABEL_SIDECAR))
    if not bool(WRITE_REFINED_HUB_LABEL_SIDECAR) or not os.path.isfile(sidecar_path):
        return None
    try:
        # This sidecar can have ~1e7 rows. Read only the one column needed by
        # the PyVista label path and keep it as NumPy unicode instead of object.
        df = pd.read_csv(
            sidecar_path,
            sep="\t",
            usecols=["refined_cell_label"],
            dtype={"refined_cell_label": "string"},
        )
        vals = df["refined_cell_label"].astype("string").fillna("").astype(str).to_numpy(dtype="U")
        if int(vals.shape[0]) != int(n_expected):
            print(
                f"[ComponentSplit] Refined hub-label sidecar length mismatch for {filepath}: "
                f"{vals.shape[0]:,} vs expected {int(n_expected):,}; ignoring sidecar."
            )
            return None
        return np.array(vals, copy=True)
    except Exception as e:
        print(f"[ComponentSplit] Failed reading refined hub-label sidecar for {filepath} ({e}); ignoring sidecar.")
        return None


def _replace_clusters_with_refined_sidecar(
    filepath: str,
    clusters: np.ndarray,
    *,
    log_prefix: str,
    return_tokens: bool = False,
):
    refined = _load_refined_hub_labels_for_file(filepath, int(np.asarray(clusters).shape[0]))
    if refined is None:
        out = np.array(clusters, copy=True)
        return (out, False) if bool(return_tokens) else out
    refined_s = np.asarray(refined).astype("U", copy=False)
    n_labeled = int(np.count_nonzero(np.char.str_len(refined_s) > 0))
    print(f"{log_prefix} Using refined component hub labels from {REFINED_HUB_LABEL_SIDECAR}: labeled_hubs={n_labeled:,}")
    out = np.array(refined_s, copy=True)
    return (out, True) if bool(return_tokens) else out


def _build_cluster_to_label_map_for_sample(
    adata_cells: ad.AnnData,
    sample_name: str,
    *,
    label_col: str,
    cluster_id_col: str = "cluster_id",
    target_dtype: Optional[np.dtype] = None,
) -> Dict:
    """Map per-sample hub cluster ID -> Leiden label using QC-passed cells."""
    if label_col not in adata_cells.obs.columns:
        raise KeyError(f"Missing '{label_col}' in adata_cells.obs")
    if "sample" not in adata_cells.obs.columns:
        raise KeyError("Missing 'sample' in adata_cells.obs")

    df = adata_cells.obs.loc[adata_cells.obs["sample"].astype(str) == str(sample_name), :]
    if df.shape[0] == 0:
        return {}
    cluster_vals = df[cluster_id_col] if cluster_id_col in df.columns else pd.Index(df.index.astype(str)).str.rsplit("-", n=1, expand=True).get_level_values(0)
    label_vals = df[label_col].astype(str)

    if target_dtype is not None and np.issubdtype(target_dtype, np.number):
        cluster_num = pd.to_numeric(pd.Series(cluster_vals).astype(str), errors="coerce")
        ok = cluster_num.notna()
        keys = cluster_num.loc[ok].astype(np.int64).to_numpy()
        vals = label_vals.loc[ok].to_numpy(dtype=object)
        return dict(zip(keys, vals))

    if target_dtype is not None:
        keys = pd.Series(cluster_vals).astype(str).to_numpy(dtype=object)
        vals = label_vals.to_numpy(dtype=object)
        return dict(zip(keys, vals))

    cluster_num = pd.to_numeric(pd.Series(cluster_vals).astype(str), errors="coerce")
    if float(cluster_num.notna().mean()) >= 0.99:
        keys = cluster_num.astype(np.int64).to_numpy()
    else:
        keys = pd.Series(cluster_vals).astype(str).to_numpy(dtype=object)
    vals = label_vals.to_numpy(dtype=object)
    return dict(zip(keys, vals))


def _clusters_to_rgba(
    clusters: np.ndarray,
    *,
    cluster_to_label: Dict,
    label_to_rgba: Dict[str, np.ndarray],
    unassigned_rgba: Tuple[int, int, int, int] = GSE_UNASSIGNED_RGBA,
    cluster_tokens: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Map cluster/refined-component IDs to RGBA without sorting mixed dtypes."""
    cluster_tokens = _cluster_array_to_tokens(clusters) if cluster_tokens is None else np.asarray(cluster_tokens).reshape(-1)
    codes, uniq = pd.factorize(cluster_tokens, sort=False)
    lut = np.empty((len(uniq), 4), dtype=np.uint8)
    ua = np.array(unassigned_rgba, dtype=np.uint8)
    lookup = _normalized_cluster_label_lookup(cluster_to_label)
    for i, token in enumerate(uniq):
        lab = lookup.get(str(token))
        rgba = label_to_rgba.get(str(lab)) if lab is not None else None
        lut[i] = rgba if rgba is not None else ua
    if codes.size == 0:
        return np.empty((0, 4), dtype=np.uint8)
    return lut[codes]

def _stable_uint32_seed_from_text(text_value: str, *, salt: int = RANDOM_SEED) -> int:
    """Return a process-stable uint32 seed from text without using Python hash()."""
    data = np.frombuffer(str(text_value).encode("utf-8", errors="ignore"), dtype=np.uint8)
    x = (2166136261 ^ int(salt)) & 0xFFFFFFFF
    for b in data:
        x ^= int(b)
        x = (x * 16777619) & 0xFFFFFFFF
    return int(x)


def _valid_cluster_key_mask_for_random_cell_render(
    clusters: np.ndarray,
    *,
    drop_cluster_values: Optional[set] = DROP_CLUSTER_VALUES,
) -> np.ndarray:
    """Return hubs whose CLUSTER_KEY value is a real aggregate-cell id."""
    clusters = np.asarray(clusters)
    mask = np.ones(clusters.shape[0], dtype=bool)
    clusters_s = None

    if np.issubdtype(clusters.dtype, np.floating):
        mask &= np.isfinite(clusters)
    elif clusters.dtype.kind in {"S", "U", "O"}:
        clusters_s = _decode_bytes_to_str_array(clusters).astype(str)
        mask &= ~np.isin(clusters_s, np.array(["", "nan", "None", "<NA>"], dtype=object))

    if drop_cluster_values:
        if np.issubdtype(clusters.dtype, np.number):
            for dv in drop_cluster_values:
                try:
                    dv_num = np.asarray(dv).astype(clusters.dtype).item()
                    mask &= clusters != dv_num
                except Exception:
                    # Rare mixed/object fallback; only materialize strings if needed.
                    clusters_s = _decode_bytes_to_str_array(clusters).astype(str)
                    mask &= clusters_s != str(dv)
        else:
            if clusters_s is None:
                clusters_s = _decode_bytes_to_str_array(clusters).astype(str)
            mask &= ~np.isin(clusters_s, np.array([str(x) for x in drop_cluster_values], dtype=object))
    return mask


def _postfilter_cluster_ids_for_sample(
    adata_cells: Optional[ad.AnnData],
    sample_name: str,
    *,
    cluster_id_col: str = GSE_CELL_SURFACE_CLUSTER_ID_COL,
    target_dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """Return cluster_id values for cells retained after QC/downsampling."""
    if adata_cells is None:
        return np.array([], dtype=object)
    if "sample" not in adata_cells.obs.columns:
        raise KeyError("Missing obs['sample']; cannot restrict surfaces to post-filter cells.")
    df = adata_cells.obs.loc[adata_cells.obs["sample"].astype(str) == str(sample_name), :]
    if df.shape[0] == 0:
        return np.array([], dtype=object)
    if cluster_id_col in df.columns:
        vals = pd.Series(df[cluster_id_col].astype(str).values)
    else:
        vals = pd.Series(pd.Index(df.index.astype(str)).str.rsplit("-", n=1, expand=True).get_level_values(0).astype(str))
    vals = vals.loc[vals.notna()].astype(str)
    vals = vals.loc[vals.str.len() > 0].drop_duplicates(keep="first")
    if vals.shape[0] == 0:
        return np.array([], dtype=object)
    if target_dtype is not None and np.issubdtype(np.dtype(target_dtype), np.number):
        num = pd.to_numeric(vals, errors="coerce")
        num = num.loc[num.notna()]
        if num.shape[0] == 0:
            return np.array([], dtype=np.int64)
        if np.issubdtype(np.dtype(target_dtype), np.integer):
            return num.astype(np.int64).to_numpy()
        return num.astype(float).to_numpy()
    return vals.astype(str).to_numpy(dtype=object)


def _cluster_membership_mask(clusters: np.ndarray, allowed_cluster_ids: np.ndarray) -> np.ndarray:
    """Return a hub mask whose cluster/refined-component token is allowed."""
    clusters = np.asarray(clusters)
    allowed_cluster_ids = np.asarray(allowed_cluster_ids)
    if allowed_cluster_ids.size == 0:
        return np.zeros(clusters.shape[0], dtype=bool)
    cluster_tokens = _cluster_array_to_tokens(clusters)
    allowed_tokens = pd.Series(_cluster_array_to_tokens(allowed_cluster_ids)).astype(str).drop_duplicates().to_numpy(dtype=object)
    return np.isin(cluster_tokens.astype(str), allowed_tokens)

def _cluster_id_set_fingerprint(
    cluster_ids: np.ndarray,
    *,
    sample_name: str,
    cluster_id_col: str = GSE_CELL_SURFACE_CLUSTER_ID_COL,
) -> Dict[str, object]:
    """Stable cache fingerprint for the post-filter cell-id set."""
    vals = pd.Series(np.asarray(cluster_ids)).astype(str).dropna().drop_duplicates().sort_values(kind="mergesort")
    h = hashlib.sha1()
    for v in vals.to_numpy(dtype=object):
        h.update(str(v).encode("utf-8", errors="ignore"))
        h.update(b"\n")
    preview_n = int(min(8, vals.shape[0]))
    return {
        "sample": str(sample_name),
        "cluster_id_col": str(cluster_id_col),
        "n_postfilter_cells": int(vals.shape[0]),
        "sha1": h.hexdigest(),
        "first_values": [str(x) for x in vals.iloc[:preview_n].tolist()],
        "last_values": [str(x) for x in vals.iloc[-preview_n:].tolist()] if preview_n > 0 else [],
    }


def _random_bright_rgba_lut(
    n: int,
    *,
    alpha_u8: int,
    seed: int,
    saturation_range: Tuple[float, float] = (0.70, 1.00),
    value_range: Tuple[float, float] = (0.82, 1.00),
) -> np.ndarray:
    """Generate deterministic high-saturation/high-value cell colors."""
    n = int(n)
    rgba = np.zeros((max(n, 0), 4), dtype=np.uint8)
    if n <= 0:
        return rgba
    rng = np.random.default_rng(int(seed))
    # Golden-ratio hue stepping gives broad rainbow coverage; the subsequent
    # shuffle keeps the assignment random-looking while remaining deterministic.
    hue = (rng.random() + np.arange(n, dtype=np.float64) * 0.6180339887498949) % 1.0
    rng.shuffle(hue)
    sat = rng.uniform(float(saturation_range[0]), float(saturation_range[1]), n)
    val = rng.uniform(float(value_range[0]), float(value_range[1]), n)
    rgb = mpl.colors.hsv_to_rgb(np.stack([hue, sat, val], axis=1))
    rgba[:, :3] = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    rgba[:, 3] = np.uint8(np.clip(int(alpha_u8), 0, 255))
    return rgba


def _clusters_to_label_strings(
    clusters: np.ndarray,
    *,
    cluster_to_label: Dict,
    unassigned_label: str = "__unassigned__",
    clusters_are_tokens: bool = False,
) -> np.ndarray:
    """Map per-hub cluster/refined-component IDs to Leiden label strings.

    ``clusters_are_tokens=True`` lets the PyVista hot path reuse the one
    per-sample token array instead of normalizing the full hub vector again.

    The PyVista render path scans this array many times. Keep it as a fixed-width
    NumPy unicode array, not ``object``; object arrays force per-element Python
    string comparison on ~1e7 hubs.
    """
    cluster_tokens = np.asarray(clusters).reshape(-1) if bool(clusters_are_tokens) else _cluster_array_to_tokens(clusters)
    codes, uniq = pd.factorize(cluster_tokens, sort=False)
    lookup = _normalized_cluster_label_lookup(cluster_to_label)
    lut_strs: List[str] = []
    for token in uniq:
        lab = lookup.get(str(token))
        lut_strs.append(str(lab) if lab is not None else str(unassigned_label))
    width = max([len(str(unassigned_label)), 1] + [len(x) for x in lut_strs])
    lut = np.asarray(lut_strs, dtype=f"U{width}")
    if codes.size == 0:
        return np.array([], dtype=lut.dtype)
    return lut[np.asarray(codes, dtype=np.int64)]


def _leiden_scatter_excluded_label_set() -> set:
    """Labels that should never be drawn in pyvista_gse_scatter_leiden PNGs."""
    return {str(x) for x in GSE_EXCLUDED_LEIDEN_LABELS if str(x) != ""}


def _mask_excluded_leiden_scatter_labels(label_strings: np.ndarray) -> np.ndarray:
    """Convert globally excluded Leiden labels, e.g. L0, to an explicit sentinel."""
    labels = np.asarray(label_strings).reshape(-1)
    if labels.dtype.kind == "S":
        labels = np.char.decode(labels, "utf-8")
    elif labels.dtype.kind != "U":
        labels = labels.astype(str)
    exclude = _leiden_scatter_excluded_label_set()
    if labels.size == 0 or not exclude:
        return labels
    exclude_arr = np.asarray(sorted(exclude), dtype=labels.dtype)
    excluded = np.isin(labels, exclude_arr)
    if not np.any(excluded):
        return labels
    sentinel = str(GSE_EXCLUDED_LABEL_SENTINEL)
    width = max(int(labels.dtype.itemsize // 4), len(sentinel), 1)
    out = labels.astype(f"U{width}", copy=True)
    out[excluded] = sentinel
    return out


def _label_code_cache(
    label_strings: np.ndarray,
    label_codes: Optional[np.ndarray] = None,
    label_values: Optional[np.ndarray] = None,
    label_code_of_label: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Return unicode labels plus integer codes for hot PyVista label scans."""
    labels = np.asarray(label_strings).reshape(-1)
    if labels.dtype.kind == "S":
        labels = np.char.decode(labels, "utf-8")
    elif labels.dtype.kind != "U":
        labels = labels.astype(str)

    codes_ok = label_codes is not None and np.asarray(label_codes).reshape(-1).shape[0] == labels.shape[0]
    values_ok = label_values is not None
    map_ok = isinstance(label_code_of_label, dict)
    if codes_ok and values_ok and map_ok:
        codes = np.asarray(label_codes).reshape(-1)
        values = np.asarray(label_values).astype(str)
        mapping = {str(k): int(v) for k, v in dict(label_code_of_label).items()}
        return labels, codes, values, mapping

    codes, uniques = pd.factorize(labels, sort=False)
    codes = np.asarray(codes, dtype=np.int32 if len(uniques) < np.iinfo(np.int32).max else np.int64)
    values = np.asarray(pd.Index(uniques).astype(str).to_numpy(), dtype=labels.dtype if labels.dtype.kind == "U" else object)
    mapping = {str(v): int(i) for i, v in enumerate(values.tolist())}
    return labels, codes, values, mapping


def _label_counts_from_codes(
    label_codes: np.ndarray,
    label_values: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
) -> pd.Series:
    """Count labels with np.bincount instead of constructing a 1e7-row object Series."""
    codes = np.asarray(label_codes).reshape(-1)
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
        codes = codes[mask_arr]
    codes = codes[codes >= 0]
    values = np.asarray(label_values).astype(str)
    if codes.size == 0 or values.size == 0:
        return pd.Series(dtype=np.int64)
    counts = np.bincount(codes.astype(np.int64, copy=False), minlength=int(values.shape[0])).astype(np.int64, copy=False)
    keep = counts > 0
    if not np.any(keep):
        return pd.Series(dtype=np.int64)
    out = pd.Series(counts[keep], index=pd.Index(values[keep].astype(str)))
    return out.sort_values(ascending=False, kind="mergesort")


def _label_indices_for_value(label_codes: np.ndarray, code_of_label: Dict[str, int], label: object) -> np.ndarray:
    code = code_of_label.get(str(label))
    if code is None:
        return np.zeros(0, dtype=np.int64)
    return np.flatnonzero(np.asarray(label_codes).reshape(-1) == int(code)).astype(np.int64, copy=False)


def _is_drawn_leiden_scatter_label_array(labels: np.ndarray) -> np.ndarray:
    """True for labels drawn as Leiden foreground/remainder in scatter modes."""
    arr, codes, values, code_of_label = _label_code_cache(labels)
    drawn = np.ones(arr.shape[0], dtype=bool)
    unassigned_code = code_of_label.get("__unassigned__")
    if unassigned_code is not None:
        drawn &= codes != int(unassigned_code)
    sentinel_code = code_of_label.get(str(GSE_EXCLUDED_LABEL_SENTINEL))
    if sentinel_code is not None:
        drawn &= codes != int(sentinel_code)
    for lab in _leiden_scatter_excluded_label_set():
        code = code_of_label.get(str(lab))
        if code is not None:
            drawn &= codes != int(code)
    return drawn

def _rng_subset_indices(idx: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    """Deterministically cap a set of point indices for display."""
    idx = np.asarray(idx, dtype=np.int64).ravel()
    max_points = int(max_points)
    if max_points <= 0 or idx.size <= max_points:
        return idx
    return np.sort(rng.choice(idx, size=max_points, replace=False).astype(np.int64, copy=False))



def _unique_cluster_values_for_filter(values: np.ndarray) -> np.ndarray:
    """Return unique normalized cluster/refined-component tokens preserving first-seen order."""
    arr = _cluster_array_to_tokens(values)
    if arr.size == 0:
        return np.array([], dtype=object)
    vals = pd.Series(arr).astype(str)
    vals = vals.loc[vals.str.len() > 0].drop_duplicates(keep="first")
    return vals.to_numpy(dtype=object)


def _write_leiden_colored_surface_filter_table(
    rows: Sequence[Dict[str, object]],
    *,
    out_dir: str,
) -> None:
    if not rows:
        return
    try:
        os.makedirs(str(out_dir), exist_ok=True)
        pd.DataFrame(rows).to_csv(
            os.path.join(str(out_dir), str(GSE_CELL_SURFACE_LEIDEN_FILTER_TABLE)),
            sep="\t",
            index=False,
        )
    except Exception as e:
        print(f"[PyVistaCellSurface] Warning: failed writing Leiden-colored cell filter table ({e}).")


def _enable_pyvista_antialiasing(plotter: "pv.Plotter", aa: object = None, *, log_prefix: str = "[PyVista]") -> None:
    """Enable anti-aliasing with visible diagnostics instead of silent fallback."""
    aa_type = str(GSE_ANTIALIASING if aa is None else aa).strip().lower()
    if not aa_type or aa_type in {"none", "off", "false", "0"}:
        return
    try:
        plotter.enable_anti_aliasing(aa_type)
        print(f"{log_prefix} Anti-aliasing enabled: {aa_type}")
        return
    except Exception as e:
        print(f"{log_prefix} Warning: anti-aliasing={aa_type!r} failed ({e}); trying PyVista default.")
    try:
        plotter.enable_anti_aliasing()
        print(f"{log_prefix} Anti-aliasing enabled with PyVista default.")
    except Exception as e:
        print(f"{log_prefix} Warning: anti-aliasing could not be enabled ({e}).")


def _save_pyvista_screenshot(plotter: "pv.Plotter", out_png: str, *, log_prefix: str = "[PyVista]") -> bool:
    """Render and save a supersampled PyVista screenshot with stable fallbacks."""
    scale = int(max(1, globals().get("GSE_SCREENSHOT_SCALE", 1)))
    try:
        plotter.render()
    except Exception:
        pass
    try:
        plotter.screenshot(str(out_png), scale=scale)
        print(f"{log_prefix} Saved: {out_png} (screenshot_scale={scale})")
        return True
    except TypeError:
        try:
            old_scale = getattr(plotter, "image_scale", 1)
            plotter.image_scale = scale
            plotter.screenshot(str(out_png))
            try:
                plotter.image_scale = old_scale
            except Exception:
                pass
            print(f"{log_prefix} Saved: {out_png} (image_scale={scale})")
            return True
        except Exception as e:
            print(f"{log_prefix} Warning: scaled screenshot failed ({e}); trying show(screenshot=...).")
    except Exception as e:
        print(f"{log_prefix} Warning: scaled screenshot failed ({e}); trying show(screenshot=...).")
    try:
        plotter.show(screenshot=str(out_png), auto_close=False)
        print(f"{log_prefix} Saved: {out_png} (fallback show screenshot)")
        return True
    except Exception as e:
        print(f"{log_prefix} Failed rendering screenshot {out_png} ({e})")
        return False


def _add_gse_point_actor(
    plotter: "pv.Plotter",
    points: np.ndarray,
    *,
    color: str,
    opacity: float,
    point_size: float,
    name: str,
) -> None:
    """Add one independently controlled point-cloud actor."""
    if pv is None:
        return
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] == 0:
        return
    cloud = pv.PolyData(points)
    actor = plotter.add_mesh(
        cloud,
        color=str(color),
        opacity=float(opacity),
        render_points_as_spheres=bool(GSE_RENDER_POINTS_AS_SPHERES),
        point_size=float(max(0.3, point_size)),
        lighting=bool(GSE_LIGHTING),
        name=str(name),
    )
    if GSE_LIGHTING:
        try:
            prop = actor.GetProperty()
            prop.SetAmbient(float(GSE_MATERIAL_AMBIENT))
            prop.SetDiffuse(float(GSE_MATERIAL_DIFFUSE))
            prop.SetSpecular(float(GSE_MATERIAL_SPECULAR))
            prop.SetSpecularPower(float(GSE_MATERIAL_SPECULAR_POWER))
        except Exception:
            pass


def _add_balanced_leiden_point_actors(
    plotter: "pv.Plotter",
    coords: np.ndarray,
    label_strings: np.ndarray,
    *,
    label_palette_hex: Dict[str, str],
    rng: np.random.Generator,
    cluster_tokens: Optional[np.ndarray] = None,
    label_codes: Optional[np.ndarray] = None,
    label_values: Optional[np.ndarray] = None,
    label_code_of_label: Optional[Dict[str, int]] = None,
) -> Dict[str, object]:
    """Render a dense Leiden cloud after dropping reserved/excluded labels.

    L0 is the reserved gray palette label and is excluded from every PNG written
    under pyvista_gse_scatter_leiden.  No fallback "dominant gray context" actor
    is drawn; all plotted labels use identical opacity and point size so the
    full-Leiden, top-5, and arc render modes share the same visual scale.
    """
    labels, label_codes, label_values, label_code_of_label = _label_code_cache(
        label_strings,
        label_codes=label_codes,
        label_values=label_values,
        label_code_of_label=label_code_of_label,
    )
    cluster_tokens_arr = None
    if cluster_tokens is not None:
        # The parameter is already normalized by the per-sample PyVista hot path.
        # Do not re-tokenize the full hub array here.
        cluster_tokens_arr = np.asarray(cluster_tokens).reshape(-1)
        if cluster_tokens_arr.shape[0] != labels.shape[0]:
            cluster_tokens_arr = None
    unassigned_code = label_code_of_label.get("__unassigned__")
    assigned_mask = (label_codes >= 0) if unassigned_code is None else (label_codes != int(unassigned_code))
    if not np.any(assigned_mask):
        return {"balanced_render": False, "reason": "no_assigned_labels"}

    excluded_labels = {str(x) for x in GSE_EXCLUDED_LEIDEN_LABELS} | {str(GSE_EXCLUDED_LABEL_SENTINEL)}
    label_counts_all = _label_counts_from_codes(label_codes, label_values, mask=assigned_mask)
    plotted_label_counts = label_counts_all.loc[[str(x) for x in label_counts_all.index.astype(str) if str(x) not in excluded_labels]]
    if plotted_label_counts.empty:
        return {
            "balanced_render": False,
            "reason": "no_nonexcluded_labels",
            "excluded_labels": sorted([str(x) for x in excluded_labels if str(x) in set(label_counts_all.index.astype(str))], key=_natural_sort_key),
            "label_counts_all": {str(k): int(v) for k, v in label_counts_all.items()},
        }

    plotted_labels = [str(x) for x in plotted_label_counts.index.astype(str).tolist()]
    minor_cap = int(GSE_MINOR_CLUSTER_MAX_POINTS)
    total_cap = int(GSE_BALANCED_TOTAL_MAX_POINTS)
    if total_cap > 0 and len(plotted_labels) > 0:
        if minor_cap > 0:
            minor_cap = min(minor_cap, max(1, total_cap // max(1, len(plotted_labels))))
        else:
            minor_cap = max(1, total_cap // max(1, len(plotted_labels)))

    foreground_index_parts: List[np.ndarray] = []
    draw_summary: Dict[str, object] = {
        "balanced_render": True,
        "excluded_labels": sorted([str(x) for x in excluded_labels if str(x) in set(label_counts_all.index.astype(str))], key=_natural_sort_key),
        "major_labels": [],
        "minor_labels": sorted(plotted_labels, key=_natural_sort_key),
        "label_counts": {str(k): int(v) for k, v in plotted_label_counts.items()},
        "label_counts_all": {str(k): int(v) for k, v in label_counts_all.items()},
        "actors": [],
        "foreground_cluster_ids": [],
        "filter_source": "leiden_pyvista_draw_summary",
    }
    foreground_cluster_parts: List[np.ndarray] = []

    if bool(GSE_RENDER_RARE_CLUSTERS_LAST):
        foreground_order = sorted(plotted_labels, key=lambda x: int(plotted_label_counts.get(x, 0)), reverse=True)
    else:
        foreground_order = sorted(plotted_labels, key=_natural_sort_key)

    for lab in foreground_order:
        idx_all = _label_indices_for_value(label_codes, label_code_of_label, lab)
        if idx_all.size == 0:
            continue
        idx = _rng_subset_indices(idx_all, minor_cap, rng)
        if idx.size > 0:
            foreground_index_parts.append(np.asarray(idx, dtype=np.int64))
            if cluster_tokens_arr is not None:
                foreground_cluster_parts.append(cluster_tokens_arr[idx])
        color = _leiden_render_color_for_label(str(lab), label_palette_hex)
        _add_gse_point_actor(
            plotter,
            coords[idx],
            color=color,
            opacity=float(GSE_MINOR_CLUSTER_OPACITY),
            point_size=float(GSE_MINOR_CLUSTER_POINT_SIZE),
            name=f"gse_foreground_L{lab}",
        )
        draw_summary["actors"].append({
            "label": str(lab),
            "n_drawn": int(idx.size),
            "n_total": int(idx_all.size),
            "role": "foreground",
            "opacity": float(GSE_MINOR_CLUSTER_OPACITY),
            "point_size": float(GSE_MINOR_CLUSTER_POINT_SIZE),
            "color": str(color),
        })

    if foreground_index_parts:
        fg = np.unique(np.concatenate(foreground_index_parts).astype(np.int64, copy=False))
    else:
        fg = np.zeros(0, dtype=np.int64)
    if foreground_cluster_parts:
        fg_clusters = _unique_cluster_values_for_filter(np.concatenate(foreground_cluster_parts))
        draw_summary["foreground_cluster_ids"] = [str(x) for x in pd.Series(fg_clusters).astype(str).tolist()]
        draw_summary["n_foreground_cells"] = int(fg_clusters.size)
    else:
        draw_summary["n_foreground_cells"] = 0
    draw_summary["n_foreground_points"] = int(fg.size)
    draw_summary["_foreground_point_indices"] = fg
    return draw_summary

def _sample_top5_minor_cap_effective(n_major_labels: int, n_minor_labels: int) -> int:
    """Mirror the full-Leiden plotted-label cap without drawing excluded labels."""
    minor_cap = int(GSE_MINOR_CLUSTER_MAX_POINTS)
    total_cap = int(GSE_BALANCED_TOTAL_MAX_POINTS)
    n_minor = int(max(0, n_minor_labels))
    if total_cap > 0 and n_minor > 0:
        if minor_cap > 0:
            minor_cap = min(minor_cap, max(1, total_cap // max(1, n_minor)))
        else:
            minor_cap = max(1, total_cap // max(1, n_minor))
    return int(minor_cap)


def _select_sample_top_plotted_labels(
    label_strings: np.ndarray,
    *,
    label_codes: Optional[np.ndarray] = None,
    label_values: Optional[np.ndarray] = None,
    label_code_of_label: Optional[Dict[str, int]] = None,
) -> Dict[str, object]:
    """Select current-sample top-N plotted labels after excluding L0/unassigned."""
    labels, label_codes, label_values, label_code_of_label = _label_code_cache(
        label_strings,
        label_codes=label_codes,
        label_values=label_values,
        label_code_of_label=label_code_of_label,
    )
    unassigned_code = label_code_of_label.get("__unassigned__")
    assigned_mask = (label_codes >= 0) if unassigned_code is None else (label_codes != int(unassigned_code))
    if not np.any(assigned_mask):
        return {"top_labels": [], "reason": "no_assigned_labels"}
    label_counts_all = _label_counts_from_codes(label_codes, label_values, mask=assigned_mask)
    if label_counts_all.empty:
        return {"top_labels": [], "reason": "no_label_counts"}

    exclude = {str(x) for x in GSE_EXCLUDED_LEIDEN_LABELS} | {str(GSE_EXCLUDED_LABEL_SENTINEL)}
    plotted_labels = [str(x) for x in label_counts_all.index.astype(str).tolist() if str(x) not in exclude]
    if not plotted_labels:
        return {
            "top_labels": [],
            "reason": "no_nonexcluded_plotted_labels",
            "excluded_labels": sorted([str(x) for x in exclude if str(x) in set(label_counts_all.index.astype(str))], key=_natural_sort_key),
            "label_counts_all": {str(k): int(v) for k, v in label_counts_all.items()},
        }
    minor_cap = _sample_top5_minor_cap_effective(0, len(plotted_labels))

    plotted_counts: Dict[str, int] = {}
    for lab in plotted_labels:
        n = int(label_counts_all.get(str(lab), 0))
        plotted_counts[str(lab)] = int(min(n, minor_cap)) if minor_cap > 0 else int(n)

    top_labels = sorted(
        [lab for lab in plotted_labels if plotted_counts.get(lab, 0) > 0],
        key=lambda lab: (-int(plotted_counts.get(str(lab), 0)), -int(label_counts_all.get(str(lab), 0)), _natural_sort_key(str(lab))),
    )[: int(max(0, GSE_SAMPLE_TOP_PLOTTED_CLUSTER_N))]
    return {
        "top_labels": [str(x) for x in top_labels],
        "excluded_labels": sorted([str(x) for x in exclude if str(x) in set(label_counts_all.index.astype(str))], key=_natural_sort_key),
        "major_labels_for_cap": [],
        "minor_cap_effective": int(minor_cap),
        "label_counts": {str(k): int(label_counts_all.get(str(k), 0)) for k in plotted_labels},
        "label_counts_all": {str(k): int(v) for k, v in label_counts_all.items()},
        "plotted_counts": {str(k): int(v) for k, v in plotted_counts.items()},
        "reason": "ok" if top_labels else "no_nonexcluded_plotted_labels",
    }

def _configure_extra_gse_plotter(plotter: "pv.Plotter") -> None:
    """Apply the same view/backdrop options as the base Leiden GSE renderer."""
    try:
        if GSE_BACKGROUND_TOP is not None:
            plotter.set_background(GSE_BACKGROUND, top=GSE_BACKGROUND_TOP)
        else:
            plotter.set_background(GSE_BACKGROUND)
    except Exception:
        plotter.set_background(GSE_BACKGROUND)
    _enable_pyvista_antialiasing(plotter, GSE_ANTIALIASING, log_prefix="[PyVista]")
    if GSE_ENABLE_LIGHTKIT and hasattr(plotter, "enable_lightkit"):
        try:
            plotter.enable_lightkit()
        except Exception:
            pass
    if GSE_ENABLE_EYE_DOME_LIGHTING:
        try:
            plotter.enable_eye_dome_lighting(strength=float(GSE_EDL_STRENGTH), radius=float(GSE_EDL_RADIUS))
        except TypeError:
            try:
                plotter.enable_eye_dome_lighting()
            except Exception:
                pass
        except Exception:
            pass
    if GSE_ENABLE_SSAO:
        try:
            plotter.enable_ssao(radius=float(GSE_SSAO_RADIUS), bias=float(GSE_SSAO_BIAS), kernel_size=int(GSE_SSAO_KERNEL_SIZE))
        except TypeError:
            try:
                plotter.enable_ssao()
            except Exception:
                pass
        except Exception:
            pass
    plotter.camera.parallel_projection = bool(GSE_CAMERA_PARALLEL_PROJECTION)
    if not plotter.camera.parallel_projection:
        try:
            plotter.camera.view_angle = float(GSE_CAMERA_VIEW_ANGLE_DEG)
        except Exception:
            pass


def _set_gse_view_camera(
    plotter: "pv.Plotter",
    *,
    az: float,
    el: float,
    pts_samp: np.ndarray,
    pts_mid: np.ndarray,
    radius_guess: float,
    focal: np.ndarray,
) -> None:
    cam_pos_unit = _camera_position_from_az_el(azimuth_deg=float(az), elevation_deg=float(el), radius=1.0, center=focal)
    cam_dir = cam_pos_unit - focal
    cam_dir /= np.linalg.norm(cam_dir) + 1e-12
    if plotter.camera.parallel_projection:
        cam_pos = focal + cam_dir * float(radius_guess)
        plotter.camera_position = (cam_pos.tolist(), focal.tolist(), (0.0, 0.0, 1.0))
        plotter.camera.parallel_scale = _parallel_scale_for_midpoints(
            pts_samp,
            pts_mid,
            camera_pos=cam_pos,
            focal=focal,
            window_size=GSE_WINDOW_SIZE,
            frame_fraction=float(GSE_INFRAME_FRACTION),
        )
    else:
        dist = _perspective_distance_for_midpoints(
            pts_samp,
            pts_mid,
            camera_pos=(focal + cam_dir),
            focal=focal,
            window_size=GSE_WINDOW_SIZE,
            view_angle_deg=float(GSE_CAMERA_VIEW_ANGLE_DEG),
            frame_fraction=float(GSE_INFRAME_FRACTION),
            near_quantile=float(GSE_INFRAME_NEAR_QUANTILE),
        )
        cam_pos = focal + cam_dir * float(dist)
        plotter.camera_position = (cam_pos.tolist(), focal.tolist(), (0.0, 0.0, 1.0))
        try:
            plotter.reset_camera_clipping_range()
        except Exception:
            pass

def _gse_view_id(view_index: int, az: float, el: float) -> str:
    """Stable view identifier shared by Leiden and arc PyVista routes."""
    return f"view{int(view_index):02d}_az{float(az):.0f}_el{float(el):.0f}"


def _camera_state_for_manifest(plotter: "pv.Plotter") -> Dict[str, object]:
    """Serialize enough PyVista camera state to audit view pairing."""
    try:
        pos, focal, up = plotter.camera_position
    except Exception:
        pos, focal, up = ([np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [0.0, 0.0, 1.0])
    cam = getattr(plotter, "camera", None)
    return {
        "camera_position_json": json.dumps([float(x) for x in np.asarray(pos, dtype=float).reshape(-1)[:3]]),
        "camera_focal_point_json": json.dumps([float(x) for x in np.asarray(focal, dtype=float).reshape(-1)[:3]]),
        "camera_view_up_json": json.dumps([float(x) for x in np.asarray(up, dtype=float).reshape(-1)[:3]]),
        "camera_parallel_projection": bool(getattr(cam, "parallel_projection", False)),
        "camera_parallel_scale": float(getattr(cam, "parallel_scale", float("nan"))),
        "camera_view_angle_deg": float(getattr(cam, "view_angle", float("nan"))),
    }


def _write_leiden_view_manifest(out_dir: str, sample_name: str, rows: Sequence[Dict[str, object]]) -> Optional[str]:
    """Write the canonical per-sample Leiden view registry consumed by the arc renderer."""
    rows = list(rows or [])
    if not rows:
        return None
    os.makedirs(str(out_dir), exist_ok=True)
    tsv_path = os.path.join(str(out_dir), f"{sample_name}_leiden_view_manifest.tsv")
    json_path = os.path.join(str(out_dir), f"{sample_name}_leiden_view_manifest.json")
    try:
        pd.DataFrame(rows).to_csv(tsv_path, sep="\t", index=False)
        with open(json_path, "w") as fh:
            json.dump(rows, fh, indent=2, sort_keys=True, default=str)
        return tsv_path
    except Exception as e:
        print(f"[PyVista] Warning: failed writing Leiden view manifest for {sample_name} ({e}).")
        return None



def _render_sample_top_plotted_leiden_snapshots(
    *,
    sample_name: str,
    coords: np.ndarray,
    label_strings: np.ndarray,
    label_palette_hex: Dict[str, str],
    pts_samp: np.ndarray,
    pts_mid: np.ndarray,
    radius_guess: float,
    out_dir: str,
    label_codes: Optional[np.ndarray] = None,
    label_values: Optional[np.ndarray] = None,
    label_code_of_label: Optional[Dict[str, int]] = None,
) -> Optional[Dict[str, object]]:
    """Supplemental top-5 current-sample focus render with no gray L0 leakage."""
    if pv is None or not bool(GSE_SAMPLE_TOP_PLOTTED_CLUSTER_RENDER_ENABLE):
        return None
    labels, label_codes, label_values, label_code_of_label = _label_code_cache(
        label_strings,
        label_codes=label_codes,
        label_values=label_values,
        label_code_of_label=label_code_of_label,
    )
    selection = _select_sample_top_plotted_labels(
        labels,
        label_codes=label_codes,
        label_values=label_values,
        label_code_of_label=label_code_of_label,
    )
    top_labels = [str(x) for x in selection.get("top_labels", [])]
    if not top_labels:
        print(f"[PyVistaTop5] No eligible labels for {sample_name}; reason={selection.get('reason', '')}")
        return {"sample": str(sample_name), "status": str(selection.get("reason", "no_top_labels")), "top_labels": ""}

    top_set = set(top_labels)
    excluded_set = {str(x) for x in GSE_EXCLUDED_LEIDEN_LABELS} | {str(GSE_EXCLUDED_LABEL_SENTINEL)}
    seed = _stable_uint32_seed_from_text(f"{sample_name}|{GSE_SAMPLE_TOP_PLOTTED_CLUSTER_FILE_TAG}", salt=int(RANDOM_SEED) + 271828)
    rng = np.random.default_rng(seed)
    plotter = pv.Plotter(off_screen=True, window_size=GSE_WINDOW_SIZE)
    _configure_extra_gse_plotter(plotter)

    # No need to materialize plotted_assigned_idx or scan all labels here; the
    # selection dictionary already contains only non-unassigned, non-excluded labels.
    label_counts_for_selection = selection.get("label_counts", {}) if isinstance(selection.get("label_counts", {}), dict) else {}
    remainder_labels = sorted(
        [str(lab) for lab in label_counts_for_selection.keys() if str(lab) not in top_set],
        key=lambda lab: int(label_counts_for_selection.get(str(lab), 0)),
        reverse=True,
    )

    remainder_drawn: Dict[str, int] = {}
    for lab in remainder_labels:
        idx_all = _label_indices_for_value(label_codes, label_code_of_label, lab)
        if idx_all.size == 0:
            continue
        idx = _rng_subset_indices(idx_all, int(GSE_SAMPLE_TOP_PLOTTED_REMAINDER_MAX_POINTS), rng)
        color = str(GSE_SAMPLE_TOP_PLOTTED_REMAINDER_COLOR)
        remainder_drawn[str(lab)] = int(idx.size)
        _add_gse_point_actor(
            plotter,
            coords[idx],
            color=color,
            opacity=float(GSE_SAMPLE_TOP_PLOTTED_REMAINDER_OPACITY),
            point_size=float(GSE_SAMPLE_TOP_PLOTTED_REMAINDER_POINT_SIZE),
            name=f"gse_{GSE_SAMPLE_TOP_PLOTTED_CLUSTER_FILE_TAG}_muted_L{lab}",
        )

    label_counts = selection.get("label_counts", {}) if isinstance(selection.get("label_counts", {}), dict) else {}
    minor_cap = int(selection.get("minor_cap_effective", GSE_MINOR_CLUSTER_MAX_POINTS) or GSE_MINOR_CLUSTER_MAX_POINTS)
    if bool(GSE_RENDER_RARE_CLUSTERS_LAST):
        top_draw_order = sorted(top_labels, key=lambda x: int(label_counts.get(str(x), 0)), reverse=True)
    else:
        top_draw_order = sorted(top_labels, key=_natural_sort_key)
    top_drawn: Dict[str, int] = {}
    for lab in top_draw_order:
        idx_all = _label_indices_for_value(label_codes, label_code_of_label, lab)
        if idx_all.size == 0:
            continue
        idx = _rng_subset_indices(idx_all, minor_cap, rng)
        top_drawn[str(lab)] = int(idx.size)
        _add_gse_point_actor(
            plotter,
            coords[idx],
            color=_leiden_render_color_for_label(str(lab), label_palette_hex),
            opacity=float(GSE_SAMPLE_TOP_PLOTTED_HIGHLIGHT_OPACITY),
            point_size=float(GSE_SAMPLE_TOP_PLOTTED_HIGHLIGHT_POINT_SIZE),
            name=f"gse_{GSE_SAMPLE_TOP_PLOTTED_CLUSTER_FILE_TAG}_L{lab}",
        )

    focal = np.array([0.0, 0.0, 0.0], dtype=float)
    for vi, (az, el) in enumerate(GSE_VIEW_ANGLES_DEG, start=1):
        _set_gse_view_camera(plotter, az=float(az), el=float(el), pts_samp=pts_samp, pts_mid=pts_mid, radius_guess=float(radius_guess), focal=focal)
        if GSE_SCALE_BAR_ENABLE:
            try:
                _add_camera_space_scale_bar(plotter, pts_samp)
            except Exception as e:
                print(f"[PyVistaTop5] Warning: failed to add scale bar ({e})")
        out_png = os.path.join(
            out_dir,
            f"{sample_name}_GSE_scatter_{GSE_SAMPLE_TOP_PLOTTED_CLUSTER_FILE_TAG}_view{vi:02d}_az{float(az):.0f}_el{float(el):.0f}.png",
        )
        if _save_pyvista_screenshot(plotter, out_png, log_prefix="[PyVistaTop5]"):
            print(f"[PyVistaTop5] Saved: {out_png}")
        else:
            print(f"[PyVistaTop5] Failed rendering {sample_name} view {vi}")

    try:
        plotter.close()
    except Exception:
        pass
    return {
        "sample": str(sample_name),
        "status": "written",
        "top_labels": ",".join(top_labels),
        "excluded_labels": ",".join([str(x) for x in selection.get("excluded_labels", [])]),
        "minor_cap_effective": int(selection.get("minor_cap_effective", 0) or 0),
        "n_top_labels": int(len(top_labels)),
        "top_label_drawn_counts_json": json.dumps(top_drawn, sort_keys=True),
        "remainder_label_drawn_counts_json": json.dumps(remainder_drawn, sort_keys=True),
        "label_counts_json": json.dumps(selection.get("label_counts", {}), sort_keys=True),
        "label_counts_all_json": json.dumps(selection.get("label_counts_all", {}), sort_keys=True),
        "plotted_counts_json": json.dumps(selection.get("plotted_counts", {}), sort_keys=True),
        "remainder_mode": "neutral_gray_no_l0_same_opacity_point_size",
        "remainder_color": str(GSE_SAMPLE_TOP_PLOTTED_REMAINDER_COLOR),
        "remainder_opacity": float(GSE_SAMPLE_TOP_PLOTTED_REMAINDER_OPACITY),
        "remainder_point_size": float(GSE_SAMPLE_TOP_PLOTTED_REMAINDER_POINT_SIZE),
        "highlight_opacity": float(GSE_SAMPLE_TOP_PLOTTED_HIGHLIGHT_OPACITY),
        "highlight_point_size": float(GSE_SAMPLE_TOP_PLOTTED_HIGHLIGHT_POINT_SIZE),
    }

def _camera_basis(
    camera_pos: np.ndarray,
    focal: np.ndarray,
    *,
    view_up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_pos = np.asarray(camera_pos, dtype=float)
    focal = np.asarray(focal, dtype=float)
    view_dir = focal - camera_pos
    view_dir /= np.linalg.norm(view_dir) + 1e-12
    up = np.asarray(view_up, dtype=float)
    up /= np.linalg.norm(up) + 1e-12
    right = np.cross(view_dir, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.asarray((0.0, 1.0, 0.0), dtype=float)
        right = np.cross(view_dir, up)
    right /= np.linalg.norm(right) + 1e-12
    up = np.cross(right, view_dir)
    up /= np.linalg.norm(up) + 1e-12
    return view_dir, right, up


def _parallel_scale_for_midpoints(
    pts_sample: np.ndarray,
    pts_mid: np.ndarray,
    *,
    camera_pos: np.ndarray,
    focal: np.ndarray,
    window_size: Tuple[int, int],
    frame_fraction: float = 0.99,
    pad: float = 1.05,
) -> float:
    _, right, up = _camera_basis(camera_pos, focal)
    rel_all = pts_sample - focal
    u_all = rel_all @ right
    v_all = rel_all @ up
    rel_mid = pts_mid - focal
    u_mid = rel_mid @ right
    v_mid = rel_mid @ up
    frac = float(np.clip(frame_fraction, 0.0, 0.999))
    tail = float(np.clip(0.5 * (1.0 - frac), 0.0, 0.49))
    u_lo, u_hi = np.quantile(u_all, [tail, 1.0 - tail])
    v_lo, v_hi = np.quantile(v_all, [tail, 1.0 - tail])
    u_ext_all = float(max(abs(u_lo), abs(u_hi)))
    v_ext_all = float(max(abs(v_lo), abs(v_hi)))
    u_ext_mid = float(np.max(np.abs(u_mid))) if u_mid.size else 0.0
    v_ext_mid = float(np.max(np.abs(v_mid))) if v_mid.size else 0.0
    aspect = float(window_size[0]) / float(window_size[1])
    scale_mid = max(2.0 * v_ext_mid, 2.0 * u_ext_mid / aspect)
    scale_all = max(v_ext_all, u_ext_all / aspect)
    return pad * max(scale_mid, scale_all, 1e-6)


def _perspective_distance_for_midpoints(
    pts_sample: np.ndarray,
    pts_mid: np.ndarray,
    *,
    camera_pos: np.ndarray,
    focal: np.ndarray,
    window_size: Tuple[int, int],
    view_angle_deg: float,
    frame_fraction: float = 0.99,
    pad: float = 1.05,
    near_quantile: float = 0.005,
) -> float:
    view_dir, right, up = _camera_basis(camera_pos, focal)
    rel_all = pts_sample - focal
    u_all = rel_all @ right
    v_all = rel_all @ up
    w_all = rel_all @ view_dir
    rel_mid = pts_mid - focal
    u_mid = rel_mid @ right
    v_mid = rel_mid @ up
    frac = float(np.clip(frame_fraction, 0.0, 0.999))
    tail = float(np.clip(0.5 * (1.0 - frac), 0.0, 0.49))
    u_lo, u_hi = np.quantile(u_all, [tail, 1.0 - tail])
    v_lo, v_hi = np.quantile(v_all, [tail, 1.0 - tail])
    u_ext_all = float(max(abs(u_lo), abs(u_hi)))
    v_ext_all = float(max(abs(v_lo), abs(v_hi)))
    u_ext_mid = float(np.max(np.abs(u_mid))) if u_mid.size else 0.0
    v_ext_mid = float(np.max(np.abs(v_mid))) if v_mid.size else 0.0
    w_near = float(np.quantile(w_all, float(near_quantile))) if w_all.size else 0.0
    aspect = float(window_size[0]) / float(window_size[1])
    scale_mid = max(2.0 * v_ext_mid, 2.0 * u_ext_mid / aspect)
    scale_all = max(v_ext_all, u_ext_all / aspect)
    half_height = max(scale_mid, scale_all, 1e-6)
    tan_half = float(np.tan(np.radians(float(view_angle_deg)) * 0.5))
    tan_half = max(tan_half, 1e-6)
    dist = half_height / tan_half
    dist = dist - w_near
    return float(pad * max(dist, 1e-6))


def _camera_position_from_az_el(*, azimuth_deg: float, elevation_deg: float, radius: float, center: np.ndarray) -> np.ndarray:
    az = np.radians(float(azimuth_deg))
    el = np.radians(float(elevation_deg))
    x = float(center[0] + radius * np.cos(el) * np.sin(az))
    y = float(center[1] + radius * np.cos(el) * np.cos(az))
    z = float(center[2] + radius * np.sin(el))
    return np.array([x, y, z], dtype=float)


def _add_camera_space_scale_bar(
    plotter: "pv.Plotter",
    pts_for_depth: np.ndarray,
    *,
    length: float = GSE_SCALE_BAR_LENGTH,
    margin_frac: Tuple[float, float] = GSE_SCALE_BAR_MARGIN_FRAC,
    color: str = GSE_SCALE_BAR_COLOR,
    line_width: float = GSE_SCALE_BAR_LINE_WIDTH,
    actor_name: str = GSE_SCALE_BAR_ACTOR_NAME,
) -> None:
    """Add a fixed-length line in camera space near the lower-left corner."""
    if pv is None:
        return
    pts = np.asarray(pts_for_depth, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] != 3:
        return

    camera = plotter.camera
    camera_pos = np.asarray(camera.position, dtype=float)
    focal = np.asarray(camera.focal_point, dtype=float)
    view_up = np.asarray(camera.up, dtype=float)
    view_dir, right, up = _camera_basis(camera_pos, focal, view_up=tuple(view_up.tolist()))

    depths = (pts - camera_pos[None, :]) @ view_dir
    depths = depths[np.isfinite(depths) & (depths > 1e-6)]
    depth = float(np.median(depths)) if depths.size else float(np.linalg.norm(focal - camera_pos))
    plane_center = camera_pos + view_dir * depth
    aspect = float(plotter.window_size[0]) / float(plotter.window_size[1])

    if bool(camera.parallel_projection):
        half_h = float(getattr(camera, "parallel_scale", 1.0))
        half_w = float(aspect * half_h)
    else:
        view_angle = float(getattr(camera, "view_angle", GSE_CAMERA_VIEW_ANGLE_DEG))
        half_h = float(depth * np.tan(np.radians(view_angle) * 0.5))
        half_w = float(aspect * half_h)

    mx, my = float(margin_frac[0]), float(margin_frac[1])
    start_point = plane_center + right * (-half_w * (1.0 - mx)) + up * (-half_h * (1.0 - my))
    end_point = start_point + right * float(length)
    scale_line = pv.Line(start_point, end_point)

    try:
        plotter.remove_actor(str(actor_name), reset_camera=False)
    except Exception:
        try:
            plotter.remove_actor(str(actor_name))
        except Exception:
            pass
    try:
        plotter.add_mesh(scale_line, color=str(color), line_width=float(line_width), lighting=False, name=str(actor_name))
    except TypeError:
        plotter.add_mesh(scale_line, color=str(color), line_width=float(line_width), name=str(actor_name))


def render_gse_pyvista_snapshots_for_all_samples(
    adata_cells: ad.AnnData,
    *,
    file_paths: Sequence[str],
    out_dir: str,
    coord_keys: Tuple[str, str, str] = GSE_COORD_KEYS,
    cluster_key: str = CLUSTER_KEY,
    label_col: str = LEIDEN_KEY,
    label_palette_hex: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, object]]:
    """Render hub-level GSE point clouds colored by Leiden labels.

    The label_palette_hex argument is the exact palette used by the 2D figures.
    """
    if not GSE_RENDER_ENABLE:
        print("[PyVista] GSE rendering disabled.")
        return {}
    if pv is None:
        print("[PyVista] pyvista not available; skipping GSE rendering.")
        return {}
    if label_col not in adata_cells.obs.columns:
        print(f"[PyVista] '{label_col}' not present; skipping GSE rendering.")
        return {}

    os.makedirs(out_dir, exist_ok=True)
    _pv_start_virtual_framebuffer_if_needed()

    safe_key = str(label_col).replace(os.sep, "_")
    if label_palette_hex is None:
        label_palette_hex = leiden_palette_from_adata(adata_cells, label_col)
    label_palette_hex = {str(k): str(v) for k, v in label_palette_hex.items()}
    save_palette_tsvs(label_palette_hex, key=safe_key, out_dirs=[TABLE_DIR, out_dir])
    save_categorical_palette_legend(
        label_palette_hex,
        out_pdf=os.path.join(out_dir, f"{safe_key}_palette_legend.pdf"),
        out_png=os.path.join(out_dir, f"{safe_key}_palette_legend.png"),
        title=f"{label_col} color key",
        subtitle="Exact palette used by FigureS2 panels and PyVista Leiden renders.",
        max_cols=3,
    )

    rng = np.random.default_rng(RANDOM_SEED)
    top_plotted_render_rows: List[Dict[str, object]] = []
    leiden_foreground_by_sample: Dict[str, Dict[str, object]] = {}
    GSE_LEIDEN_FOREGROUND_CLUSTER_IDS_BY_SAMPLE.clear()

    for fp in file_paths:
        sample_name = _infer_sample_name_from_filepath(fp)
        print(f"\n[PyVista] Rendering GSE point cloud for {sample_name} ({fp})")
        if not os.path.exists(fp):
            print(f"[PyVista] File not found: {fp}. Skipping.")
            continue

        try:
            coords, clusters = _read_gse_coords_and_cluster_from_h5ad(fp, coord_keys=coord_keys, cluster_key=cluster_key)
        except Exception as e:
            print(f"[PyVista] Failed reading obs columns from {fp}: {e}")
            continue
        clusters, clusters_are_tokens = _replace_clusters_with_refined_sidecar(
            fp,
            clusters,
            log_prefix="[PyVista]",
            return_tokens=True,
        )

        n_pts = int(coords.shape[0])
        # Preserve parent final.h5ad row ids for exact reuse by the cell-to-slice
        # arc renderer.  The Leiden renderer may later subset/reorder coords, so
        # all drawn point-index summaries must be converted back through this array.
        original_parent_rows = np.arange(n_pts, dtype=np.int64)
        print(f"[PyVista] Points: {n_pts:,}")
        cluster_to_label = _build_cluster_to_label_map_for_sample(
            adata_cells,
            sample_name,
            label_col=label_col,
            target_dtype=clusters.dtype,
        )
        if len(cluster_to_label) == 0:
            print(f"[PyVista] No QC-passed cells for sample '{sample_name}' or mapping failed.")

        cluster_tokens = None
        try:
            cluster_tokens = np.asarray(clusters).reshape(-1) if bool(clusters_are_tokens) else _cluster_array_to_tokens(clusters)
            assigned_token_dtype = cluster_tokens.dtype if getattr(cluster_tokens.dtype, "kind", "") in {"U", "S"} else object
            assigned_tokens = np.asarray(sorted(_normalized_cluster_label_lookup(cluster_to_label).keys()), dtype=assigned_token_dtype)
            assigned_mask = np.isin(cluster_tokens, assigned_tokens)
            n_assigned = int(np.sum(assigned_mask))
        except Exception:
            assigned_mask = None
            n_assigned = 0

        if GSE_RENDER_ONLY_ASSIGNED and assigned_mask is not None and n_assigned > 0:
            original_parent_rows = original_parent_rows[assigned_mask]
            coords = coords[assigned_mask]
            clusters = clusters[assigned_mask]
            if cluster_tokens is not None:
                cluster_tokens = np.asarray(cluster_tokens)[assigned_mask]
            n_pts = int(coords.shape[0])
            assigned_mask = np.ones(n_pts, dtype=bool)
            n_assigned = n_pts
            print(f"[PyVista] Rendering only assigned hubs: {n_pts:,} points")

        eff_n = int(n_assigned) if (n_assigned > 0 and not GSE_RENDER_ONLY_ASSIGNED) else int(n_pts)
        if GSE_AUTO_OPACITY:
            scale = (1_000_000.0 / max(float(eff_n), 1.0)) ** float(GSE_OPACITY_SCALING_EXPONENT)
            op_assigned = float(GSE_ASSIGNED_OPACITY_AT_1M) * float(scale)
            ps = float(max(GSE_POINT_SIZE, 1.0))
            op_assigned *= 1.0 / ps
            op_assigned = float(np.clip(op_assigned, float(GSE_ASSIGNED_OPACITY_MIN), float(GSE_ASSIGNED_OPACITY_MAX)))
            op_unassigned = float(op_assigned) * float(GSE_UNASSIGNED_OPACITY_FRACTION)
            op_unassigned = float(np.clip(op_unassigned, float(GSE_UNASSIGNED_OPACITY_MIN), float(GSE_UNASSIGNED_OPACITY_MAX)))
        else:
            op_assigned = float(GSE_ASSIGNED_OPACITY)
            op_unassigned = float(GSE_UNASSIGNED_OPACITY)

        assigned_alpha_u8 = int(np.clip(int(round(op_assigned * 255.0)), 1, 255))
        unassigned_alpha_u8 = int(np.clip(int(round(op_unassigned * 255.0)), 0, assigned_alpha_u8))
        unassigned_rgba = (211, 211, 211, unassigned_alpha_u8)
        label_to_rgba = {k: _hex_to_rgba_u8(v, alpha=assigned_alpha_u8) for k, v in label_palette_hex.items()}
        print(
            f"[PyVista] Opacity: assigned={op_assigned:.4f} (u8={assigned_alpha_u8}), "
            f"unassigned={op_unassigned:.5f} (u8={unassigned_alpha_u8}), point_size={GSE_POINT_SIZE}"
        )

        n_samp = int(min(GSE_STATS_SAMPLE_N, n_pts))
        if n_samp <= 0:
            print("[PyVista] No points. Skipping.")
            continue

        if (
            bool(GSE_FRAMING_USE_ASSIGNED_POINTS)
            and assigned_mask is not None
            and int(n_assigned) >= int(GSE_FRAMING_MIN_ASSIGNED_POINTS)
        ):
            pool = np.flatnonzero(assigned_mask)
            pool_label = "assigned"
        else:
            pool = np.arange(n_pts, dtype=np.int64)
            pool_label = "all"

        samp_idx = pool if pool.size <= n_samp else rng.choice(pool, size=n_samp, replace=False)
        pts_samp = coords[samp_idx, :].astype(np.float32, copy=False)
        print(f"[PyVista] Camera stats sample: n={pts_samp.shape[0]:,} from {pool_label} hubs")

        center = np.median(pts_samp, axis=0)
        coords = coords - center
        pts_samp = pts_samp - center
        r2 = np.sum(pts_samp ** 2, axis=1)
        q25, q75 = np.quantile(r2, [0.25, 0.75])
        mid_mask = (r2 >= q25) & (r2 <= q75)
        pts_mid = pts_samp[mid_mask]
        if pts_mid.shape[0] < 64:
            pts_mid = pts_samp

        plotter = pv.Plotter(off_screen=True, window_size=GSE_WINDOW_SIZE)
        try:
            if GSE_BACKGROUND_TOP is not None:
                plotter.set_background(GSE_BACKGROUND, top=GSE_BACKGROUND_TOP)
            else:
                plotter.set_background(GSE_BACKGROUND)
        except Exception:
            plotter.set_background(GSE_BACKGROUND)

        _enable_pyvista_antialiasing(plotter, GSE_ANTIALIASING, log_prefix="[PyVista]")
        if GSE_ENABLE_LIGHTKIT and hasattr(plotter, "enable_lightkit"):
            try:
                plotter.enable_lightkit()
            except Exception:
                pass

        if cluster_tokens is not None:
            label_strings_for_sample_raw = _clusters_to_label_strings(
                cluster_tokens,
                cluster_to_label=cluster_to_label,
                unassigned_label="__unassigned__",
                clusters_are_tokens=True,
            )
        else:
            label_strings_for_sample_raw = _clusters_to_label_strings(
                clusters,
                cluster_to_label=cluster_to_label,
                unassigned_label="__unassigned__",
            )
        label_strings_for_sample = _mask_excluded_leiden_scatter_labels(label_strings_for_sample_raw)
        (
            label_strings_for_sample,
            label_codes_for_sample,
            label_values_for_sample,
            label_code_of_label_for_sample,
        ) = _label_code_cache(label_strings_for_sample)
        try:
            excl_code = label_code_of_label_for_sample.get(str(GSE_EXCLUDED_LABEL_SENTINEL))
            n_excluded = int(np.sum(label_codes_for_sample == int(excl_code))) if excl_code is not None else 0
            if n_excluded > 0:
                print(f"[PyVista] Excluding {n_excluded:,} hubs from Leiden scatter labels {sorted(_leiden_scatter_excluded_label_set(), key=_natural_sort_key)}")
        except Exception:
            pass
        if bool(GSE_CLUSTER_BALANCED_RENDER):
            draw_summary = _add_balanced_leiden_point_actors(
                plotter,
                coords,
                label_strings_for_sample,
                label_palette_hex=label_palette_hex,
                rng=rng,
                cluster_tokens=cluster_tokens,
                label_codes=label_codes_for_sample,
                label_values=label_values_for_sample,
                label_code_of_label=label_code_of_label_for_sample,
            )
            fg_idx = np.asarray(draw_summary.get("_foreground_point_indices", np.zeros(0, dtype=np.int64)), dtype=np.int64)
            fg_idx = fg_idx[(fg_idx >= 0) & (fg_idx < int(clusters.shape[0]))]
            fg_parent_rows = np.asarray(original_parent_rows[fg_idx], dtype=np.int64) if fg_idx.size else np.zeros(0, dtype=np.int64)
            try:
                np.save(os.path.join(out_dir, f"{sample_name}_leiden_plotted_parent_rows.npy"), fg_parent_rows)
                pd.DataFrame({"parent_obs_row": fg_parent_rows}).to_csv(
                    os.path.join(out_dir, f"{sample_name}_leiden_plotted_parent_rows.tsv"),
                    sep="\t",
                    index=False,
                )
            except Exception as e:
                print(f"[PyVista] Warning: failed writing Leiden plotted-row cache for {sample_name} ({e}).")
            if draw_summary.get("foreground_cluster_ids"):
                fg_cluster_ids = np.asarray(draw_summary.get("foreground_cluster_ids", []), dtype=object)
            else:
                fg_cluster_ids = _unique_cluster_values_for_filter(clusters[fg_idx]) if fg_idx.size else np.array([], dtype=object)
            public_draw_summary = {k: v for k, v in draw_summary.items() if not str(k).startswith("_")}
            fg_record = {
                "foreground_cluster_ids": [str(x) for x in pd.Series(fg_cluster_ids).astype(str).tolist()],
                "filter_info": {
                    "filter_source": "leiden_pyvista_draw_summary",
                    "foreground_labels": public_draw_summary.get("minor_labels", []),
                    "major_labels": public_draw_summary.get("major_labels", []),
                    "n_foreground_hubs_drawn": int(fg_idx.size),
                    "n_foreground_parent_rows_cache": os.path.join(out_dir, f"{sample_name}_leiden_plotted_parent_rows.npy"),
                    "n_foreground_cells": int(fg_cluster_ids.size),
                },
            }
            GSE_LEIDEN_FOREGROUND_CLUSTER_IDS_BY_SAMPLE[str(sample_name)] = fg_record
            leiden_foreground_by_sample[str(sample_name)] = fg_record
            print("[PyVista] Balanced Leiden render: " + json.dumps(public_draw_summary, sort_keys=True)[:2500])
        else:
            # Respect the same label exclusion contract even if the balanced
            # actor route is disabled.
            label_to_rgba_eff = dict(label_to_rgba)
            for _lab in _leiden_scatter_excluded_label_set():
                label_to_rgba_eff[str(_lab)] = np.array([0, 0, 0, 0], dtype=np.uint8)
            rgba = _clusters_to_rgba(
                clusters,
                cluster_to_label=cluster_to_label,
                label_to_rgba=label_to_rgba_eff,
                unassigned_rgba=(unassigned_rgba[0], unassigned_rgba[1], unassigned_rgba[2], 0),
                cluster_tokens=cluster_tokens,
            )
            try:
                assigned_n = int(np.sum(rgba[:, 3] == assigned_alpha_u8))
                total_n = int(rgba.shape[0])
                print(f"[PyVista] RGBA: assigned={assigned_n:,} ({assigned_n / max(total_n, 1):.1%}), unassigned={total_n - assigned_n:,}")
            except Exception:
                pass
            drawn_mask = np.asarray(rgba[:, 3] > 0, dtype=bool)
            fg_parent_rows = np.asarray(original_parent_rows[drawn_mask], dtype=np.int64) if np.any(drawn_mask) else np.zeros(0, dtype=np.int64)
            try:
                np.save(os.path.join(out_dir, f"{sample_name}_leiden_plotted_parent_rows.npy"), fg_parent_rows)
                pd.DataFrame({"parent_obs_row": fg_parent_rows}).to_csv(
                    os.path.join(out_dir, f"{sample_name}_leiden_plotted_parent_rows.tsv"),
                    sep="\t",
                    index=False,
                )
            except Exception as e:
                print(f"[PyVista] Warning: failed writing Leiden plotted-row cache for {sample_name} ({e}).")
            fg = _unique_cluster_values_for_filter(np.asarray(clusters)[drawn_mask]) if np.any(drawn_mask) else np.array([], dtype=object)
            leiden_foreground_by_sample[str(sample_name)] = {
                "balanced_render": False,
                "filter_source": "leiden_pyvista_rgba_draw_summary",
                "foreground_cluster_ids": [str(x) for x in pd.Series(fg).astype(str).tolist()],
                "n_foreground_cells": int(fg.size),
                "major_labels": [],
                "foreground_labels": [],
            }

            cloud = pv.PolyData(coords)
            cloud.point_data["rgba"] = rgba
            point_size = float(GSE_POINT_SIZE)
            min_ps = 0.8
            if GSE_RENDER_POINTS_AS_SPHERES:
                min_ps = float(max(min_ps, GSE_MIN_POINT_SIZE_FOR_SPHERES))
            if point_size < min_ps:
                print(f"[PyVista] Note: clamping point_size {GSE_POINT_SIZE} -> {min_ps}")
                point_size = min_ps

            actor = plotter.add_mesh(
                cloud,
                scalars="rgba",
                rgba=True,
                render_points_as_spheres=bool(GSE_RENDER_POINTS_AS_SPHERES),
                point_size=point_size,
                lighting=bool(GSE_LIGHTING),
            )
            if GSE_LIGHTING:
                try:
                    prop = actor.GetProperty()
                    prop.SetAmbient(float(GSE_MATERIAL_AMBIENT))
                    prop.SetDiffuse(float(GSE_MATERIAL_DIFFUSE))
                    prop.SetSpecular(float(GSE_MATERIAL_SPECULAR))
                    prop.SetSpecularPower(float(GSE_MATERIAL_SPECULAR_POWER))
                except Exception:
                    pass

        if GSE_ENABLE_EYE_DOME_LIGHTING:
            try:
                plotter.enable_eye_dome_lighting(strength=float(GSE_EDL_STRENGTH), radius=float(GSE_EDL_RADIUS))
            except TypeError:
                try:
                    plotter.enable_eye_dome_lighting()
                except Exception:
                    pass
            except Exception:
                pass

        if GSE_ENABLE_SSAO:
            try:
                plotter.enable_ssao(radius=float(GSE_SSAO_RADIUS), bias=float(GSE_SSAO_BIAS), kernel_size=int(GSE_SSAO_KERNEL_SIZE))
            except TypeError:
                try:
                    plotter.enable_ssao()
                except Exception:
                    pass
            except Exception:
                pass

        plotter.camera.parallel_projection = bool(GSE_CAMERA_PARALLEL_PROJECTION)
        if not plotter.camera.parallel_projection:
            try:
                plotter.camera.view_angle = float(GSE_CAMERA_VIEW_ANGLE_DEG)
            except Exception:
                pass

        focal = np.array([0.0, 0.0, 0.0], dtype=float)
        tail = float(np.clip(0.5 * (1.0 - float(GSE_INFRAME_FRACTION)), 0.0, 0.49))
        try:
            lo = np.quantile(pts_samp, tail, axis=0)
            hi = np.quantile(pts_samp, 1.0 - tail, axis=0)
            diag = float(np.linalg.norm(hi - lo))
        except Exception:
            diag = float(np.linalg.norm(np.ptp(pts_samp, axis=0)))
        radius_guess = 2.0 * diag if diag > 0 else 1.0

        leiden_view_rows: List[Dict[str, object]] = []
        for vi, (az, el) in enumerate(GSE_VIEW_ANGLES_DEG, start=1):
            view_id = _gse_view_id(vi, az, el)
            _set_gse_view_camera(
                plotter,
                az=float(az),
                el=float(el),
                pts_samp=pts_samp,
                pts_mid=pts_mid,
                radius_guess=float(radius_guess),
                focal=focal,
            )


            if GSE_SCALE_BAR_ENABLE:
                try:
                    _add_camera_space_scale_bar(plotter, pts_samp)
                except Exception as e:
                    print(f"[PyVista] Warning: failed to add scale bar ({e})")

            out_png = os.path.join(out_dir, f"{sample_name}_GSE_scatter_{view_id}.png")
            saved_ok = _save_pyvista_screenshot(plotter, out_png, log_prefix="[PyVista]")
            camera_state = _camera_state_for_manifest(plotter)
            leiden_view_rows.append({
                "sample": str(sample_name),
                "view_index": int(vi),
                "view_id": str(view_id),
                "azimuth_deg": float(az),
                "elevation_deg": float(el),
                "png_path": os.path.abspath(out_png),
                "saved": bool(saved_ok),
                "route": "leiden_pyvista",
                "window_width": int(GSE_WINDOW_SIZE[0]),
                "window_height": int(GSE_WINDOW_SIZE[1]),
                "inframe_fraction": float(GSE_INFRAME_FRACTION),
                "near_quantile": float(GSE_INFRAME_NEAR_QUANTILE),
                "radius_guess": float(radius_guess),
                "focal_json": json.dumps([float(x) for x in np.asarray(focal, dtype=float).reshape(-1)[:3]]),
                **camera_state,
            })

        _write_leiden_view_manifest(out_dir, sample_name, leiden_view_rows)

        try:
            plotter.close()
        except Exception:
            pass

        top_row = _render_sample_top_plotted_leiden_snapshots(
            sample_name=sample_name,
            coords=coords,
            label_strings=label_strings_for_sample,
            label_palette_hex=label_palette_hex,
            pts_samp=pts_samp,
            pts_mid=pts_mid,
            radius_guess=float(radius_guess),
            out_dir=out_dir,
            label_codes=label_codes_for_sample,
            label_values=label_values_for_sample,
            label_code_of_label=label_code_of_label_for_sample,
        )
        if top_row is not None:
            top_plotted_render_rows.append(top_row)

    if top_plotted_render_rows:
        top_path = os.path.join(out_dir, str(GSE_SAMPLE_TOP_PLOTTED_SUMMARY_TABLE))
        try:
            pd.DataFrame(top_plotted_render_rows).to_csv(top_path, sep="\t", index=False)
            print(f"[PyVistaTop5] Wrote summary: {top_path}")
        except Exception as e:
            print(f"[PyVistaTop5] Warning: failed writing summary ({e}).")
    gc.collect()
    return leiden_foreground_by_sample




def _cell_surface_template(direction_count: int = GSE_CELL_SURFACE_DIRECTION_COUNT) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Return a reusable spherical direction template and triangulation."""
    cache = getattr(_cell_surface_template, "_cache", {})
    n = int(max(12, direction_count))
    if n in cache:
        return cache[n]

    i = np.arange(n, dtype=np.float64)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (i + 0.5) / float(n)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = golden_angle * i
    dirs = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12

    hull = ConvexHull(dirs)
    faces = np.asarray(hull.simplices, dtype=np.int64)
    for fi, tri in enumerate(faces):
        a, b, c = dirs[tri[0]], dirs[tri[1]], dirs[tri[2]]
        normal = np.cross(b - a, c - a)
        if float(np.dot(normal, (a + b + c) / 3.0)) < 0.0:
            faces[fi, 1], faces[fi, 2] = faces[fi, 2], faces[fi, 1]

    neighbors: List[set] = [set() for _ in range(dirs.shape[0])]
    for tri in faces:
        a, b, c = [int(x) for x in tri]
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    neighbor_arrays = [np.asarray(sorted(x), dtype=np.int64) for x in neighbors]
    out = (dirs.astype(np.float32, copy=False), faces.astype(np.int64, copy=False), neighbor_arrays)
    cache[n] = out
    setattr(_cell_surface_template, "_cache", cache)
    return out


def _cell_surface_config_dict() -> Dict[str, object]:
    """Geometry-affecting settings for cache validation."""
    return {
        "method": str(GSE_CELL_SURFACE_METHOD),
        "require_postfilter_cells": bool(GSE_CELL_SURFACE_REQUIRE_POSTFILTER_CELLS),
        "filter_to_leiden_colored_cells": bool(GSE_CELL_SURFACE_FILTER_TO_LEIDEN_COLORED_CELLS),
        "leiden_filter_label_col": str(GSE_CELL_SURFACE_LEIDEN_FILTER_LABEL_COL),
        "cluster_id_col": str(GSE_CELL_SURFACE_CLUSTER_ID_COL),
        "direction_count": int(GSE_CELL_SURFACE_DIRECTION_COUNT),
        "min_points": int(GSE_CELL_SURFACE_MIN_POINTS),
        "max_points_per_cell": int(GSE_CELL_SURFACE_MAX_POINTS_PER_CELL),
        "candidate_multiplier": int(GSE_CELL_SURFACE_CANDIDATE_MULTIPLIER),
        "boundary_sample_fraction": float(GSE_CELL_SURFACE_BOUNDARY_SAMPLE_FRACTION),
        "alpha_min_points": int(GSE_CELL_SURFACE_ALPHA_MIN_POINTS),
        "alpha_max_points": int(GSE_CELL_SURFACE_ALPHA_MAX_POINTS),
        "alpha_knn_k": int(GSE_CELL_SURFACE_ALPHA_KNN_K),
        "alpha_knn_quantile": float(GSE_CELL_SURFACE_ALPHA_KNN_QUANTILE),
        "alpha_radius_multiplier": float(GSE_CELL_SURFACE_ALPHA_RADIUS_MULTIPLIER),
        "alpha_min_relative_radius": float(GSE_CELL_SURFACE_ALPHA_MIN_RELATIVE_RADIUS),
        "alpha_max_relative_radius": float(GSE_CELL_SURFACE_ALPHA_MAX_RELATIVE_RADIUS),
        "alpha_relaxation_factors": [float(x) for x in GSE_CELL_SURFACE_ALPHA_RELAXATION_FACTORS],
        "alpha_min_boundary_faces": int(GSE_CELL_SURFACE_ALPHA_MIN_BOUNDARY_FACES),
        "alpha_trim_radial_quantile": float(GSE_CELL_SURFACE_ALPHA_TRIM_RADIAL_QUANTILE),
        "alpha_expansion": float(GSE_CELL_SURFACE_ALPHA_EXPANSION),
        "alpha_qhull_options": str(GSE_CELL_SURFACE_ALPHA_QHULL_OPTIONS),
        "hull_min_points": int(GSE_CELL_SURFACE_HULL_MIN_POINTS),
        "hull_support_top_k": int(GSE_CELL_SURFACE_HULL_SUPPORT_TOP_K),
        "hull_max_witness_points": int(GSE_CELL_SURFACE_HULL_MAX_WITNESS_POINTS),
        "hull_radial_witness_fraction": float(GSE_CELL_SURFACE_HULL_RADIAL_WITNESS_FRACTION),
        "hull_trim_radial_quantile": float(GSE_CELL_SURFACE_HULL_TRIM_RADIAL_QUANTILE),
        "hull_expansion": float(GSE_CELL_SURFACE_HULL_EXPANSION),
        "hull_qhull_options": str(GSE_CELL_SURFACE_HULL_QHULL_OPTIONS),
        "radius_quantile": float(GSE_CELL_SURFACE_RADIUS_QUANTILE),
        "radius_expansion": float(GSE_CELL_SURFACE_RADIUS_EXPANSION),
        "angular_neighbor_fraction": float(GSE_CELL_SURFACE_ANGULAR_NEIGHBOR_FRACTION),
        "angular_neighbor_min": int(GSE_CELL_SURFACE_ANGULAR_NEIGHBOR_MIN),
        "tangential_drift_fraction": float(GSE_CELL_SURFACE_TANGENTIAL_DRIFT_FRACTION),
        "max_tangential_drift_fraction": float(GSE_CELL_SURFACE_MAX_TANGENTIAL_DRIFT_FRACTION),
        "smooth_iterations": int(GSE_CELL_SURFACE_SMOOTH_ITERATIONS),
        "axis_floor_fraction": float(GSE_CELL_SURFACE_AXIS_FLOOR_FRACTION),
        "degenerate_radius_fraction": float(GSE_CELL_SURFACE_DEGENERATE_RADIUS_FRACTION),
        "random_seed": int(RANDOM_SEED),
        "deterministic_per_cell_surface_seeds": True,
    }


def _cell_surface_cache_tag() -> str:
    cfg = _cell_surface_config_dict()
    hexp = int(round(float(cfg["hull_expansion"]) * 1000.0))
    aexp = int(round(float(cfg["alpha_expansion"]) * 1000.0))
    amul = int(round(float(cfg["alpha_radius_multiplier"]) * 100.0))
    return (
        f"{str(cfg['method']).replace(os.sep, '_')}_m{cfg['max_points_per_cell']}"
        f"_am{cfg['alpha_max_points']}_ak{cfg['alpha_knn_k']}_amul{amul}_aex{aexp}"
        f"_hw{cfg['hull_max_witness_points']}_hex{hexp}_fb{cfg['direction_count']}"
        f"_pf{int(bool(cfg['require_postfilter_cells']))}_lc{int(bool(cfg['filter_to_leiden_colored_cells']))}"
    )


def _file_fingerprint_for_cache(filepath: str) -> Dict[str, object]:
    try:
        st = os.stat(filepath)
        return {
            "filepath": str(filepath),
            "size_bytes": int(st.st_size),
            "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
        }
    except Exception:
        return {"filepath": str(filepath), "size_bytes": None, "mtime_ns": None}


def _cell_surface_cache_paths(out_dir: str, sample_name: str) -> Tuple[str, str, str]:
    safe_sample = str(sample_name).replace(os.sep, "_").replace(" ", "_")
    geom_dir = os.path.join(str(out_dir), str(GSE_CELL_SURFACE_CACHE_DIR_NAME))
    tag = _cell_surface_cache_tag()
    mesh_path = os.path.join(geom_dir, f"{safe_sample}_infomap_cell_surfaces_{tag}.vtp")
    meta_path = os.path.join(geom_dir, f"{safe_sample}_infomap_cell_surfaces_{tag}.json")
    cell_table_path = os.path.join(geom_dir, f"{safe_sample}_infomap_cell_surfaces_{tag}_cells.tsv")
    return mesh_path, meta_path, cell_table_path


def _load_cached_cell_surface_mesh(
    *,
    mesh_path: str,
    meta_path: str,
    filepath: str,
    postfilter_fingerprint: Optional[Dict[str, object]] = None,
    force_rebuild: bool = False,
) -> Optional["pv.PolyData"]:
    if pv is None or force_rebuild or not bool(GSE_CELL_SURFACE_CACHE_ENABLE):
        return None
    if not (os.path.exists(mesh_path) and os.path.exists(meta_path)):
        return None
    try:
        meta = _read_json_safely(meta_path)
        if meta.get("surface_config") != _cell_surface_config_dict():
            return None
        if meta.get("source_file") != _file_fingerprint_for_cache(filepath):
            return None
        if meta.get("postfilter_cell_fingerprint") != postfilter_fingerprint:
            return None
        mesh = pv.read(mesh_path)
        if int(mesh.n_points) <= 0 or int(mesh.n_cells) <= 0:
            return None
        if "rgba" not in mesh.point_data and "rgba" not in mesh.cell_data:
            return None
        mesh = _ensure_surface_mesh_point_rgba(mesh)
        if "rgba" not in mesh.point_data:
            return None
        print(f"[PyVistaCellSurface] Using cached geometry: {mesh_path}")
        return mesh
    except Exception as e:
        print(f"[PyVistaCellSurface] Cache read failed ({e}); rebuilding geometry.")
        return None


def _save_cached_cell_surface_mesh(
    mesh: "pv.PolyData",
    cell_rows: Sequence[Dict[str, object]],
    *,
    mesh_path: str,
    meta_path: str,
    cell_table_path: str,
    filepath: str,
    sample_name: str,
    postfilter_fingerprint: Optional[Dict[str, object]] = None,
) -> None:
    if pv is None or not bool(GSE_CELL_SURFACE_CACHE_ENABLE) or not bool(GSE_CELL_SURFACE_SAVE_GEOMETRY):
        return
    try:
        os.makedirs(os.path.dirname(mesh_path), exist_ok=True)
        mesh.save(mesh_path)
        _write_json_safely(
            {
                "sample": str(sample_name),
                "source_file": _file_fingerprint_for_cache(filepath),
                "postfilter_cell_fingerprint": postfilter_fingerprint,
                "surface_config": _cell_surface_config_dict(),
                "n_points": int(mesh.n_points),
                "n_faces": int(mesh.n_cells),
                "n_infomap_cells_rendered": int(len(cell_rows)),
            },
            meta_path,
        )
        if cell_rows:
            pd.DataFrame(cell_rows).to_csv(cell_table_path, sep="\t", index=False)
        print(f"[PyVistaCellSurface] Cached geometry: {mesh_path}")
    except Exception as e:
        print(f"[PyVistaCellSurface] Warning: failed to save cached geometry ({e}).")


CELL_SURFACE_METRIC_NUMERIC_COLUMNS: Tuple[str, ...] = (
    "surface_pca_lambda1",
    "surface_pca_lambda2",
    "surface_pca_lambda3",
    "surface_linearity_lambda1_over_lambda2",
    "surface_planarity_lambda2_over_lambda3",
    "surface_area_approx",
    "surface_volume_approx",
    "surface_area_to_volume_ratio_approx",
    "surface_metric_n_points",
)

CELL_SURFACE_METRIC_INT_COLUMNS: Tuple[str, ...] = (
    "n_hubs_total",
    "n_hubs_surface_sampled",
    "n_surface_vertices",
    "n_surface_faces",
)


def _cell_surface_nan_metric_dict() -> Dict[str, float]:
    return {k: float("nan") for k in CELL_SURFACE_METRIC_NUMERIC_COLUMNS}


def _ratio_or_nan(num: float, den: float) -> float:
    try:
        num_f = float(num)
        den_f = float(den)
    except Exception:
        return float("nan")
    if (not np.isfinite(num_f)) or (not np.isfinite(den_f)) or den_f <= 0.0:
        return float("nan")
    return float(num_f / den_f)


def _cell_surface_pca_metric_dict(points: object) -> Dict[str, float]:
    """PCA shape metrics from the finite hub/sampled points used for a surface."""
    out = _cell_surface_nan_metric_dict()
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        out["surface_metric_n_points"] = 0.0
        return out
    pts = pts[np.isfinite(pts).all(axis=1)]
    n = int(pts.shape[0])
    out["surface_metric_n_points"] = float(n)
    if n < 2:
        return out
    X = pts - np.nanmean(pts, axis=0, keepdims=True)
    try:
        cov = (X.T @ X) / max(float(n - 1), 1.0)
        eigvals = np.linalg.eigvalsh(np.asarray(cov, dtype=np.float64))
        eigvals = np.maximum(np.sort(eigvals)[::-1], 0.0)
    except Exception:
        return out
    if eigvals.size >= 3 and np.all(np.isfinite(eigvals[:3])):
        l1, l2, l3 = [float(x) for x in eigvals[:3]]
        out["surface_pca_lambda1"] = l1
        out["surface_pca_lambda2"] = l2
        out["surface_pca_lambda3"] = l3
        out["surface_linearity_lambda1_over_lambda2"] = _ratio_or_nan(l1, l2)
        out["surface_planarity_lambda2_over_lambda3"] = _ratio_or_nan(l2, l3)
    return out


def _triangular_surface_metric_dict(vertices: object, faces: object) -> Dict[str, float]:
    """Approximate area, enclosed volume, and S/V for a triangular surface."""
    out = {
        "surface_area_approx": float("nan"),
        "surface_volume_approx": float("nan"),
        "surface_area_to_volume_ratio_approx": float("nan"),
    }
    try:
        verts = np.asarray(vertices, dtype=np.float64)
        tri = np.asarray(faces, dtype=np.int64)
    except Exception:
        return out
    if verts.ndim != 2 or verts.shape[1] != 3 or tri.ndim != 2 or tri.shape[1] != 3:
        return out
    if verts.shape[0] <= 0 or tri.shape[0] <= 0:
        return out
    ok = np.isfinite(verts).all(axis=1)
    if not np.all(ok):
        # Keep indexing simple: metric is undefined if the surface references non-finite vertices.
        return out
    good_faces = np.all((tri >= 0) & (tri < int(verts.shape[0])), axis=1)
    tri = tri[good_faces]
    if tri.shape[0] <= 0:
        return out
    P = verts[tri]
    cross = np.cross(P[:, 1, :] - P[:, 0, :], P[:, 2, :] - P[:, 0, :])
    area = float(0.5 * np.sum(np.linalg.norm(cross, axis=1)))
    volume = float(abs(np.sum(np.einsum("ij,ij->i", P[:, 0, :], np.cross(P[:, 1, :], P[:, 2, :]))) / 6.0))
    if ((not np.isfinite(volume)) or volume <= 1e-12) and verts.shape[0] >= 4:
        try:
            volume_hull = float(ConvexHull(verts).volume)
            if np.isfinite(volume_hull) and volume_hull > 0.0:
                volume = volume_hull
        except Exception:
            pass
    if np.isfinite(area):
        out["surface_area_approx"] = area
    if np.isfinite(volume) and volume > 0.0:
        out["surface_volume_approx"] = volume
        out["surface_area_to_volume_ratio_approx"] = float(area / volume) if np.isfinite(area) else float("nan")
    return out


def _cell_surface_metric_dict(points: object, vertices: object = None, faces: object = None) -> Dict[str, float]:
    out = _cell_surface_pca_metric_dict(points)
    out.update(_triangular_surface_metric_dict(vertices, faces))
    return out


def _cell_surface_rows_have_metrics(cell_rows: Sequence[Dict[str, object]]) -> bool:
    if not cell_rows:
        return False
    required = {
        "surface_linearity_lambda1_over_lambda2",
        "surface_planarity_lambda2_over_lambda3",
        "surface_area_to_volume_ratio_approx",
    }
    return required.issubset(set(dict(cell_rows[0]).keys()))


def _mesh_surface_metrics_by_cell_index(mesh: Optional["pv.PolyData"]) -> Dict[int, Dict[str, float]]:
    """Recover per-cell mesh area/volume metrics from cached merged PolyData."""
    if mesh is None:
        return {}
    try:
        faces4 = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 4)
        tri = faces4[:, 1:4]
        cell_idx = np.asarray(mesh.cell_data["infomap_cell_index"], dtype=np.int64).reshape(-1)
        points = np.asarray(mesh.points, dtype=np.float64)
    except Exception:
        return {}
    if tri.shape[0] != cell_idx.shape[0]:
        return {}
    out: Dict[int, Dict[str, float]] = {}
    for ci in np.unique(cell_idx):
        mask = cell_idx == int(ci)
        out[int(ci)] = _triangular_surface_metric_dict(points, tri[mask])
    return out


def _backfill_cell_surface_metric_rows(
    *,
    coords: np.ndarray,
    clusters: np.ndarray,
    valid_mask: np.ndarray,
    mesh: Optional["pv.PolyData"],
    cell_rows: Sequence[Dict[str, object]],
    sample_name: str,
) -> List[Dict[str, object]]:
    """Ensure cached surface tables have PCA and surface S/V metrics."""
    valid_idx = np.flatnonzero(np.asarray(valid_mask, dtype=bool))
    if valid_idx.size == 0:
        return [dict(r) for r in cell_rows]
    labels_valid = _cluster_array_to_tokens(clusters[valid_idx])
    inv, unique_labels = pd.factorize(labels_valid, sort=False)
    inv = np.asarray(inv, dtype=np.int64)
    n_cells_total = int(len(unique_labels))
    counts = np.bincount(inv, minlength=n_cells_total).astype(np.int64, copy=False)
    order = np.argsort(inv, kind="mergesort")
    grouped_valid_idx = valid_idx[order]
    offsets = np.zeros(n_cells_total + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)

    existing_by_label: Dict[str, Dict[str, object]] = {}
    for row in cell_rows or []:
        row_d = dict(row)
        label = str(row_d.get("infomap_cell_id", ""))
        if label:
            existing_by_label[label] = row_d

    mesh_metrics = _mesh_surface_metrics_by_cell_index(mesh)
    out_rows: List[Dict[str, object]] = []
    for ci, label_obj in enumerate(unique_labels):
        label = str(label_obj)
        group = grouped_valid_idx[offsets[ci]: offsets[ci + 1]]
        row = dict(existing_by_label.get(label, {}))
        row.setdefault("sample", str(sample_name))
        row.setdefault("infomap_cell_id", label)
        row.setdefault("infomap_cell_index", int(ci))
        row.setdefault("n_hubs_total", int(group.size))
        row.update(_cell_surface_pca_metric_dict(coords[group, :]))
        if int(ci) in mesh_metrics:
            row.update(mesh_metrics[int(ci)])
        else:
            row.update(_triangular_surface_metric_dict(None, None))
        out_rows.append(row)
    return out_rows


def _cell_connectivity_h5ad_path_for_sample(
    sample_name: str,
    *,
    cell_connectivity_h5ad_dir: Optional[str],
    cell_connectivity_suffix: str,
) -> Optional[str]:
    if not cell_connectivity_h5ad_dir:
        return None
    path = os.path.join(str(cell_connectivity_h5ad_dir), f"{sample_name}_{cell_connectivity_suffix}.h5ad")
    return os.path.abspath(path) if os.path.exists(path) else None


def _surface_metrics_safe_scalar_to_str(x: object) -> str:
    try:
        if x is None or pd.isna(x):
            return ""
    except Exception:
        if x is None:
            return ""
    return str(x)


def _surface_metrics_sanitize_dataframe_for_h5ad(df: pd.DataFrame) -> pd.DataFrame:
    """Local fallback matching caller_module.safe_write_h5ad's dataframe handling."""
    out = df.copy()
    out.index = pd.Index([_surface_metrics_safe_scalar_to_str(x) for x in list(out.index)], dtype=object)
    for col in list(out.columns):
        s_col = out[col]
        try:
            if isinstance(s_col.dtype, pd.CategoricalDtype):
                out[col] = pd.Series([_surface_metrics_safe_scalar_to_str(x) for x in s_col.astype(object).values], index=out.index, dtype=object)
                continue
            if pd.api.types.is_string_dtype(s_col) or s_col.dtype == object:
                out[col] = pd.Series([_surface_metrics_safe_scalar_to_str(x) for x in s_col.values], index=out.index, dtype=object)
                continue
            if pd.api.types.is_bool_dtype(s_col):
                if bool(pd.isna(s_col).any()):
                    out[col] = pd.Series([_surface_metrics_safe_scalar_to_str(x) for x in s_col.values], index=out.index, dtype=object)
                else:
                    out[col] = s_col.astype(bool).to_numpy()
                continue
            if pd.api.types.is_integer_dtype(s_col):
                if bool(pd.isna(s_col).any()):
                    out[col] = s_col.astype("float64").to_numpy()
                else:
                    out[col] = s_col.astype("int64").to_numpy()
                continue
            if pd.api.types.is_float_dtype(s_col):
                out[col] = s_col.astype("float64").to_numpy()
                continue
        except Exception:
            pass
        try:
            out[col] = pd.Series([_surface_metrics_safe_scalar_to_str(x) for x in s_col.values], index=out.index, dtype=object)
        except Exception:
            out[col] = out[col].astype(str).astype(object)
    return out


def _surface_metrics_prepare_anndata_for_h5ad(adata_in: ad.AnnData) -> None:
    """Prepare obs/var dtypes before writing a copied H5AD back out."""
    try:
        from caller_module import sanitize_anndata_for_h5ad as _caller_sanitize_anndata_for_h5ad
        _caller_sanitize_anndata_for_h5ad(adata_in)
    except Exception:
        adata_in.obs_names = pd.Index([_surface_metrics_safe_scalar_to_str(x) for x in list(adata_in.obs_names)], dtype=object)
        adata_in.var_names = pd.Index([_surface_metrics_safe_scalar_to_str(x) for x in list(adata_in.var_names)], dtype=object)
        adata_in.obs = _surface_metrics_sanitize_dataframe_for_h5ad(adata_in.obs)
        adata_in.var = _surface_metrics_sanitize_dataframe_for_h5ad(adata_in.var)
    try:
        if hasattr(ad, "settings") and hasattr(ad.settings, "allow_write_nullable_strings"):
            ad.settings.allow_write_nullable_strings = True
    except Exception:
        pass


def _validate_surface_metrics_temp_h5ad(
    tmp_path: str,
    *,
    expected_n_obs: int,
    expected_n_vars: int,
    min_obs_cols: int,
    min_var_cols: int,
    expected_obs_cols: Sequence[str],
    expected_var_cols: Sequence[str],
    expected_uns_keys: Sequence[str],
    expected_obsm_keys: Sequence[str],
    expected_obsp_keys: Sequence[str],
    expected_layer_keys: Sequence[str],
    required_obs_cols: Sequence[str],
) -> None:
    """Read back a candidate H5AD before replacing the real one."""
    chk = ad.read_h5ad(str(tmp_path), backed="r")
    try:
        if int(chk.n_obs) != int(expected_n_obs) or int(chk.n_vars) != int(expected_n_vars):
            raise RuntimeError(
                f"candidate H5AD shape changed from {expected_n_obs} x {expected_n_vars} "
                f"to {chk.n_obs} x {chk.n_vars}"
            )
        if int(min_obs_cols) > 0 and int(chk.obs.shape[1]) < int(min_obs_cols):
            raise RuntimeError(
                f"candidate H5AD lost obs columns: expected at least {min_obs_cols}, "
                f"found {chk.obs.shape[1]}"
            )
        if int(min_var_cols) > 0 and int(chk.var.shape[1]) < int(min_var_cols):
            raise RuntimeError(
                f"candidate H5AD lost var columns: expected at least {min_var_cols}, "
                f"found {chk.var.shape[1]}"
            )

        def _missing(expected: Sequence[str], got: Iterable[str]) -> List[str]:
            got_set = {str(x) for x in got}
            return [str(x) for x in expected if str(x) not in got_set]

        missing_obs = _missing(list(expected_obs_cols) + list(required_obs_cols), chk.obs.columns)
        if missing_obs:
            raise RuntimeError("candidate H5AD is missing obs columns: " + ", ".join(missing_obs))
        missing_var = _missing(expected_var_cols, chk.var.columns)
        if missing_var:
            raise RuntimeError("candidate H5AD is missing var columns: " + ", ".join(missing_var))
        missing_uns = _missing(expected_uns_keys, chk.uns.keys())
        if missing_uns:
            raise RuntimeError("candidate H5AD is missing uns keys: " + ", ".join(missing_uns))
        missing_obsm = _missing(expected_obsm_keys, chk.obsm.keys())
        if missing_obsm:
            raise RuntimeError("candidate H5AD is missing obsm keys: " + ", ".join(missing_obsm))
        missing_obsp = _missing(expected_obsp_keys, chk.obsp.keys())
        if missing_obsp:
            raise RuntimeError("candidate H5AD is missing obsp keys: " + ", ".join(missing_obsp))
        missing_layers = _missing(expected_layer_keys, chk.layers.keys())
        if missing_layers:
            raise RuntimeError("candidate H5AD is missing layer keys: " + ", ".join(missing_layers))
    finally:
        try:
            chk.file.close()
        except Exception:
            pass


def _write_surface_metrics_h5ad_atomically(
    adata_conn: ad.AnnData,
    h5ad_path: str,
    *,
    required_obs_cols: Sequence[str],
) -> str:
    """Write to a temp file, validate, then atomically replace the target."""
    path = os.path.abspath(str(h5ad_path))
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    expected_n_obs = int(adata_conn.n_obs)
    expected_n_vars = int(adata_conn.n_vars)
    min_obs_cols = int(adata_conn.obs.shape[1])
    min_var_cols = int(adata_conn.var.shape[1])
    expected_obs_cols = [str(x) for x in adata_conn.obs.columns]
    expected_var_cols = [str(x) for x in adata_conn.var.columns]
    expected_uns_keys = [str(x) for x in adata_conn.uns.keys()]
    expected_obsm_keys = [str(x) for x in adata_conn.obsm.keys()]
    expected_obsp_keys = [str(x) for x in adata_conn.obsp.keys()]
    expected_layer_keys = [str(x) for x in adata_conn.layers.keys()]
    tmp_path = os.path.join(out_dir, f".{os.path.basename(path)}.surface_metrics_tmp_{os.getpid()}.h5ad")
    backup_path = path + ".pre_surface_metrics.bak.h5ad"

    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    if os.path.exists(path) and not os.path.exists(backup_path):
        shutil.copy2(path, backup_path)

    _surface_metrics_prepare_anndata_for_h5ad(adata_conn)
    adata_conn.write_h5ad(tmp_path)
    _validate_surface_metrics_temp_h5ad(
        tmp_path,
        expected_n_obs=expected_n_obs,
        expected_n_vars=expected_n_vars,
        min_obs_cols=min_obs_cols,
        min_var_cols=min_var_cols,
        expected_obs_cols=expected_obs_cols,
        expected_var_cols=expected_var_cols,
        expected_uns_keys=expected_uns_keys,
        expected_obsm_keys=expected_obsm_keys,
        expected_obsp_keys=expected_obsp_keys,
        expected_layer_keys=expected_layer_keys,
        required_obs_cols=required_obs_cols,
    )
    os.replace(tmp_path, path)
    return backup_path


def _augment_cell_connectivity_h5ad_with_surface_metrics(
    *,
    h5ad_path: Optional[str],
    cell_rows: Sequence[Dict[str, object]],
    sample_name: str,
    geometry_cache_path: str,
) -> None:
    """Attach per-cell surface metrics to the sample_cell_connectivity_h5ad obs table."""
    if not h5ad_path or not os.path.exists(str(h5ad_path)):
        print(f"[PyVistaCellSurface] sample_cell_connectivity_h5ad not found for {sample_name}; surface metrics not attached.")
        return
    if not cell_rows:
        print(f"[PyVistaCellSurface] No cell surface rows available for {sample_name}; surface metrics not attached to H5AD.")
        return
    df = pd.DataFrame(cell_rows).copy()
    if "infomap_cell_id" not in df.columns:
        print(f"[PyVistaCellSurface] Surface metric table lacks infomap_cell_id for {sample_name}; not attaching to H5AD.")
        return
    df["infomap_cell_id"] = df["infomap_cell_id"].astype(str)
    df = df.drop_duplicates("infomap_cell_id", keep="last").set_index("infomap_cell_id", drop=False)
    try:
        adata_conn = ad.read_h5ad(str(h5ad_path))
    except Exception as e:
        print(f"[PyVistaCellSurface] Could not read {h5ad_path}; surface metrics not attached ({e}).")
        return

    # This mode is specifically for ann12 sample_cell_connectivity_h5ad objects.
    # Refuse to write if the target already looks stripped/corrupt; otherwise a
    # fallback to integer obs_names would silently fail to align cell labels.
    if "cell_node_label" not in adata_conn.obs.columns:
        raise RuntimeError(
            f"Refusing to attach surface metrics to {h5ad_path}: obs['cell_node_label'] is missing. "
            "Restore/regenerate the sample_cell_connectivity_h5ad first; the file may already be stripped."
        )
    if int(adata_conn.obs.shape[1]) == 0 or int(adata_conn.var.shape[1]) == 0:
        raise RuntimeError(
            f"Refusing to attach surface metrics to {h5ad_path}: obs/var metadata are empty. "
            "Restore/regenerate the sample_cell_connectivity_h5ad first."
        )

    obs_labels = adata_conn.obs["cell_node_label"].astype(str)
    attached_cols: List[str] = []
    for col in CELL_SURFACE_METRIC_NUMERIC_COLUMNS + CELL_SURFACE_METRIC_INT_COLUMNS:
        if col not in df.columns:
            continue
        vals = obs_labels.map(df[col])
        adata_conn.obs[col] = pd.to_numeric(vals, errors="coerce").astype(np.float64)
        attached_cols.append(str(col))
    # Keep the method as a string audit column; all requested quantitative fields above stay numeric.
    if "surface_method" in df.columns:
        adata_conn.obs["surface_method"] = obs_labels.map(df["surface_method"]).fillna("").astype(str).values
        attached_cols.append("surface_method")
    adata_conn.obs["surface_metrics_available"] = np.isfinite(
        pd.to_numeric(adata_conn.obs.get("surface_area_to_volume_ratio_approx", np.nan), errors="coerce").to_numpy(dtype=float)
    )
    attached_cols.append("surface_metrics_available")

    # JSON strings avoid adding new object arrays into .uns.
    adata_conn.uns["cell_surface_metrics_source"] = str(geometry_cache_path)
    adata_conn.uns["cell_surface_metrics_sample"] = str(sample_name)
    adata_conn.uns["cell_surface_metrics_columns_json"] = json.dumps([str(x) for x in attached_cols], sort_keys=False)
    adata_conn.uns["cell_surface_metrics_semantics"] = (
        "surface_pca_lambda1/2/3 are descending covariance eigenvalues from the finite hub/sample points used to build "
        "each rendered cell surface. surface_linearity_lambda1_over_lambda2=lambda1/lambda2 and "
        "surface_planarity_lambda2_over_lambda3=lambda2/lambda3. surface_area_approx, surface_volume_approx, and "
        "surface_area_to_volume_ratio_approx are computed from the rendered triangular cell surface; volume falls back to "
        "ConvexHull(vertices).volume only when signed mesh volume is degenerate."
    )
    try:
        backup_path = _write_surface_metrics_h5ad_atomically(
            adata_conn,
            str(h5ad_path),
            required_obs_cols=attached_cols,
        )
        print(
            f"[PyVistaCellSurface] Attached surface metrics to {h5ad_path}: {len(attached_cols)} columns. "
            f"Backup: {backup_path}"
        )
    except Exception as e:
        print(f"[PyVistaCellSurface] Failed writing surface metrics to {h5ad_path}; original file was not replaced ({e})")
        raise

def _sample_points_for_cell_surface(coords: np.ndarray, group_idx: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """Boundary-aware per-cell sampling for surface construction."""
    group_idx = np.asarray(group_idx, dtype=np.int64)
    n = int(group_idx.size)
    if n <= 0:
        return np.empty((0, 3), dtype=np.float32)

    max_points = int(GSE_CELL_SURFACE_MAX_POINTS_PER_CELL)
    if max_points <= 0 or n <= max_points:
        pts = np.asarray(coords[group_idx, :], dtype=np.float32)
        return pts[np.all(np.isfinite(pts), axis=1)]

    cand_n = int(min(n, max(max_points, max_points * int(max(1, GSE_CELL_SURFACE_CANDIDATE_MULTIPLIER)))))
    cand_idx = rng.choice(group_idx, size=cand_n, replace=False).astype(np.int64, copy=False)
    cand = np.asarray(coords[cand_idx, :], dtype=np.float32)
    cand = cand[np.all(np.isfinite(cand), axis=1)]
    if cand.shape[0] <= max_points:
        return cand

    center = np.median(cand, axis=0)
    d2 = np.sum((cand - center[None, :]) ** 2, axis=1)
    boundary_frac = float(np.clip(float(GSE_CELL_SURFACE_BOUNDARY_SAMPLE_FRACTION), 0.0, 1.0))
    boundary_n = int(np.clip(int(round(boundary_frac * max_points)), 1, max_points))
    boundary_n = int(min(boundary_n, cand.shape[0]))
    boundary_loc = np.argpartition(-d2, boundary_n - 1)[:boundary_n]
    rest_n = int(max_points - boundary_n)
    if rest_n <= 0:
        return cand[np.sort(boundary_loc)].astype(np.float32, copy=False)
    keep = np.ones(cand.shape[0], dtype=bool)
    keep[boundary_loc] = False
    rest_pool = np.flatnonzero(keep)
    if rest_pool.size <= rest_n:
        chosen = np.concatenate([boundary_loc, rest_pool])
    else:
        chosen = np.concatenate([boundary_loc, rng.choice(rest_pool, size=rest_n, replace=False).astype(np.int64, copy=False)])
    return cand[np.sort(chosen)].astype(np.float32, copy=False)



def _pca_normalized_points_for_surface(
    pts: np.ndarray,
    *,
    global_radius_floor: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Return center, PCA axes/scales, normalized points, and status."""
    pts = np.asarray(pts, dtype=np.float32)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    n = int(pts.shape[0])
    base_floor = float(max(float(global_radius_floor), 1e-6))
    if n <= 0:
        return np.zeros(3), np.eye(3), np.full(3, base_floor), np.empty((0, 3)), "no_points"
    center = np.median(pts.astype(np.float64, copy=False), axis=0)
    X = pts.astype(np.float64, copy=False) - center[None, :]
    if n < 2:
        return center, np.eye(3), np.full(3, base_floor), X / base_floor, "degenerate_single_point"
    cov = (X.T @ X) / max(float(n), 1.0)
    cov = np.asarray(cov, dtype=np.float64)
    cov[~np.isfinite(cov)] = 0.0
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except Exception:
        eigvals = np.array([base_floor ** 2, base_floor ** 2, base_floor ** 2], dtype=np.float64)
        eigvecs = np.eye(3, dtype=np.float64)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    largest = float(max(np.sqrt(float(eigvals[0])) if eigvals.size else 0.0, base_floor))
    axis_floor = float(max(base_floor, largest * float(GSE_CELL_SURFACE_AXIS_FLOOR_FRACTION)))
    scales = np.sqrt(np.maximum(eigvals, axis_floor ** 2))
    scales = np.where(np.isfinite(scales) & (scales > 0.0), scales, axis_floor)
    Y = (X @ eigvecs) / scales[None, :]
    Y = Y[np.all(np.isfinite(Y), axis=1)]
    return center, eigvecs, scales, Y, "ok"


def _stable_unique_rows_preserve_first(points: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """Return stable row indices after coarse coordinate quantization."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] <= 1:
        return np.arange(pts.shape[0], dtype=np.int64)
    scale = float(np.nanmax(np.ptp(pts, axis=0))) if pts.size else 1.0
    step = max(float(eps), scale * float(eps), 1e-12)
    q = np.round(pts / step).astype(np.int64, copy=False)
    _, idx = np.unique(q, axis=0, return_index=True)
    return np.sort(idx.astype(np.int64, copy=False))


def _farthest_shape_subset(points: np.ndarray, max_points: int) -> np.ndarray:
    """Coverage-preserving cap for hull witness points."""
    pts = np.asarray(points, dtype=np.float64)
    n = int(pts.shape[0])
    max_points = int(max_points)
    if max_points <= 0 or n <= max_points:
        return pts
    radial = np.linalg.norm(pts, axis=1)
    seed: List[int] = [int(np.nanargmax(radial))]
    for j in range(min(3, pts.shape[1])):
        seed.append(int(np.nanargmax(pts[:, j])))
        seed.append(int(np.nanargmin(pts[:, j])))
    selected: List[int] = []
    seen = set()
    for i in seed:
        if i not in seen:
            selected.append(i)
            seen.add(i)
        if len(selected) >= max_points:
            return pts[np.asarray(selected, dtype=np.int64)]
    sel_arr = np.asarray(selected, dtype=np.int64)
    d2 = np.sum((pts[:, None, :] - pts[sel_arr][None, :, :]) ** 2, axis=2)
    min_d2 = np.min(d2, axis=1)
    min_d2[sel_arr] = -np.inf
    radial_scale = float(np.nanmedian(radial[radial > 0])) if np.any(radial > 0) else 1.0
    while len(selected) < max_points:
        score = min_d2 + 0.05 * (radial / max(radial_scale, 1e-12))
        j = int(np.nanargmax(score))
        if (not np.isfinite(score[j])) or j in seen:
            break
        selected.append(j)
        seen.add(j)
        d2_new = np.sum((pts - pts[j][None, :]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2_new)
        min_d2[np.asarray(selected, dtype=np.int64)] = -np.inf
    return pts[np.asarray(selected, dtype=np.int64)]


def _orient_triangles_outward(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Orient triangular faces consistently away from the mesh centroid."""
    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(faces, dtype=np.int64).copy()
    if verts.ndim != 2 or verts.shape[1] != 3 or tris.ndim != 2 or tris.shape[1] != 3:
        return tris
    c0 = np.nanmean(verts, axis=0)
    for i in range(tris.shape[0]):
        a, b, c = [int(x) for x in tris[i]]
        va, vb, vc = verts[a], verts[b], verts[c]
        normal = np.cross(vb - va, vc - va)
        tri_c = (va + vb + vc) / 3.0
        if float(np.dot(normal, tri_c - c0)) < 0.0:
            tris[i, 1], tris[i, 2] = tris[i, 2], tris[i, 1]
    return tris


def _build_support_hull_surface_for_points(
    pts: np.ndarray,
    *,
    directions: np.ndarray,
    global_radius_floor: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str, int]:
    """Build a PCA-normalized support-hull surface from per-cell witnesses."""
    pts = np.asarray(pts, dtype=np.float32)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    n = int(pts.shape[0])
    if n < int(GSE_CELL_SURFACE_HULL_MIN_POINTS):
        return None, None, "below_hull_min_points", n

    center, eigvecs, scales, Y, status = _pca_normalized_points_for_surface(
        pts,
        global_radius_floor=global_radius_floor,
    )
    if status != "ok" or Y.shape[0] < int(GSE_CELL_SURFACE_HULL_MIN_POINTS):
        return None, None, f"hull_pca_{status}", n

    dist = np.linalg.norm(Y, axis=1)
    good = np.isfinite(dist) & (dist > 1e-9)
    Y = Y[good]
    dist = dist[good]
    if Y.shape[0] < int(GSE_CELL_SURFACE_HULL_MIN_POINTS):
        return None, None, "hull_not_enough_nonzero_points", n

    trim_q = float(np.clip(float(GSE_CELL_SURFACE_HULL_TRIM_RADIAL_QUANTILE), 0.50, 1.0))
    if trim_q < 0.999999 and Y.shape[0] > int(GSE_CELL_SURFACE_HULL_MIN_POINTS) + 4:
        lim = float(np.quantile(dist, trim_q))
        keep = dist <= lim
        if int(np.sum(keep)) >= int(GSE_CELL_SURFACE_HULL_MIN_POINTS):
            Y = Y[keep]
            dist = dist[keep]

    dirs = np.asarray(directions, dtype=np.float64)
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
    m = int(Y.shape[0])
    top_k = int(np.clip(int(GSE_CELL_SURFACE_HULL_SUPPORT_TOP_K), 1, max(1, m)))
    scores = Y @ dirs.T
    if top_k >= m:
        directional_idx = np.arange(m, dtype=np.int64)
    else:
        directional_idx = np.argpartition(-scores, top_k - 1, axis=0)[:top_k, :].reshape(-1).astype(np.int64, copy=False)

    max_witness = int(max(8, GSE_CELL_SURFACE_HULL_MAX_WITNESS_POINTS))
    radial_n = int(round(float(GSE_CELL_SURFACE_HULL_RADIAL_WITNESS_FRACTION) * float(max_witness)))
    radial_n = int(np.clip(radial_n, 4, min(m, max_witness)))
    if radial_n >= m:
        radial_idx = np.arange(m, dtype=np.int64)
    else:
        radial_idx = np.argpartition(-dist, radial_n - 1)[:radial_n].astype(np.int64, copy=False)

    axis_idx: List[int] = []
    for j in range(3):
        axis_idx.append(int(np.nanargmax(Y[:, j])))
        axis_idx.append(int(np.nanargmin(Y[:, j])))

    witness_idx = np.unique(np.concatenate([
        np.asarray(directional_idx, dtype=np.int64),
        np.asarray(radial_idx, dtype=np.int64),
        np.asarray(axis_idx, dtype=np.int64),
    ]))
    W = np.asarray(Y[witness_idx], dtype=np.float64)
    W = W[_stable_unique_rows_preserve_first(W, eps=1e-6)]
    if W.shape[0] > max_witness:
        W = _farthest_shape_subset(W, max_witness)
    if W.shape[0] < 4:
        return None, None, "hull_too_few_unique_witnesses", n

    expansion = float(max(1.0, GSE_CELL_SURFACE_HULL_EXPANSION))
    W = W * expansion
    try:
        hull = ConvexHull(W, qhull_options=str(GSE_CELL_SURFACE_HULL_QHULL_OPTIONS))
    except Exception as e:
        try:
            hull = ConvexHull(W, qhull_options="QJ Pp")
        except Exception:
            return None, None, f"hull_failed:{type(e).__name__}", n

    faces = np.asarray(hull.simplices, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[0] <= 0 or faces.shape[1] != 3:
        return None, None, "hull_no_triangles", n
    used = np.unique(faces.reshape(-1)).astype(np.int64, copy=False)
    remap = np.full(W.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size, dtype=np.int64)
    faces = remap[faces]
    W_used = W[used]
    faces = _orient_triangles_outward(W_used, faces)
    verts = center[None, :] + (W_used * scales[None, :]) @ eigvecs.T
    verts = verts.astype(np.float32, copy=False)
    if not np.all(np.isfinite(verts)) or faces.shape[0] <= 0:
        return None, None, "hull_nonfinite_vertices", n
    return verts, faces.astype(np.int64, copy=False), "pca_support_hull", n



def _alpha_shape_radius_from_knn(points: np.ndarray) -> float:
    """Infer an alpha radius from local spacing in normalized coordinates."""
    W = np.asarray(points, dtype=np.float64)
    n = int(W.shape[0])
    if n < 2:
        return 0.0
    diag = float(np.linalg.norm(np.ptp(W, axis=0)))
    diag = max(diag, 1e-6)
    k = int(np.clip(int(GSE_CELL_SURFACE_ALPHA_KNN_K), 1, max(1, n - 1)))
    try:
        tree = cKDTree(W)
        d = tree.query(W, k=k + 1)[0]
        kth = np.asarray(d[:, -1], dtype=np.float64)
        kth = kth[np.isfinite(kth) & (kth > 0)]
        if kth.size > 0:
            q = float(np.clip(float(GSE_CELL_SURFACE_ALPHA_KNN_QUANTILE), 0.05, 0.95))
            base = float(np.quantile(kth, q)) * float(GSE_CELL_SURFACE_ALPHA_RADIUS_MULTIPLIER)
        else:
            base = 0.0
    except Exception:
        base = 0.0
    if (not np.isfinite(base)) or base <= 0.0:
        base = 0.16 * diag
    lo = float(max(1e-6, diag * float(GSE_CELL_SURFACE_ALPHA_MIN_RELATIVE_RADIUS)))
    hi = float(max(lo, diag * float(GSE_CELL_SURFACE_ALPHA_MAX_RELATIVE_RADIUS)))
    return float(np.clip(base, lo, hi))


def _tetra_circumradii(points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    """Compute circumsphere radii for tetrahedra; singular tets become inf."""
    W = np.asarray(points, dtype=np.float64)
    tets = np.asarray(simplices, dtype=np.int64)
    out = np.full(int(tets.shape[0]), np.inf, dtype=np.float64)
    if tets.ndim != 2 or tets.shape[1] != 4 or tets.shape[0] == 0:
        return out
    P = W[tets]
    A = 2.0 * (P[:, 1:, :] - P[:, :1, :])
    b = np.sum((P[:, 1:, :] - P[:, :1, :]) ** 2, axis=2)
    try:
        sol = np.linalg.solve(A, b)
        r = np.linalg.norm(sol, axis=1)
        out[np.isfinite(r)] = r[np.isfinite(r)]
        return out
    except Exception:
        pass
    for i in range(tets.shape[0]):
        try:
            sol_i = np.linalg.solve(A[i], b[i])
            r_i = float(np.linalg.norm(sol_i))
            if np.isfinite(r_i):
                out[i] = r_i
        except Exception:
            continue
    return out


def _boundary_faces_from_tetrahedra(tets: np.ndarray) -> np.ndarray:
    """Return boundary triangular faces from kept tetrahedra."""
    tets = np.asarray(tets, dtype=np.int64)
    if tets.ndim != 2 or tets.shape[1] != 4 or tets.shape[0] == 0:
        return np.empty((0, 3), dtype=np.int64)
    faces = np.vstack([
        tets[:, [0, 1, 2]],
        tets[:, [0, 3, 1]],
        tets[:, [0, 2, 3]],
        tets[:, [1, 3, 2]],
    ]).astype(np.int64, copy=False)
    faces_sorted = np.sort(faces, axis=1)
    uniq, counts = np.unique(faces_sorted, axis=0, return_counts=True)
    boundary = uniq[counts == 1]
    return np.asarray(boundary, dtype=np.int64)


def _build_adaptive_alpha_surface_for_points(
    pts: np.ndarray,
    *,
    global_radius_floor: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str, int]:
    """Build a local, concavity-preserving PCA-normalized alpha surface."""
    pts = np.asarray(pts, dtype=np.float32)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    n = int(pts.shape[0])
    if n < int(GSE_CELL_SURFACE_ALPHA_MIN_POINTS):
        return None, None, "below_alpha_min_points", n

    center, eigvecs, scales, Y, status = _pca_normalized_points_for_surface(
        pts,
        global_radius_floor=global_radius_floor,
    )
    if status != "ok" or Y.shape[0] < int(GSE_CELL_SURFACE_ALPHA_MIN_POINTS):
        return None, None, f"alpha_pca_{status}", n

    dist = np.linalg.norm(Y, axis=1)
    good = np.isfinite(dist) & (dist > 1e-9)
    Y = Y[good]
    dist = dist[good]
    if Y.shape[0] < int(GSE_CELL_SURFACE_ALPHA_MIN_POINTS):
        return None, None, "alpha_not_enough_nonzero_points", n

    trim_q = float(np.clip(float(GSE_CELL_SURFACE_ALPHA_TRIM_RADIAL_QUANTILE), 0.50, 1.0))
    if trim_q < 0.999999 and Y.shape[0] > int(GSE_CELL_SURFACE_ALPHA_MIN_POINTS) + 8:
        lim = float(np.quantile(dist, trim_q))
        keep = dist <= lim
        if int(np.sum(keep)) >= int(GSE_CELL_SURFACE_ALPHA_MIN_POINTS):
            Y = Y[keep]
            dist = dist[keep]

    max_alpha_pts = int(max(16, GSE_CELL_SURFACE_ALPHA_MAX_POINTS))
    if Y.shape[0] > max_alpha_pts:
        # Coverage-preserving cap retains lobes and boundary structure rather than
        # taking a purely random subset.
        Y = _farthest_shape_subset(Y, max_alpha_pts)
        dist = np.linalg.norm(Y, axis=1)

    Y = Y[_stable_unique_rows_preserve_first(Y, eps=1e-7)]
    if Y.shape[0] < int(GSE_CELL_SURFACE_ALPHA_MIN_POINTS):
        return None, None, "alpha_too_few_unique_points", n

    try:
        delaunay = Delaunay(Y, qhull_options=str(GSE_CELL_SURFACE_ALPHA_QHULL_OPTIONS))
    except Exception as e:
        try:
            delaunay = Delaunay(Y, qhull_options="QJ Qbb Qc Q12")
        except Exception:
            return None, None, f"alpha_delaunay_failed:{type(e).__name__}", n

    tets = np.asarray(delaunay.simplices, dtype=np.int64)
    if tets.ndim == 2 and tets.shape[1] == 4:
        # Some Qhull option combinations can include the point-at-infinity
        # sentinel; discard those tetrahedra before indexing Y.
        tets = tets[np.all((tets >= 0) & (tets < Y.shape[0]), axis=1)]
    if tets.ndim != 2 or tets.shape[0] <= 0 or tets.shape[1] != 4:
        return None, None, "alpha_no_tetrahedra", n
    radii = _tetra_circumradii(Y, tets)
    finite = np.isfinite(radii) & (radii > 0)
    if not np.any(finite):
        return None, None, "alpha_no_finite_tetra_radii", n

    base_alpha = _alpha_shape_radius_from_knn(Y)
    relax = [float(x) for x in GSE_CELL_SURFACE_ALPHA_RELAXATION_FACTORS]
    if not relax:
        relax = [1.0]
    min_faces = int(max(4, GSE_CELL_SURFACE_ALPHA_MIN_BOUNDARY_FACES))
    best_faces: Optional[np.ndarray] = None
    best_alpha = float(base_alpha)
    best_kept = 0
    for factor in relax:
        alpha = float(base_alpha) * float(max(factor, 1e-6))
        keep_tets = finite & (radii <= alpha)
        n_kept = int(np.sum(keep_tets))
        if n_kept <= 0:
            continue
        faces = _boundary_faces_from_tetrahedra(tets[keep_tets])
        if faces.shape[0] > (0 if best_faces is None else best_faces.shape[0]):
            best_faces = faces
            best_alpha = alpha
            best_kept = n_kept
        if faces.shape[0] >= min_faces:
            best_faces = faces
            best_alpha = alpha
            best_kept = n_kept
            break

    if best_faces is None or best_faces.shape[0] < 4:
        return None, None, "alpha_no_boundary_faces", n

    faces = np.asarray(best_faces, dtype=np.int64)
    used = np.unique(faces.reshape(-1)).astype(np.int64, copy=False)
    remap = np.full(Y.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size, dtype=np.int64)
    faces = remap[faces]
    W_used = Y[used] * float(max(1.0, GSE_CELL_SURFACE_ALPHA_EXPANSION))
    faces = _orient_triangles_outward(W_used, faces)
    verts = center[None, :] + (W_used * scales[None, :]) @ eigvecs.T
    verts = verts.astype(np.float32, copy=False)
    if not np.all(np.isfinite(verts)) or faces.shape[0] <= 0:
        return None, None, "alpha_nonfinite_vertices", n
    return (
        verts,
        faces.astype(np.int64, copy=False),
        f"pca_adaptive_alpha_surface:a={best_alpha:.4g}:tets={best_kept}",
        n,
    )


def _build_cell_surface_geometry_for_points(
    pts: np.ndarray,
    *,
    directions: np.ndarray,
    tri_faces: np.ndarray,
    neighbor_arrays: Sequence[np.ndarray],
    global_radius_floor: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str, int]:
    """Build the configured per-cell surface, with robust fallbacks."""
    method = str(GSE_CELL_SURFACE_METHOD).strip().lower()
    last_status = "not_attempted"

    if method in {"adaptive_alpha_surface", "alpha", "alpha_surface", "pca_alpha_surface"}:
        verts_a, faces_a, status_a, n_a = _build_adaptive_alpha_surface_for_points(
            pts,
            global_radius_floor=global_radius_floor,
        )
        last_status = status_a
        if verts_a is not None and faces_a is not None and faces_a.shape[0] > 0:
            return verts_a, faces_a, status_a, n_a

    if method in {"adaptive_alpha_surface", "alpha", "alpha_surface", "pca_alpha_surface", "support_hull", "hull", "pca_support_hull"}:
        verts, faces, status, n_used = _build_support_hull_surface_for_points(
            pts,
            directions=directions,
            global_radius_floor=global_radius_floor,
        )
        last_status = f"{last_status}|support:{status}"
        if verts is not None and faces is not None and faces.shape[0] > 0:
            prefix = "support_hull_fallback" if method in {"adaptive_alpha_surface", "alpha", "alpha_surface", "pca_alpha_surface"} else "pca_support_hull"
            return verts, faces, f"{prefix}:{status}", n_used

    verts_r, radial_status, n_r = _build_radial_witness_shell_for_points(
        pts,
        directions=directions,
        neighbor_arrays=neighbor_arrays,
        global_radius_floor=global_radius_floor,
    )
    if verts_r is None:
        return None, None, f"{last_status}|radial:{radial_status}", n_r
    return verts_r, np.asarray(tri_faces, dtype=np.int64), f"radial_fallback:{radial_status}", n_r


def _build_radial_witness_shell_for_points(
    pts: np.ndarray,
    *,
    directions: np.ndarray,
    neighbor_arrays: Sequence[np.ndarray],
    global_radius_floor: float,
) -> Tuple[Optional[np.ndarray], str, int]:
    """Approximate one cell's scatter by a non-ellipsoid PCA-radial shell."""
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return None, "invalid_points", 0
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    n = int(pts.shape[0])
    if n < int(GSE_CELL_SURFACE_MIN_POINTS):
        return None, "below_min_points", n

    dirs = np.asarray(directions, dtype=np.float64)
    nd = int(dirs.shape[0])
    center = np.median(pts.astype(np.float64, copy=False), axis=0)
    X = pts.astype(np.float64, copy=False) - center[None, :]
    radial_raw = np.linalg.norm(X, axis=1)
    base_floor = float(max(float(global_radius_floor), 1e-6))

    if n < 2 or (not np.any(radial_raw > base_floor * 1e-3)):
        eigvecs = np.eye(3, dtype=np.float64)
        scales = np.full(3, base_floor, dtype=np.float64)
        radii = np.ones(nd, dtype=np.float64)
        tangent = np.zeros((nd, 3), dtype=np.float64)
        status = "degenerate_small_shell"
    else:
        cov = (X.T @ X) / max(float(n), 1.0)
        cov = np.asarray(cov, dtype=np.float64)
        cov[~np.isfinite(cov)] = 0.0
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
        except Exception:
            eigvals = np.array([base_floor ** 2, base_floor ** 2, base_floor ** 2], dtype=np.float64)
            eigvecs = np.eye(3, dtype=np.float64)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order]
        largest = float(max(np.sqrt(float(eigvals[0])) if eigvals.size else 0.0, base_floor))
        axis_floor = float(max(base_floor, largest * float(GSE_CELL_SURFACE_AXIS_FLOOR_FRACTION)))
        scales = np.sqrt(np.maximum(eigvals, axis_floor ** 2))
        scales = np.where(np.isfinite(scales) & (scales > 0.0), scales, axis_floor)

        Y = (X @ eigvecs) / scales[None, :]
        dist = np.linalg.norm(Y, axis=1)
        good = np.isfinite(dist) & (dist > 1e-9)
        if not np.any(good):
            radii = np.ones(nd, dtype=np.float64)
            tangent = np.zeros((nd, 3), dtype=np.float64)
            status = "degenerate_small_shell"
        else:
            Y = Y[good]
            dist = dist[good]
            unit = Y / (dist[:, None] + 1e-12)
            sim = dirs @ unit.T
            m = int(dist.size)
            top_k = int(np.ceil(float(GSE_CELL_SURFACE_ANGULAR_NEIGHBOR_FRACTION) * float(m)))
            top_k = int(np.clip(max(top_k, int(GSE_CELL_SURFACE_ANGULAR_NEIGHBOR_MIN)), 1, m))
            q = float(np.clip(float(GSE_CELL_SURFACE_RADIUS_QUANTILE), 0.50, 1.0))
            if top_k >= m:
                r0 = float(np.quantile(dist, q))
                radii = np.full(nd, r0, dtype=np.float64)
                top_idx = np.tile(np.arange(m, dtype=np.int64)[None, :], (nd, 1))
            else:
                top_idx = np.argpartition(-sim, top_k - 1, axis=1)[:, :top_k]
                radii = np.quantile(dist[top_idx], q, axis=1).astype(np.float64, copy=False)

            expansion = float(max(1.0, GSE_CELL_SURFACE_RADIUS_EXPANSION))
            nearest = np.argmax(sim, axis=0).astype(np.int64, copy=False)
            support = dist * expansion
            np.maximum.at(radii, nearest, support)
            r_floor = float(max(0.20, 0.50 * np.quantile(dist, 0.10))) if dist.size else 0.20
            radii = np.maximum(radii * expansion, r_floor)

            drift_fraction = float(max(0.0, GSE_CELL_SURFACE_TANGENTIAL_DRIFT_FRACTION))
            if drift_fraction > 0.0 and m >= 6:
                witness = Y[top_idx].mean(axis=1)
                axial = np.sum(witness * dirs, axis=1)
                tangent = witness - axial[:, None] * dirs
                tangent_norm = np.linalg.norm(tangent, axis=1)
                max_tangent = float(max(0.0, GSE_CELL_SURFACE_MAX_TANGENTIAL_DRIFT_FRACTION)) * radii
                scale = np.divide(max_tangent, tangent_norm + 1e-12, out=np.ones_like(tangent_norm), where=tangent_norm > max_tangent)
                tangent = tangent * scale[:, None] * drift_fraction
                status = "pca_radial_witness_shell"
            else:
                tangent = np.zeros((nd, 3), dtype=np.float64)
                status = "pca_radial_quantile_shell"

            smooth_n = int(max(0, GSE_CELL_SURFACE_SMOOTH_ITERATIONS))
            smooth_w = float(np.clip(float(GSE_CELL_SURFACE_SMOOTH_WEIGHT), 0.0, 1.0))
            for _ in range(smooth_n):
                if smooth_w <= 0.0:
                    break
                r_new = radii.copy()
                t_new = tangent.copy()
                for i, nb in enumerate(neighbor_arrays):
                    if nb.size:
                        r_new[i] = (1.0 - smooth_w) * radii[i] + smooth_w * float(np.mean(radii[nb]))
                        t_new[i] = (1.0 - smooth_w) * tangent[i] + smooth_w * np.mean(tangent[nb], axis=0)
                radii = np.maximum(r_new, r_floor)
                tangent = t_new
                np.maximum.at(radii, nearest, support)

    local = dirs * radii[:, None] + tangent
    verts = center[None, :] + (local * scales[None, :]) @ eigvecs.T
    verts = verts.astype(np.float32, copy=False)
    if not np.all(np.isfinite(verts)):
        return None, "nonfinite_vertices", n
    return verts, status, n


def _remove_dataset_array(attrs, name: str) -> None:
    try:
        del attrs[str(name)]
        return
    except Exception:
        pass
    try:
        attrs.remove(str(name))
    except Exception:
        pass


def _ensure_surface_mesh_point_rgba(mesh: "pv.PolyData", rgba: Optional[np.ndarray] = None) -> "pv.PolyData":
    """Use per-vertex RGBA for large merged meshes to avoid cell-data texture limits."""
    if mesh is None:
        return mesh
    try:
        if "rgba" in mesh.point_data and np.asarray(mesh.point_data["rgba"]).shape[0] == int(mesh.n_points):
            _remove_dataset_array(mesh.cell_data, "rgba")
            return mesh
    except Exception:
        pass
    if rgba is None:
        try:
            if "rgba" not in mesh.cell_data:
                return mesh
            rgba = np.asarray(mesh.cell_data["rgba"], dtype=np.uint8)
        except Exception:
            return mesh
    rgba = np.asarray(rgba, dtype=np.uint8)
    if rgba.ndim != 2 or rgba.shape[0] != int(mesh.n_cells) or rgba.shape[1] not in (3, 4):
        return mesh
    if rgba.shape[1] == 3:
        rgba = np.hstack([rgba, np.full((rgba.shape[0], 1), 255, dtype=np.uint8)])
    try:
        faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 4)
    except Exception:
        return mesh
    if faces.shape[0] != rgba.shape[0] or not np.all(faces[:, 0] == 3):
        return mesh
    tri = faces[:, 1:4].astype(np.int64, copy=False)
    point_rgba = np.zeros((int(mesh.n_points), 4), dtype=np.uint8)
    point_rgba[tri.reshape(-1)] = np.repeat(rgba, 3, axis=0)
    mesh.point_data["rgba"] = point_rgba
    _remove_dataset_array(mesh.cell_data, "rgba")
    return mesh


def _configure_cell_surface_lighting(plotter: "pv.Plotter") -> None:
    """Use explicit surface lighting while keeping SSAO as the depth cue."""
    if pv is None:
        return
    if bool(GSE_CELL_SURFACE_ENABLE_LIGHTKIT) and hasattr(plotter, "enable_lightkit"):
        try:
            plotter.enable_lightkit()
        except Exception as e:
            print(f"[PyVistaCellSurface] Warning: LightKit setup failed ({e}); using explicit fallback lights.")
    if not bool(GSE_CELL_SURFACE_ENABLE_CUSTOM_LIGHTS):
        return
    specs = [
        ((220.0, -260.0, 360.0), float(GSE_CELL_SURFACE_KEY_LIGHT_INTENSITY)),
        ((-280.0, 160.0, 220.0), float(GSE_CELL_SURFACE_FILL_LIGHT_INTENSITY)),
        ((120.0, 280.0, -180.0), float(GSE_CELL_SURFACE_RIM_LIGHT_INTENSITY)),
    ]
    for pos, inten in specs:
        try:
            plotter.add_light(pv.Light(position=pos, focal_point=(0.0, 0.0, 0.0), intensity=float(inten), positional=False))
        except TypeError:
            try:
                plotter.add_light(pv.Light(position=pos, focal_point=(0.0, 0.0, 0.0), intensity=float(inten)))
            except Exception:
                pass
        except Exception:
            pass


def _build_infomap_cell_surface_mesh(
    coords: np.ndarray,
    clusters: np.ndarray,
    valid_mask: np.ndarray,
    *,
    rng: np.random.Generator,
    sample_name: str,
) -> Tuple[Optional["pv.PolyData"], List[Dict[str, object]]]:
    """Build one merged colored PolyData containing all infomap-cell surfaces.

    The expensive per-cell geometry step is embarrassingly parallel and does not
    touch VTK.  Build those cell envelopes in a bounded ThreadPool, then create
    the merged PyVista PolyData once on the main thread.  This avoids unsafe GPU
    or VTK multithreading while letting NumPy/SciPy/QHull-heavy work overlap.
    """
    if pv is None:
        return None, []
    valid_idx = np.flatnonzero(np.asarray(valid_mask, dtype=bool))
    if valid_idx.size == 0:
        return None, []

    labels_valid = _cluster_array_to_tokens(clusters[valid_idx])
    inv, unique_labels = pd.factorize(labels_valid, sort=False)
    inv = np.asarray(inv, dtype=np.int64)
    n_cells_total = int(len(unique_labels))
    counts = np.bincount(inv, minlength=n_cells_total).astype(np.int64, copy=False)
    order = np.argsort(inv, kind="mergesort")
    grouped_valid_idx = valid_idx[order]
    offsets = np.zeros(n_cells_total + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)

    directions, tri_faces, neighbor_arrays = _cell_surface_template(int(GSE_CELL_SURFACE_DIRECTION_COUNT))
    try:
        extent_idx = valid_idx if valid_idx.size <= int(GSE_STATS_SAMPLE_N) else rng.choice(valid_idx, size=int(GSE_STATS_SAMPLE_N), replace=False)
        pts_for_extent = coords[extent_idx, :]
        lo = np.nanquantile(pts_for_extent, 0.01, axis=0)
        hi = np.nanquantile(pts_for_extent, 0.99, axis=0)
        diag = float(np.linalg.norm(hi - lo))
    except Exception:
        diag = 1.0
    global_radius_floor = float(max(diag * float(GSE_CELL_SURFACE_DEGENERATE_RADIUS_FRACTION), 1e-6))

    seed = _stable_uint32_seed_from_text(str(sample_name), salt=int(RANDOM_SEED) + 104729)
    colors = _random_bright_rgba_lut(n_cells_total, alpha_u8=255, seed=seed)
    progress_step = max(1000, n_cells_total // 10)

    def _one_cell(ci: int) -> Dict[str, object]:
        group = grouped_valid_idx[offsets[ci]: offsets[ci + 1]]
        label = str(unique_labels[ci])
        cell_seed = _stable_uint32_seed_from_text(
            f"{sample_name}:{label}:{int(ci)}",
            salt=int(RANDOM_SEED) + 271828,
        )
        rng_i = np.random.default_rng(cell_seed)
        if group.size <= 0:
            row = {
                "sample": str(sample_name),
                "infomap_cell_id": label,
                "infomap_cell_index": int(ci),
                "n_hubs_total": 0,
                "n_hubs_surface_sampled": 0,
                "n_surface_vertices": 0,
                "n_surface_faces": 0,
                "surface_method": "empty_group",
            }
            row.update(_cell_surface_metric_dict(np.empty((0, 3), dtype=np.float32)))
            return {"ci": int(ci), "verts": None, "faces_local": None, "n_faces": 0, "row": row}
        pts = _sample_points_for_cell_surface(coords, group, rng=rng_i)
        verts, faces_local, method, n_used = _build_cell_surface_geometry_for_points(
            pts,
            directions=directions,
            tri_faces=tri_faces,
            neighbor_arrays=neighbor_arrays,
            global_radius_floor=global_radius_floor,
        )
        if verts is None or faces_local is None or int(np.asarray(faces_local).shape[0]) <= 0:
            row = {
                "sample": str(sample_name),
                "infomap_cell_id": label,
                "infomap_cell_index": int(ci),
                "n_hubs_total": int(group.size),
                "n_hubs_surface_sampled": int(n_used),
                "n_surface_vertices": 0,
                "n_surface_faces": 0,
                "surface_method": str(method),
            }
            row.update(_cell_surface_metric_dict(pts))
            return {"ci": int(ci), "verts": None, "faces_local": None, "n_faces": 0, "row": row}
        faces_local = np.asarray(faces_local, dtype=np.int64)
        n_faces_i = int(faces_local.shape[0])
        row = {
            "sample": str(sample_name),
            "infomap_cell_id": label,
            "infomap_cell_index": int(ci),
            "n_hubs_total": int(group.size),
            "n_hubs_surface_sampled": int(n_used),
            "n_surface_vertices": int(np.asarray(verts).shape[0]),
            "n_surface_faces": int(n_faces_i),
            "surface_method": str(method),
            "color_r": int(colors[ci, 0]),
            "color_g": int(colors[ci, 1]),
            "color_b": int(colors[ci, 2]),
            "color_a": int(colors[ci, 3]),
        }
        row.update(_cell_surface_metric_dict(pts, verts, faces_local))
        return {
            "ci": int(ci),
            "verts": np.asarray(verts, dtype=np.float32),
            "faces_local": faces_local,
            "n_faces": int(n_faces_i),
            "row": row,
        }

    workers = int(max(1, GSE_CELL_SURFACE_WORKERS))
    use_threads = workers > 1 and n_cells_total >= int(max(1, GSE_CELL_SURFACE_PARALLEL_MIN_CELLS))
    if use_threads:
        print(f"[PyVistaCellSurface] Building {n_cells_total:,} cell envelopes with {workers} CPU worker threads (VTK rendering remains single-threaded).")
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cell_surface") as ex:
            results = list(ex.map(_one_cell, range(n_cells_total), chunksize=1))
    else:
        results = []
        for ci in range(n_cells_total):
            results.append(_one_cell(ci))
            if (ci + 1) % progress_step == 0:
                print(f"[PyVistaCellSurface] Built {ci + 1:,}/{n_cells_total:,} candidate cell envelopes for {sample_name}.")
    results.sort(key=lambda r: int(r.get("ci", -1)))

    vertex_parts: List[np.ndarray] = []
    face_parts: List[np.ndarray] = []
    rgba_parts: List[np.ndarray] = []
    cell_index_parts: List[np.ndarray] = []
    cell_rows: List[Dict[str, object]] = []
    n_vertices_so_far = 0
    n_rendered = 0

    for res in results:
        ci = int(res.get("ci", -1))
        row = res.get("row")
        if isinstance(row, dict):
            cell_rows.append(row)
        verts = res.get("verts")
        faces_local = res.get("faces_local")
        if verts is None or faces_local is None:
            continue
        verts = np.asarray(verts, dtype=np.float32)
        faces_local = np.asarray(faces_local, dtype=np.int64)
        if verts.shape[0] <= 0 or faces_local.shape[0] <= 0:
            continue
        n_faces_i = int(faces_local.shape[0])
        faces_i = np.empty((n_faces_i, 4), dtype=np.int64)
        faces_i[:, 0] = 3
        faces_i[:, 1:] = faces_local + int(n_vertices_so_far)
        vertex_parts.append(verts)
        face_parts.append(faces_i.reshape(-1))
        rgba_parts.append(np.repeat(colors[ci][None, :], n_faces_i, axis=0))
        cell_index_parts.append(np.full(n_faces_i, int(ci), dtype=np.int32))
        n_vertices_so_far += int(verts.shape[0])
        n_rendered += 1

    if not vertex_parts or not face_parts:
        return None, cell_rows

    points = np.vstack(vertex_parts).astype(np.float32, copy=False)
    faces = np.concatenate(face_parts).astype(np.int64, copy=False)
    rgba = np.vstack(rgba_parts).astype(np.uint8, copy=False)
    cell_indices = np.concatenate(cell_index_parts).astype(np.int32, copy=False)
    mesh = pv.PolyData(points, faces)
    mesh = _ensure_surface_mesh_point_rgba(mesh, rgba)
    mesh.cell_data["infomap_cell_index"] = cell_indices
    try:
        mesh = mesh.compute_normals(
            point_normals=True,
            cell_normals=True,
            consistent_normals=False,
            auto_orient_normals=False,
            inplace=False,
        )
    except Exception as e:
        print(f"[PyVistaCellSurface] Warning: compute_normals failed ({e}); rendering without precomputed normals.")
    print(
        f"[PyVistaCellSurface] Built merged mesh for {sample_name}: "
        f"cells={n_rendered:,}/{n_cells_total:,}, vertices={int(mesh.n_points):,}, faces={int(mesh.n_cells):,}, "
        f"workers={workers if use_threads else 1}."
    )
    return mesh, cell_rows


def _add_cell_surface_mesh_actor(plotter: "pv.Plotter", mesh: "pv.PolyData", *, center: np.ndarray) -> object:
    """Add the merged surface mesh using direct point RGBA and lit material."""
    mesh = _ensure_surface_mesh_point_rgba(mesh)
    if "rgba" not in mesh.point_data:
        raise RuntimeError("surface mesh lacks point_data['rgba']; refusing cell-data RGBA rendering")
    kwargs = dict(
        scalars=np.asarray(mesh.point_data["rgba"], dtype=np.uint8),
        rgba=True,
        preference=str(GSE_CELL_SURFACE_COLOR_PREFERENCE),
        show_scalar_bar=False,
        opacity=float(GSE_CELL_SURFACE_OPACITY),
        smooth_shading=bool(GSE_CELL_SURFACE_SMOOTH_SHADING),
        lighting=bool(GSE_CELL_SURFACE_LIGHTING),
        name="infomap_cell_surfaces",
    )
    try:
        actor = plotter.add_mesh(mesh, **kwargs)
    except TypeError:
        kwargs.pop("preference", None)
        try:
            actor = plotter.add_mesh(mesh, **kwargs)
        except TypeError:
            kwargs.pop("smooth_shading", None)
            actor = plotter.add_mesh(mesh, **kwargs)
    try:
        actor.SetPosition(float(-center[0]), float(-center[1]), float(-center[2]))
    except Exception:
        pass
    try:
        prop = actor.GetProperty()
        prop.SetAmbient(float(GSE_CELL_SURFACE_MATERIAL_AMBIENT))
        prop.SetDiffuse(float(GSE_CELL_SURFACE_MATERIAL_DIFFUSE))
        prop.SetSpecular(float(GSE_CELL_SURFACE_MATERIAL_SPECULAR))
        prop.SetSpecularPower(float(GSE_CELL_SURFACE_MATERIAL_SPECULAR_POWER))
        try:
            prop.SetInterpolationToPhong()
        except Exception:
            pass
    except Exception:
        pass
    return actor

def _enable_cell_surface_ssao_required(plotter: "pv.Plotter") -> None:
    """Enable SSAO for cell surfaces; raise if PyVista/VTK cannot provide it."""
    try:
        plotter.enable_ssao(
            radius=float(GSE_CELL_SURFACE_SSAO_RADIUS),
            bias=float(GSE_CELL_SURFACE_SSAO_BIAS),
            kernel_size=int(GSE_CELL_SURFACE_SSAO_KERNEL_SIZE),
        )
        return
    except TypeError:
        try:
            plotter.enable_ssao()
            return
        except Exception as e:
            raise RuntimeError(f"mandatory SSAO failed for cell-surface renderer: {e}") from e
    except Exception as e:
        raise RuntimeError(f"mandatory SSAO failed for cell-surface renderer: {e}") from e


def render_gse_pyvista_infomap_cell_surface_snapshots_for_all_samples(
    *,
    adata_cells: Optional[ad.AnnData],
    file_paths: Sequence[str],
    out_dir: str,
    coord_keys: Tuple[str, str, str] = GSE_COORD_KEYS,
    cluster_key: str = CLUSTER_KEY,
    drop_cluster_values: Optional[set] = DROP_CLUSTER_VALUES,
    label_col: str = GSE_CELL_SURFACE_LEIDEN_FILTER_LABEL_COL,
    force_rebuild: bool = False,
    leiden_foreground_by_sample: Optional[Dict[str, Dict[str, object]]] = None,
    cell_connectivity_h5ad_dir: Optional[str] = None,
    cell_connectivity_suffix: str = "ann12_leiden_cell_connectivity",
    augment_cell_connectivity_h5ad: bool = False,
    render_screenshots: bool = True,
) -> None:
    """Render AO surfaces for post-filter infomap cells selected as colored Leiden foreground.

    When ``augment_cell_connectivity_h5ad`` is true, the geometry/metric analysis
    is completed and written to the sample connectivity H5AD before any PyVista
    screenshot render is started.  ``render_screenshots=False`` performs only
    the geometry-cache/metric/H5AD phase.
    """
    if not GSE_RENDER_ENABLE:
        print("[PyVistaCellSurface] GSE rendering disabled.")
        return
    if not GSE_CELL_SURFACE_RENDER_ENABLE:
        print("[PyVistaCellSurface] Infomap-cell surface rendering disabled.")
        return
    if pv is None:
        print("[PyVistaCellSurface] pyvista not available; skipping infomap-cell surface rendering.")
        return
    if adata_cells is None and bool(GSE_CELL_SURFACE_REQUIRE_POSTFILTER_CELLS):
        print("[PyVistaCellSurface] No post-filter adata_cells supplied; skipping surfaces to avoid rendering pre-QC cells.")
        return

    os.makedirs(out_dir, exist_ok=True)
    summary_rows: List[Dict[str, object]] = []
    leiden_filter_rows: List[Dict[str, object]] = []
    for fp in file_paths:
        sample_name = _infer_sample_name_from_filepath(fp)
        print(f"\n[PyVistaCellSurface] Rendering infomap-cell surfaces for {sample_name} ({fp})")
        if not os.path.exists(fp):
            print(f"[PyVistaCellSurface] File not found: {fp}. Skipping.")
            continue
        try:
            coords, clusters = _read_gse_coords_and_cluster_from_h5ad(fp, coord_keys=coord_keys, cluster_key=cluster_key)
        except Exception as e:
            print(f"[PyVistaCellSurface] Failed reading obs columns from {fp}: {e}")
            continue
        clusters, _clusters_are_tokens_for_surface = _replace_clusters_with_refined_sidecar(
            fp,
            clusters,
            log_prefix="[PyVistaCellSurface]",
            return_tokens=True,
        )

        n_pts = int(coords.shape[0])
        if n_pts <= 0:
            print("[PyVistaCellSurface] No points. Skipping.")
            continue
        coord_ok = np.all(np.isfinite(coords), axis=1)
        raw_valid_mask = _valid_cluster_key_mask_for_random_cell_render(clusters, drop_cluster_values=drop_cluster_values) & coord_ok
        n_raw_valid = int(np.sum(raw_valid_mask))

        cluster_to_label: Dict = {}
        colored_cluster_ids = np.array([], dtype=object)
        leiden_filter_info: Dict[str, object] = {"filter_source": "disabled"}
        if bool(GSE_CELL_SURFACE_FILTER_TO_LEIDEN_COLORED_CELLS):
            if adata_cells is None:
                raise RuntimeError("Leiden-colored cell filtering requires adata_cells metadata.")
            if str(label_col) not in adata_cells.obs.columns:
                raise KeyError(f"Missing obs['{label_col}']; cannot restrict surfaces to Leiden-colored cells.")
            cluster_to_label = _build_cluster_to_label_map_for_sample(
                adata_cells,
                sample_name,
                label_col=str(label_col),
                cluster_id_col=str(GSE_CELL_SURFACE_CLUSTER_ID_COL),
                target_dtype=clusters.dtype,
            )
            cached = dict(leiden_foreground_by_sample.get(str(sample_name), {}) or GSE_LEIDEN_FOREGROUND_CLUSTER_IDS_BY_SAMPLE.get(str(sample_name), {}) or {})
            colored_cluster_ids = np.asarray(cached.get("foreground_cluster_ids", []), dtype=object)
            cached_info = cached.get("filter_info", {})
            leiden_filter_info = dict(cached_info) if isinstance(cached_info, dict) else {"filter_source": "leiden_pyvista_draw_summary"}
            if colored_cluster_ids.size == 0:
                # Surface rendering may be invoked without first rendering the
                # Leiden scatter. In that case, fall back to all non-excluded
                # Leiden-labeled post-filter cells rather than replaying RNG.
                label_strings_raw = _clusters_to_label_strings(
                    clusters,
                    cluster_to_label=cluster_to_label,
                    unassigned_label="__unassigned__",
                )
                label_strings_eff = _mask_excluded_leiden_scatter_labels(label_strings_raw)
                drawn_mask = _is_drawn_leiden_scatter_label_array(label_strings_eff)
                colored_cluster_ids = _unique_cluster_values_for_filter(clusters[drawn_mask])
                label_counts = pd.Series(label_strings_eff[drawn_mask]).astype(str).value_counts()
                leiden_filter_info = {
                    "filter_source": "leiden_label_all_foreground_no_scatter_cache",
                    "foreground_labels": sorted([str(x) for x in label_counts.index.tolist()], key=_natural_sort_key),
                    "major_labels": [],
                    "n_foreground_hubs_drawn": int(np.sum(drawn_mask)),
                    "n_foreground_cells": int(colored_cluster_ids.size),
                }
        else:
            pass

        postfilter_cluster_ids = _postfilter_cluster_ids_for_sample(
            adata_cells,
            sample_name,
            cluster_id_col=str(GSE_CELL_SURFACE_CLUSTER_ID_COL),
            target_dtype=clusters.dtype,
        )
        n_postfilter_cells_total = int(postfilter_cluster_ids.size)
        if n_postfilter_cells_total <= 0:
            print(f"[PyVistaCellSurface] No post-filter cells for sample '{sample_name}'. Skipping.")
            continue

        allowed_cluster_ids = postfilter_cluster_ids
        n_leiden_colored_cells = int(colored_cluster_ids.size)
        if bool(GSE_CELL_SURFACE_FILTER_TO_LEIDEN_COLORED_CELLS):
            if n_leiden_colored_cells <= 0:
                print(
                    f"[PyVistaCellSurface] Leiden-colored foreground selection found no infomap cells for sample "
                    f"'{sample_name}' (major/context labels={leiden_filter_info.get('major_labels', [])}). Skipping."
                )
                continue
            allowed_cluster_ids = allowed_cluster_ids[_cluster_membership_mask(allowed_cluster_ids, colored_cluster_ids)]
            allowed_cluster_ids = _unique_cluster_values_for_filter(allowed_cluster_ids)
            for cid in pd.Series(allowed_cluster_ids).astype(str).tolist():
                leiden_filter_rows.append({
                    "sample": str(sample_name),
                    "infomap_cell_id": str(cid),
                    "leiden_filter_label_col": str(label_col),
                    "filter_source": str(leiden_filter_info.get("filter_source", "leiden_pyvista_draw_summary")),
                    "major_labels": ",".join([str(x) for x in leiden_filter_info.get("major_labels", [])]),
                    "foreground_labels": ",".join([str(x) for x in leiden_filter_info.get("foreground_labels", [])]),
                })

        postfilter_fingerprint = _cluster_id_set_fingerprint(
            allowed_cluster_ids,
            sample_name=sample_name,
            cluster_id_col=str(GSE_CELL_SURFACE_CLUSTER_ID_COL),
        )
        postfilter_fingerprint["n_postfilter_cells_before_leiden_colored_filter"] = int(n_postfilter_cells_total)
        postfilter_fingerprint["leiden_colored_filter_enabled"] = bool(GSE_CELL_SURFACE_FILTER_TO_LEIDEN_COLORED_CELLS)
        postfilter_fingerprint["n_leiden_colored_cells_selected"] = int(n_leiden_colored_cells)
        postfilter_fingerprint["leiden_filter_info"] = {
            k: v for k, v in leiden_filter_info.items()
            if k not in {"label_counts", "foreground_actor_rows"}
        }

        if int(allowed_cluster_ids.size) <= 0:
            print(
                f"[PyVistaCellSurface] No post-filter cells remain after the Leiden-colored foreground filter for "
                f"sample '{sample_name}' (postfilter={n_postfilter_cells_total:,}, "
                f"leiden_colored_selected={n_leiden_colored_cells:,}). Skipping."
            )
            continue
        postfilter_mask = _cluster_membership_mask(clusters, allowed_cluster_ids)
        valid_mask = raw_valid_mask & postfilter_mask
        n_valid = int(np.sum(valid_mask))
        n_postfilter_hubs = int(np.sum(_cluster_membership_mask(clusters, postfilter_cluster_ids) & coord_ok))
        n_allowed_hubs = int(np.sum(postfilter_mask & coord_ok))
        if n_valid <= 0:
            print(
                f"[PyVistaCellSurface] No hubs remain after intersecting valid '{cluster_key}' values "
                f"with {int(allowed_cluster_ids.size):,} surface-eligible cells for sample '{sample_name}'. Skipping."
            )
            continue

        seed = _stable_uint32_seed_from_text(sample_name, salt=int(RANDOM_SEED) + 104729)
        rng = np.random.default_rng(seed)
        valid_idx = np.flatnonzero(valid_mask)
        n_samp = int(min(GSE_STATS_SAMPLE_N, n_valid))
        samp_idx = valid_idx if valid_idx.size <= n_samp else rng.choice(valid_idx, size=n_samp, replace=False)
        pts_samp_raw = coords[samp_idx, :].astype(np.float32, copy=False)
        center = np.median(pts_samp_raw, axis=0).astype(np.float64, copy=False)
        pts_samp = pts_samp_raw.astype(np.float32, copy=True)
        pts_samp -= center.astype(np.float32, copy=False)
        r2 = np.sum(pts_samp ** 2, axis=1)
        q25, q75 = np.quantile(r2, [0.25, 0.75])
        mid_mask = (r2 >= q25) & (r2 <= q75)
        pts_mid = pts_samp[mid_mask]
        if pts_mid.shape[0] < 64:
            pts_mid = pts_samp

        mesh_path, meta_path, cell_table_path = _cell_surface_cache_paths(out_dir, sample_name)
        mesh = _load_cached_cell_surface_mesh(
            mesh_path=mesh_path,
            meta_path=meta_path,
            filepath=fp,
            postfilter_fingerprint=postfilter_fingerprint,
            force_rebuild=bool(force_rebuild or GSE_CELL_SURFACE_FORCE_REBUILD),
        )
        cell_rows: List[Dict[str, object]] = []
        if mesh is None:
            mesh, cell_rows = _build_infomap_cell_surface_mesh(coords, clusters, valid_mask, rng=rng, sample_name=sample_name)
            if mesh is None or int(mesh.n_points) <= 0 or int(mesh.n_cells) <= 0:
                print("[PyVistaCellSurface] No surface geometry was created. Skipping.")
                continue
            _save_cached_cell_surface_mesh(
                mesh,
                cell_rows,
                mesh_path=mesh_path,
                meta_path=meta_path,
                cell_table_path=cell_table_path,
                filepath=fp,
                sample_name=sample_name,
                postfilter_fingerprint=postfilter_fingerprint,
            )
        elif os.path.exists(cell_table_path):
            try:
                cell_rows = pd.read_csv(cell_table_path, sep="\t").to_dict("records")
            except Exception:
                cell_rows = []

        if not _cell_surface_rows_have_metrics(cell_rows):
            cell_rows = _backfill_cell_surface_metric_rows(
                coords=coords,
                clusters=clusters,
                valid_mask=valid_mask,
                mesh=mesh,
                cell_rows=cell_rows,
                sample_name=sample_name,
            )
            if cell_rows and bool(GSE_CELL_SURFACE_CACHE_ENABLE) and bool(GSE_CELL_SURFACE_SAVE_GEOMETRY):
                try:
                    pd.DataFrame(cell_rows).to_csv(cell_table_path, sep="\t", index=False)
                except Exception as e:
                    print(f"[PyVistaCellSurface] Warning: failed writing augmented surface metric table ({e}).")

        if bool(augment_cell_connectivity_h5ad):
            h5ad_path = _cell_connectivity_h5ad_path_for_sample(
                sample_name,
                cell_connectivity_h5ad_dir=cell_connectivity_h5ad_dir,
                cell_connectivity_suffix=str(cell_connectivity_suffix),
            )
            _augment_cell_connectivity_h5ad_with_surface_metrics(
                h5ad_path=h5ad_path,
                cell_rows=cell_rows,
                sample_name=sample_name,
                geometry_cache_path=str(mesh_path),
            )

        if not bool(render_screenshots):
            print(
                f"[PyVistaCellSurface] Prepared surface geometry/metrics for {sample_name} "
                "and skipped screenshots by request. H5AD augmentation, when requested, has already completed."
            )
            summary_rows.append({
                "sample": str(sample_name),
                "filepath": str(fp),
                "cluster_key": str(cluster_key),
                "n_points_total": int(n_pts),
                "n_raw_valid_hubs": int(n_raw_valid),
                "n_postfilter_cells": int(n_postfilter_cells_total),
                "n_leiden_colored_cells_selected": int(n_leiden_colored_cells),
                "n_surface_cells_allowed": int(allowed_cluster_ids.size),
                "n_postfilter_hubs": int(n_postfilter_hubs),
                "n_allowed_hubs": int(n_allowed_hubs),
                "n_valid_hubs": int(n_valid),
                "n_infomap_cells_rendered": int(len(cell_rows)) if cell_rows else int(-1),
                "mesh_vertices": int(mesh.n_points),
                "mesh_faces": int(mesh.n_cells),
                "direction_count": int(GSE_CELL_SURFACE_DIRECTION_COUNT),
                "max_points_per_cell": int(GSE_CELL_SURFACE_MAX_POINTS_PER_CELL),
                "surface_method": str(GSE_CELL_SURFACE_METHOD),
                "surface_workers_configured": int(GSE_CELL_SURFACE_WORKERS),
                "surface_parallel_min_cells": int(GSE_CELL_SURFACE_PARALLEL_MIN_CELLS),
                "leiden_colored_filter_enabled": bool(GSE_CELL_SURFACE_FILTER_TO_LEIDEN_COLORED_CELLS),
                "leiden_filter_label_col": str(label_col),
                "leiden_filter_major_labels": ",".join([str(x) for x in leiden_filter_info.get("major_labels", [])]),
                "leiden_filter_foreground_labels": ",".join([str(x) for x in leiden_filter_info.get("foreground_labels", [])]),
                "ssao_required": True,
                "ssao_radius": float(GSE_CELL_SURFACE_SSAO_RADIUS),
                "ssao_bias": float(GSE_CELL_SURFACE_SSAO_BIAS),
                "ssao_kernel_size": int(GSE_CELL_SURFACE_SSAO_KERNEL_SIZE),
                "geometry_cache_path": str(mesh_path),
                "render_screenshots": False,
                "surface_metrics_h5ad_augmentation_requested": bool(augment_cell_connectivity_h5ad),
            })
            del coords, clusters, mesh, pts_samp, pts_mid
            gc.collect()
            continue

        _pv_start_virtual_framebuffer_if_needed()

        print(
            f"[PyVistaCellSurface] Points: total={n_pts:,}, raw_valid_hubs={n_raw_valid:,}, "
            f"postfilter_cells={n_postfilter_cells_total:,}, "
            f"leiden_colored_cells_replayed={n_leiden_colored_cells:,}, "
            f"surface_cells_allowed={int(allowed_cluster_ids.size):,}, "
            f"postfilter_hubs={n_postfilter_hubs:,}, allowed_hubs={n_allowed_hubs:,}, "
            f"rendered_hubs={n_valid:,}; mesh_vertices={int(mesh.n_points):,}, "
            f"mesh_faces={int(mesh.n_cells):,}; method={GSE_CELL_SURFACE_METHOD}; SSAO=required."
        )

        plotter = pv.Plotter(off_screen=True, window_size=GSE_WINDOW_SIZE)
        try:
            if GSE_BACKGROUND_TOP is not None:
                plotter.set_background(GSE_BACKGROUND, top=GSE_BACKGROUND_TOP)
            else:
                plotter.set_background(GSE_BACKGROUND)
        except Exception:
            plotter.set_background(GSE_BACKGROUND)
        _enable_pyvista_antialiasing(plotter, GSE_ANTIALIASING, log_prefix="[PyVistaCellSurface]")
        _configure_cell_surface_lighting(plotter)
        _add_cell_surface_mesh_actor(plotter, mesh, center=center)
        _enable_cell_surface_ssao_required(plotter)

        plotter.camera.parallel_projection = bool(GSE_CAMERA_PARALLEL_PROJECTION)
        if not plotter.camera.parallel_projection:
            try:
                plotter.camera.view_angle = float(GSE_CAMERA_VIEW_ANGLE_DEG)
            except Exception:
                pass
        focal = np.array([0.0, 0.0, 0.0], dtype=float)
        tail = float(np.clip(0.5 * (1.0 - float(GSE_INFRAME_FRACTION)), 0.0, 0.49))
        try:
            lo = np.quantile(pts_samp, tail, axis=0)
            hi = np.quantile(pts_samp, 1.0 - tail, axis=0)
            diag = float(np.linalg.norm(hi - lo))
        except Exception:
            diag = float(np.linalg.norm(np.ptp(pts_samp, axis=0)))
        radius_guess = 2.0 * diag if diag > 0 else 1.0

        for vi, (az, el) in enumerate(GSE_VIEW_ANGLES_DEG, start=1):
            cam_pos_unit = _camera_position_from_az_el(azimuth_deg=float(az), elevation_deg=float(el), radius=1.0, center=focal)
            cam_dir = cam_pos_unit - focal
            cam_dir /= np.linalg.norm(cam_dir) + 1e-12
            if plotter.camera.parallel_projection:
                cam_pos = focal + cam_dir * float(radius_guess)
                plotter.camera_position = (cam_pos.tolist(), focal.tolist(), (0.0, 0.0, 1.0))
                plotter.camera.parallel_scale = _parallel_scale_for_midpoints(
                    pts_samp,
                    pts_mid,
                    camera_pos=cam_pos,
                    focal=focal,
                    window_size=GSE_WINDOW_SIZE,
                    frame_fraction=float(GSE_INFRAME_FRACTION),
                )
            else:
                dist = _perspective_distance_for_midpoints(
                    pts_samp,
                    pts_mid,
                    camera_pos=(focal + cam_dir),
                    focal=focal,
                    window_size=GSE_WINDOW_SIZE,
                    view_angle_deg=float(GSE_CAMERA_VIEW_ANGLE_DEG),
                    frame_fraction=float(GSE_INFRAME_FRACTION),
                    near_quantile=float(GSE_INFRAME_NEAR_QUANTILE),
                )
                cam_pos = focal + cam_dir * float(dist)
                plotter.camera_position = (cam_pos.tolist(), focal.tolist(), (0.0, 0.0, 1.0))
                try:
                    plotter.reset_camera_clipping_range()
                except Exception:
                    pass
            if GSE_SCALE_BAR_ENABLE:
                try:
                    _add_camera_space_scale_bar(plotter, pts_samp)
                except Exception as e:
                    print(f"[PyVistaCellSurface] Warning: failed to add scale bar ({e})")
            out_png = os.path.join(out_dir, f"{sample_name}_GSE_infomap_cell_surfaces_view{vi:02d}_az{float(az):.0f}_el{float(el):.0f}.png")
            _save_pyvista_screenshot(plotter, out_png, log_prefix="[PyVistaCellSurface]")

        summary_rows.append({
            "sample": str(sample_name),
            "filepath": str(fp),
            "cluster_key": str(cluster_key),
            "n_points_total": int(n_pts),
            "n_raw_valid_hubs": int(n_raw_valid),
            "n_postfilter_cells": int(n_postfilter_cells_total),
            "n_leiden_colored_cells_selected": int(n_leiden_colored_cells),
            "n_surface_cells_allowed": int(allowed_cluster_ids.size),
            "n_postfilter_hubs": int(n_postfilter_hubs),
            "n_allowed_hubs": int(n_allowed_hubs),
            "n_valid_hubs": int(n_valid),
            "n_infomap_cells_rendered": int(len(cell_rows)) if cell_rows else int(-1),
            "mesh_vertices": int(mesh.n_points),
            "mesh_faces": int(mesh.n_cells),
            "direction_count": int(GSE_CELL_SURFACE_DIRECTION_COUNT),
            "max_points_per_cell": int(GSE_CELL_SURFACE_MAX_POINTS_PER_CELL),
            "surface_method": str(GSE_CELL_SURFACE_METHOD),
            "surface_workers_configured": int(GSE_CELL_SURFACE_WORKERS),
            "surface_parallel_min_cells": int(GSE_CELL_SURFACE_PARALLEL_MIN_CELLS),
            "leiden_colored_filter_enabled": bool(GSE_CELL_SURFACE_FILTER_TO_LEIDEN_COLORED_CELLS),
            "leiden_filter_label_col": str(label_col),
            "leiden_filter_major_labels": ",".join([str(x) for x in leiden_filter_info.get("major_labels", [])]),
            "leiden_filter_foreground_labels": ",".join([str(x) for x in leiden_filter_info.get("foreground_labels", [])]),
            "ssao_required": True,
            "ssao_radius": float(GSE_CELL_SURFACE_SSAO_RADIUS),
            "ssao_bias": float(GSE_CELL_SURFACE_SSAO_BIAS),
            "ssao_kernel_size": int(GSE_CELL_SURFACE_SSAO_KERNEL_SIZE),
            "geometry_cache_path": str(mesh_path),
            "render_screenshots": True,
            "surface_metrics_h5ad_augmentation_requested": bool(augment_cell_connectivity_h5ad),
        })
        try:
            plotter.close()
        except Exception:
            pass
        del coords, clusters, mesh, pts_samp, pts_mid
        gc.collect()

    if summary_rows:
        try:
            pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "infomap_cell_surface_render_summary.tsv"), sep="\t", index=False)
        except Exception as e:
            print(f"[PyVistaCellSurface] Warning: failed writing render summary ({e}).")
    if leiden_filter_rows:
        _write_leiden_colored_surface_filter_table(leiden_filter_rows, out_dir=out_dir)
    gc.collect()



# =============================================================================
# Replot/provenance helpers
# =============================================================================







def _write_json_safely(obj: Dict[str, object], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _read_json_safely(path: str) -> Dict[str, object]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception:
        pass
    return {}




def _load_saved_cell_metadata_for_render(table_dir: str) -> Optional[ad.AnnData]:
    """Build a minimal AnnData from cell_metadata_leiden.tsv for PyVista re-rendering.

    This provides cell metadata for surface-only re-rendering without rerunning
    QC, PCA, Leiden, or DE.
    """
    meta_path = os.path.join(str(table_dir), "cell_metadata_leiden.tsv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Cannot replot PyVista outputs because saved cell metadata is missing: {meta_path}")
    meta = pd.read_csv(meta_path, sep="\t")
    required = {"sample", "cluster_id", str(LEIDEN_KEY)}
    missing = sorted(required.difference(set(meta.columns)))
    if missing:
        raise ValueError(f"Cannot replot PyVista outputs because saved cell metadata lacks required columns: {missing}")
    if "cell" in meta.columns:
        meta.index = pd.Index(meta["cell"].astype(str).values, dtype=object)
    else:
        meta.index = pd.Index([f"cell_{i}" for i in range(meta.shape[0])], dtype=object)
    X_stub = sparse.csr_matrix((meta.shape[0], 1), dtype=np.float32)
    var_stub = pd.DataFrame(index=pd.Index(["__dummy__"], dtype=object))
    return ad.AnnData(X=X_stub, obs=meta, var=var_stub)


def _file_paths_from_manifest(table_dir: str) -> List[str]:
    manifest = _read_json_safely(os.path.join(str(table_dir), str(ANALYSIS_MANIFEST)))
    paths = manifest.get("file_paths", []) if isinstance(manifest, dict) else []
    if not isinstance(paths, list):
        return []
    return [str(x) for x in paths if str(x).strip()]


def _effective_cell_slice_arc_alpha(value: object) -> int:
    """Map ultra-low arc alpha values to the subordinate-but-visible default."""
    try:
        alpha = int(np.clip(int(value), 0, 255))
    except Exception:
        alpha = 96
    if 0 < alpha < 16:
        print(
            f"[CoarsenAlignSliceArcs] Requested --coarsen-align-arc-arc-alpha {alpha} is below the visible-range heuristic; "
            "using 96 so arcs remain visible but subordinate. Use 0 to hide arcs or >=16 to force a specific alpha."
        )
        return 96
    return alpha


def _render_coarsen_align_slice_arcs_from_args(
    args: argparse.Namespace,
    *,
    file_paths: Sequence[str],
    fig_dir: str,
) -> None:
    """Call the coarsen-align arc renderer from the normal route."""
    if bool(getattr(args, "skip_coarsen_align_slice_arcs", False)):
        print("[CoarsenAlignSliceArcs] Skipped by request.")
        return
    if not file_paths:
        raise RuntimeError("No file paths available for coarsen-align arc render.")
    print("\n--- PyVista: rendering coarsen-align cell-to-reference-slice arcs ---")
    try:
        import coarsen_align_slice_arcs_pyvista as _coarsen_align_arc_module
        from coarsen_align_slice_arcs_pyvista import render_coarsen_align_slice_arc_snapshots_for_files
        print(f"[CoarsenAlignSliceArcs] Using renderer module: {getattr(_coarsen_align_arc_module, '__file__', 'unknown')}")
        kwargs: Dict[str, object] = dict(
            file_paths=list(file_paths),
            slice_paths=getattr(args, "slice_paths", None),
            out_dir=os.path.join(str(fig_dir), "pyvista_coarsen_align_slice_arcs"),
            coord_keys=GSE_COORD_KEYS,
            slice_cluster_key=str(getattr(args, "slice_cluster_key", "auto")),
            slice_color_key=str(getattr(args, "slice_color_key", "auto")),
            slice_time_key=str(getattr(args, "coarsen_align_arc_slice_time_key", "auto")),
            graph_timepoint=str(getattr(args, "coarsen_align_arc_graph_timepoint", "auto")),
            max_node_points=int(getattr(args, "coarsen_align_arc_max_node_points", 1_000_000)),
            max_context_points=int(getattr(args, "coarsen_align_arc_max_context_points", 0)),
            max_arc_edges=int(getattr(args, "coarsen_align_arc_max_edges", 15_000)),
            max_slice_points=int(getattr(args, "coarsen_align_arc_max_slice_points", 100_000)),
            cell_connectivity_h5ad_dir=(
                str(getattr(args, "coarsened_graph_dir", None))
                if getattr(args, "coarsened_graph_dir", None)
                else os.path.join(str(fig_dir), "sample_cell_connectivity_h5ad")
            ),
            cell_connectivity_suffix="ann12_leiden_cell_connectivity",
            leiden_pyvista_dir=os.path.join(str(fig_dir), "pyvista_gse_scatter_leiden"),
            random_seed=int(RANDOM_SEED),
            node_rgba=(165, 175, 190, int(np.clip(getattr(args, "coarsen_align_arc_node_alpha", 255), 0, 255))),
            context_rgba=(165, 175, 190, int(np.clip(getattr(args, "coarsen_align_arc_context_alpha", 40), 0, 255))),
            arc_alpha=_effective_cell_slice_arc_alpha(getattr(args, "coarsen_align_arc_arc_alpha", 255)),
            slice_alpha=int(np.clip(getattr(args, "coarsen_align_arc_slice_alpha", 255), 1, 255)),
            node_point_size=float(getattr(args, "coarsen_align_arc_node_point_size", GSE_MINOR_CLUSTER_POINT_SIZE)),
            context_point_size=float(getattr(args, "coarsen_align_arc_context_point_size", 1.0)),
            endpoint_point_size=float(getattr(args, "coarsen_align_arc_endpoint_point_size", 5.0)),
            slice_point_size=float(getattr(args, "coarsen_align_arc_slice_point_size", GSE_MINOR_CLUSTER_POINT_SIZE)),
            arc_line_width=float(getattr(args, "coarsen_align_arc_line_width", 0.25)),
            arc_opacity_floor=float(getattr(args, "coarsen_align_arc_opacity_floor", GSE_ARC_OPACITY_FLOOR)),
            slice_frontal_view_bias=float(getattr(args, "coarsen_align_arc_slice_frontal_view_bias", 0.85)),
            external_slice_gap_fraction=float(getattr(args, "coarsen_align_arc_layout_gap_fraction", 0.055)),
            layout_anchor_arcs=int(getattr(args, "coarsen_align_arc_layout_anchor_arcs", 5_000)),
            layout_aware_views=bool(getattr(args, "coarsen_align_arc_layout_aware_views", True)),
            save_az_el_views=bool(getattr(args, "coarsen_align_arc_save_az_el_views", True)),
            window_size=GSE_WINDOW_SIZE,
            background=GSE_BACKGROUND,
            background_top=GSE_BACKGROUND_TOP,
            view_angles_deg=GSE_VIEW_ANGLES_DEG,
            enable_eye_dome_lighting=bool(GSE_ENABLE_EYE_DOME_LIGHTING),
            eye_dome_strength=float(GSE_EDL_STRENGTH),
            eye_dome_radius=float(GSE_EDL_RADIUS),
            antialiasing=GSE_ANTIALIASING,
            camera_parallel_projection=bool(GSE_CAMERA_PARALLEL_PROJECTION),
            camera_view_angle_deg=float(max(32.0, GSE_CAMERA_VIEW_ANGLE_DEG)),
            screenshot_scale=int(GSE_SCREENSHOT_SCALE),
            screenshot_downsample=False,
            save_geometry=not bool(getattr(args, "coarsen_align_arc_no_save_geometry", False)),
            require_leiden_counterpart_views=not bool(getattr(args, "coarsen_align_arc_allow_unpaired_views", False)),
        )
        render_coarsen_align_slice_arc_snapshots_for_files(**kwargs)
    except Exception as e:
        print(f"[CoarsenAlignSliceArcs] Failed: {e}")
        raise


def replot_existing_cell_surfaces_only(
    fig_dir: str,
    table_dir: str,
    *,
    file_paths: Optional[Sequence[str]] = None,
    sample_k: int = 1,
    coarsened_graph_dir: Optional[str] = None,
    force_rebuild_cell_surfaces: bool = False,
) -> None:
    """Replot only cell-surface PNGs for one existing sample and attach surface metrics.

    This intentionally does not re-render PCA/UMAP/marker-map, Leiden scatter, or
    coarsen-align arcs.  It also disables the usual non-L0 Leiden foreground
    filter so L0 and every other post-QC Leiden-labeled cell in the sample is
    eligible for surface rendering.
    """
    global GSE_CELL_SURFACE_FILTER_TO_LEIDEN_COLORED_CELLS

    render_paths = [str(x) for x in file_paths] if file_paths is not None else _file_paths_from_manifest(table_dir)
    if not render_paths:
        raise RuntimeError("Cannot replot surfaces only: no file paths supplied and none found in the manifest.")
    k = int(sample_k)
    if k < 1 or k > len(render_paths):
        raise IndexError(f"--replot-sample-k is 1-based and must be in 1..{len(render_paths)}; got {k}.")
    render_path = str(render_paths[k - 1])
    sample_name = _infer_sample_name_from_filepath(render_path)

    adata_stub = _load_saved_cell_metadata_for_render(table_dir)
    if adata_stub is None:
        raise RuntimeError("Cannot replot surfaces only: saved cell metadata is missing or incomplete.")
    if "sample" in adata_stub.obs.columns:
        keep = adata_stub.obs["sample"].astype(str).values == str(sample_name)
        if not np.any(keep):
            raise RuntimeError(f"Cannot replot surfaces only: saved cell metadata has no rows for sample {sample_name!r}.")
        adata_stub = adata_stub[keep, :].copy()

    cell_graph_dir = str(coarsened_graph_dir) if coarsened_graph_dir else os.path.join(str(fig_dir), "sample_cell_connectivity_h5ad")
    old_filter = bool(GSE_CELL_SURFACE_FILTER_TO_LEIDEN_COLORED_CELLS)
    GSE_CELL_SURFACE_FILTER_TO_LEIDEN_COLORED_CELLS = False
    try:
        print(
            f"[ReplotCellSurfacesOnly] sample_k={k}/{len(render_paths)}, sample={sample_name}; "
            "rendering surfaces only with Leiden-colored filtering disabled, so L0 and all other Leiden clusters are included."
        )
        surface_out_dir = os.path.join(str(fig_dir), str(GSE_CELL_SURFACE_PNG_DIR_NAME))
        print("[ReplotCellSurfacesOnly] Phase 1/2: build/reuse surface geometry, compute metrics, and update H5AD before screenshots.")
        render_gse_pyvista_infomap_cell_surface_snapshots_for_all_samples(
            adata_cells=adata_stub,
            file_paths=[render_path],
            out_dir=surface_out_dir,
            coord_keys=GSE_COORD_KEYS,
            cluster_key=CLUSTER_KEY,
            drop_cluster_values=DROP_CLUSTER_VALUES,
            label_col=str(GSE_CELL_SURFACE_LEIDEN_FILTER_LABEL_COL),
            force_rebuild=bool(force_rebuild_cell_surfaces),
            leiden_foreground_by_sample={},
            cell_connectivity_h5ad_dir=cell_graph_dir,
            cell_connectivity_suffix="ann12_leiden_cell_connectivity",
            augment_cell_connectivity_h5ad=True,
            render_screenshots=False,
        )
        print("[ReplotCellSurfacesOnly] Phase 2/2: render surface screenshots from the now-updated geometry cache/H5AD state.")
        render_gse_pyvista_infomap_cell_surface_snapshots_for_all_samples(
            adata_cells=adata_stub,
            file_paths=[render_path],
            out_dir=surface_out_dir,
            coord_keys=GSE_COORD_KEYS,
            cluster_key=CLUSTER_KEY,
            drop_cluster_values=DROP_CLUSTER_VALUES,
            label_col=str(GSE_CELL_SURFACE_LEIDEN_FILTER_LABEL_COL),
            force_rebuild=False,
            leiden_foreground_by_sample={},
            cell_connectivity_h5ad_dir=cell_graph_dir,
            cell_connectivity_suffix="ann12_leiden_cell_connectivity",
            augment_cell_connectivity_h5ad=False,
            render_screenshots=True,
        )
    finally:
        GSE_CELL_SURFACE_FILTER_TO_LEIDEN_COLORED_CELLS = old_filter



# =============================================================================
# Main focused route
# =============================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse exactly the two supported command forms."""
    forms = (
        "Allowed command forms:\n"
        "  python -u ann12_new3a.py --file-paths STR_1 ... STR_N "
        "--slice-paths spatial_sixtime_slice_stereoseq.h5ad "
        "--coarsen-align-arc-arc-alpha FLOAT "
        "--coarsen-align-arc-line-width FLOAT "
        "--coarsen-align-arc-min-opacity FLOAT "
        "--coarsen-align-arc-frontal-view-bias FLOAT "
        "--fig-dir arcs_envelopes/\n"
        "  python -u ann12_new3a.py --fig-dir STR "
        "--replot-existing-surfaces-only --replot-sample-k 2 "
        "--coarsened-graph-dir arcs_envelopes/sample_cell_connectivity_h5ad"
    )
    parser = argparse.ArgumentParser(
        description="Focused Leiden + PyVista arc/surface pipeline.",
        epilog=forms,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--fig-dir", required=True, help="Output figure directory.")
    parser.add_argument("--file-paths", nargs="+", default=argparse.SUPPRESS, help="Hub-level final.h5ad path(s).")
    parser.add_argument("--slice-paths", nargs=1, default=argparse.SUPPRESS, help="Single spatial_sixtime_slice_stereoseq.h5ad path.")
    parser.add_argument("--coarsen-align-arc-arc-alpha", type=float, default=argparse.SUPPRESS, help="Arc opacity control.")
    parser.add_argument("--coarsen-align-arc-line-width", type=float, default=argparse.SUPPRESS, help="Arc line width.")
    parser.add_argument("--coarsen-align-arc-min-opacity", dest="coarsen_align_arc_opacity_floor", type=float, default=argparse.SUPPRESS, help="Minimum arc opacity after scaling.")
    parser.add_argument("--coarsen-align-arc-frontal-view-bias", dest="coarsen_align_arc_slice_frontal_view_bias", type=float, default=argparse.SUPPRESS, help="Slice-frontal view bias.")
    parser.add_argument("--replot-existing-surfaces-only", action="store_true", default=argparse.SUPPRESS, help="Render only existing cell-surface outputs for one sample.")
    parser.add_argument("--replot-sample-k", type=int, default=argparse.SUPPRESS, help="1-based manifest sample index for surface-only rendering.")
    parser.add_argument("--coarsened-graph-dir", default=argparse.SUPPRESS, help="Existing sample_cell_connectivity_h5ad directory for surface-only rendering.")

    args = parser.parse_args(argv)
    seen = vars(args)
    surface_only = bool(seen.get("replot_existing_surfaces_only", False))
    normal_required = [
        "file_paths",
        "slice_paths",
        "coarsen_align_arc_arc_alpha",
        "coarsen_align_arc_line_width",
        "coarsen_align_arc_opacity_floor",
        "coarsen_align_arc_slice_frontal_view_bias",
    ]
    surface_required = ["replot_sample_k", "coarsened_graph_dir"]
    option_names = {
        "file_paths": "--file-paths",
        "slice_paths": "--slice-paths",
        "coarsen_align_arc_arc_alpha": "--coarsen-align-arc-arc-alpha",
        "coarsen_align_arc_line_width": "--coarsen-align-arc-line-width",
        "coarsen_align_arc_opacity_floor": "--coarsen-align-arc-min-opacity",
        "coarsen_align_arc_slice_frontal_view_bias": "--coarsen-align-arc-frontal-view-bias",
        "replot_sample_k": "--replot-sample-k",
        "coarsened_graph_dir": "--coarsened-graph-dir",
    }

    if surface_only:
        missing = [x for x in surface_required if x not in seen]
        forbidden = [x for x in normal_required if x in seen]
    else:
        missing = [x for x in normal_required if x not in seen]
        forbidden = [x for x in surface_required if x in seen]
    if missing:
        parser.error("missing required option(s) for this command form: " + ", ".join(option_names[x] for x in missing))
    if forbidden:
        parser.error("option(s) not allowed in this command form: " + ", ".join(option_names[x] for x in forbidden))

    defaults = dict(
        file_paths=None,
        slice_paths=None,
        coarsen_align_arc_arc_alpha=96.0,
        coarsen_align_arc_line_width=0.25,
        coarsen_align_arc_opacity_floor=GSE_ARC_OPACITY_FLOOR,
        coarsen_align_arc_slice_frontal_view_bias=0.85,
        replot_existing_surfaces_only=False,
        replot_sample_k=1,
        coarsened_graph_dir=None,
        skip_de=False,
        skip_marker_map=False,
        skip_pyvista=False,
        skip_cell_surfaces_pyvista=False,
        force_rebuild_cell_surfaces_pyvista=False,
        cell_surface_workers=GSE_CELL_SURFACE_WORKERS,
        cell_surface_parallel_min_cells=GSE_CELL_SURFACE_PARALLEL_MIN_CELLS,
        skip_coarsened_cell_graphs=False,
        coarsened_graph_pooled_path=None,
        skip_coarsened_graph_pool=False,
        coarsened_graph_disable_transport=False,
        coarsened_graph_primary=None,
        coarsened_graph_top_genes=10,
        coarsened_graph_binarize_raw_uei=False,
        skip_coarsened_graph_cell_gene_counts=False,
        coarsened_graph_cell_gene_layers=f"{RAW_SUBCONSENSUS_COUNTS_LAYER},{ANALYSIS_COUNTS_LAYER}",
        coarsened_graph_strict=False,
        skip_coarsen_align_slice_arcs=False,
        slice_cluster_key="auto",
        slice_color_key="auto",
        coarsen_align_arc_max_node_points=GSE_BALANCED_TOTAL_MAX_POINTS,
        coarsen_align_arc_max_context_points=0,
        coarsen_align_arc_max_edges=15_000,
        coarsen_align_arc_max_slice_points=100_000,
        coarsen_align_arc_slice_time_key="auto",
        coarsen_align_arc_graph_timepoint="auto",
        coarsen_align_arc_node_alpha=255,
        coarsen_align_arc_context_alpha=25,
        coarsen_align_arc_slice_alpha=255,
        coarsen_align_arc_node_point_size=GSE_MINOR_CLUSTER_POINT_SIZE,
        coarsen_align_arc_context_point_size=1.0,
        coarsen_align_arc_endpoint_point_size=5.0,
        coarsen_align_arc_slice_point_size=GSE_MINOR_CLUSTER_POINT_SIZE,
        coarsen_align_arc_layout_gap_fraction=0.055,
        coarsen_align_arc_layout_anchor_arcs=5_000,
        coarsen_align_arc_layout_aware_views=False,
        coarsen_align_arc_allow_unpaired_views=False,
        coarsen_align_arc_save_az_el_views=True,
        coarsen_align_arc_no_save_geometry=False,
        no_auto_leiden=False,
        leiden_resolution=None,
        force_retune_leiden=False,
        expected_leiden_clusters=None,
        expected_de_ok_clusters=None,
    )
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def main() -> None:
    global FIG_DIR, TABLE_DIR, GSE_CELL_SURFACE_WORKERS, GSE_CELL_SURFACE_PARALLEL_MIN_CELLS

    args = parse_args()
    FIG_DIR = str(args.fig_dir)
    TABLE_DIR = os.path.join(FIG_DIR, "tables")
    GSE_CELL_SURFACE_WORKERS = int(max(1, int(getattr(args, "cell_surface_workers", GSE_CELL_SURFACE_WORKERS))))
    GSE_CELL_SURFACE_PARALLEL_MIN_CELLS = int(max(1, int(getattr(args, "cell_surface_parallel_min_cells", GSE_CELL_SURFACE_PARALLEL_MIN_CELLS))))
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)

    if bool(getattr(args, "replot_existing_surfaces_only", False)):
        print("=" * 72)
        print("Replotting ONLY existing infomap-cell surface renders for one sample; no PCA/UMAP/marker/scatter/arc plots will be written.")
        print(f"Figures: {os.path.abspath(FIG_DIR)}")
        print(f"Tables:  {os.path.abspath(TABLE_DIR)}")
        print("=" * 72)
        replot_file_paths = list(args.file_paths) if args.file_paths else None
        coarsened_graph_dir = (
            str(args.coarsened_graph_dir)
            if getattr(args, "coarsened_graph_dir", None)
            else os.path.join(FIG_DIR, "sample_cell_connectivity_h5ad")
        )
        replot_existing_cell_surfaces_only(
            FIG_DIR,
            TABLE_DIR,
            file_paths=replot_file_paths,
            sample_k=int(getattr(args, "replot_sample_k", 1)),
            coarsened_graph_dir=coarsened_graph_dir,
            force_rebuild_cell_surfaces=bool(args.force_rebuild_cell_surfaces_pyvista),
        )
        return

    file_paths = list(args.file_paths)

    print("=" * 72)
    print("Minimal Leiden-DE + Leiden-colored PyVista pipeline")
    print("Outputs:")
    print(f"  - {os.path.join(FIG_DIR, PCA_FIG)}")
    print(f"  - {os.path.join(FIG_DIR, UMAP_FIG)}")
    print(f"  - {os.path.join(FIG_DIR, UMAP_AGE_FIG)}")
    print(f"  - {os.path.join(FIG_DIR, UMAP_SUBCONS_FIG)}")
    print(f"  - {os.path.join(FIG_DIR, MARKER_MAP_FIG)}")
    print(f"  - {os.path.join(FIG_DIR, 'pyvista_gse_scatter_leiden')}/")
    if (not bool(args.skip_pyvista)) and (not bool(args.skip_coarsen_align_slice_arcs)):
        print(f"  - {os.path.join(FIG_DIR, 'pyvista_coarsen_align_slice_arcs')}/")
    if bool(GSE_CELL_SURFACE_RENDER_ENABLE) and not bool(args.skip_cell_surfaces_pyvista):
        print(f"  - {os.path.join(FIG_DIR, str(GSE_CELL_SURFACE_PNG_DIR_NAME))}/")
    if not bool(args.skip_coarsened_cell_graphs):
        print(f"  - {args.coarsened_graph_dir or os.path.join(FIG_DIR, 'sample_cell_connectivity_h5ad')}/")
    print("=" * 72)

    print("\n--- Loading + aggregating hub-level files ---")
    adatas: List[ad.AnnData] = []
    loaded_file_paths: List[str] = []
    per_file_diags: List[Dict[str, object]] = []
    for fp in file_paths:
        diag: Dict[str, object] = {}
        adata_cell = aggregate_nodes_to_cells(fp, diagnostics_out=diag)
        if diag:
            per_file_diags.append(diag)
        if adata_cell is not None and adata_cell.n_obs > 0 and adata_cell.n_vars > 0:
            adatas.append(adata_cell)
            loaded_file_paths.append(fp)

    if not adatas:
        raise ValueError("No valid query data loaded from --file-paths.")

    keys = [str(d.obs["sample"].iloc[0]) for d in adatas]
    key_counts = Counter(keys)
    duplicate_keys = sorted([k for k, n in key_counts.items() if int(n) > 1])
    if duplicate_keys:
        key_map = ", ".join(f"{k} <- {fp}" for k, fp in zip(keys, loaded_file_paths))
        raise ValueError(f"Non-unique sample keys inferred. Duplicate keys: {duplicate_keys}. Inferred mapping: {key_map}")

    adata_global = ad.concat(adatas, join="outer", label="batch", keys=keys, index_unique="-")
    adata_global.obs_names_make_unique()
    if sparse.isspmatrix(adata_global.X):
        adata_global.X.eliminate_zeros()
    _ensure_counts_layer(adata_global, "counts")
    del adatas
    gc.collect()

    print("\n--- QC + PCA-neighbor graph ---")
    qc_diag: Dict[str, object] = {}
    adata_qc = apply_qc_gates(adata_global.copy(), diagnostics_out=qc_diag)
    _ensure_raw_subconsensus_counts_layer(adata_qc)

    # Generate the cell-filtering diagnostic BEFORE moving on to PCA/Leiden so
    # that even if downstream steps fail we still get a complete record of why
    # the input data was reduced to the QC-passed cell set.
    if DIAG_ENABLE:
        diag_dir = os.path.join(FIG_DIR, DIAG_DIR_NAME)
        try:
            generate_cell_filtering_diagnostics(
                per_file_diags,
                qc_diag,
                out_dir=diag_dir,
            )
        except Exception as e:
            print(f"[Diag] Warning: failed to generate filtering diagnostics ({e}).")

    if DOWNSAMPLE_SUBCONSENSUS_GENE_CALLS:
        print("\n--- Downsampling QC-passed cells to fixed sub-consensus gene-call depth ---")
        downsample_summary = downsample_subconsensus_gene_calls_to_target(
            adata_qc,
            target_counts=int(DOWNSAMPLE_TARGET_COUNTS_PER_CELL),
            layer="counts",
            random_seed=int(RANDOM_SEED),
            reapply_min_genes=True,
        )
        if downsample_summary.shape[0] > 0:
            downsample_summary.to_csv(
                os.path.join(TABLE_DIR, str(DOWNSAMPLE_SUMMARY_TABLE)),
                sep="\t",
                index=False,
            )
            print(f"[Downsample] Wrote table: {DOWNSAMPLE_SUMMARY_TABLE}")

    if not process_for_leiden(adata_qc):
        raise RuntimeError("PCA-neighbor preprocessing failed or insufficient cells.")
    annotate_age_from_sample(adata_qc, sample_col="sample", age_col="age_hpf")
    print(f"   Post-QC: {adata_qc.n_obs:,} cells, {adata_qc.n_vars:,} genes")

    print("\n--- Leiden clustering ---")
    leiden_res = float(LEIDEN_RESOLUTION if args.leiden_resolution is None else args.leiden_resolution)
    manifest_path = os.path.join(TABLE_DIR, str(ANALYSIS_MANIFEST))
    previous_manifest = _read_json_safely(manifest_path)
    if LEIDEN_AUTO_RESOLUTION and not args.no_auto_leiden:
        prev_res = previous_manifest.get("selected_leiden_resolution") if isinstance(previous_manifest, dict) else None
        if (prev_res is not None) and (not bool(args.force_retune_leiden)) and (args.leiden_resolution is None):
            leiden_res = float(prev_res)
            print(
                f"[Leiden] Reusing selected resolution from manifest: {leiden_res:.6g}. "
                "Use --force-retune-leiden to retune."
            )
        else:
            scan_table_path = os.path.join(TABLE_DIR, str(LEIDEN_RESOLUTION_SEARCH_TABLE))
            leiden_res = auto_tune_leiden_resolution_for_palette(
                adata_qc,
                key_added=str(LEIDEN_KEY),
                resolution_grid=list(LEIDEN_RESOLUTION_GRID),
                min_clusters=int(LEIDEN_MIN_CLUSTERS),
                max_clusters=int(LEIDEN_MAX_DISTINCT_COLORS),
                min_cluster_size=int(LEIDEN_MIN_CLUSTER_SIZE),
                random_state=int(LEIDEN_RANDOM_STATE),
                out_table_path=scan_table_path,
            )

    _run_leiden(adata_qc, resolution=float(leiden_res), key_added=str(LEIDEN_KEY), random_state=int(LEIDEN_RANDOM_STATE))
    adata_qc.obs[LEIDEN_KEY] = adata_qc.obs[LEIDEN_KEY].astype(str)
    adata_qc.uns[f"{LEIDEN_KEY}_resolution"] = float(leiden_res)
    adata_qc.uns[f"{LEIDEN_KEY}_n_clusters"] = int(adata_qc.obs[LEIDEN_KEY].nunique())
    n_leiden_clusters = int(adata_qc.obs[LEIDEN_KEY].nunique())
    print(
        f"   Leiden clustering: obs['{LEIDEN_KEY}'] with n_clusters={n_leiden_clusters} "
        f"(resolution={leiden_res:.3g}; PCs={GLOBAL_N_PCS})"
    )
    if args.expected_leiden_clusters is not None and n_leiden_clusters != int(args.expected_leiden_clusters):
        raise RuntimeError(
            f"Leiden cluster count changed: expected {int(args.expected_leiden_clusters)}, got {n_leiden_clusters}. "
            "This would change the marker-map data; rerun with the intended fixed settings."
        )

    out_leiden_dir = os.path.join(FIG_DIR, "pyvista_gse_scatter_leiden")
    leiden_palette = leiden_palette_from_adata(adata_qc, LEIDEN_KEY)
    save_palette_tsvs(leiden_palette, key=LEIDEN_KEY, out_dirs=[TABLE_DIR, out_leiden_dir])

    # Cell metadata table for reproducibility of cluster-to-hub mapping.
    meta = adata_qc.obs.copy()
    meta.insert(0, "cell", adata_qc.obs_names.astype(str))
    meta.to_csv(os.path.join(TABLE_DIR, "cell_metadata_leiden.tsv"), sep="\t", index=False)

    print("\n--- PCA figure: PC1/PC2 colored by the Leiden clustering used downstream ---")
    pca_outpath = os.path.join(FIG_DIR, str(PCA_FIG))
    create_figure_leiden_pca(
        adata_qc,
        outpath=pca_outpath,
        leiden_palette=leiden_palette,
        leiden_key=str(LEIDEN_KEY),
    )
    if not os.path.exists(pca_outpath):
        raise RuntimeError(f"PCA figure was requested but was not written: {os.path.abspath(pca_outpath)}")
    try:
        X_pca = np.asarray(adata_qc.obsm["X_pca"])
        pca_df = pd.DataFrame({
            "cell": adata_qc.obs_names.astype(str),
            "sample": adata_qc.obs["sample"].astype(str).values if "sample" in adata_qc.obs.columns else "",
            str(LEIDEN_KEY): adata_qc.obs[str(LEIDEN_KEY)].astype(str).values,
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
        })
        pca_df.to_csv(os.path.join(TABLE_DIR, "leiden_pca_coordinates.tsv"), sep="\t", index=False)
        print("[PCA] Wrote table: leiden_pca_coordinates.tsv")
    except Exception as e:
        print(f"[PCA] Warning: failed writing PCA coordinates table ({e}).")

    print("\n--- UMAP figure: UMAP1/UMAP2 colored by the same Leiden clustering ---")
    umap_outpath = os.path.join(FIG_DIR, str(UMAP_FIG))
    create_figure_leiden_umap(
        adata_qc,
        outpath=umap_outpath,
        leiden_palette=leiden_palette,
        leiden_key=str(LEIDEN_KEY),
    )
    if not os.path.exists(umap_outpath):
        raise RuntimeError(f"UMAP figure was requested but was not written: {os.path.abspath(umap_outpath)}")
    try:
        X_umap = np.asarray(adata_qc.obsm["X_umap"])
        umap_df = pd.DataFrame({
            "cell": adata_qc.obs_names.astype(str),
            "sample": adata_qc.obs["sample"].astype(str).values if "sample" in adata_qc.obs.columns else "",
            "age_hpf": adata_qc.obs["age_hpf"].astype(str).values if "age_hpf" in adata_qc.obs.columns else "",
            str(LEIDEN_KEY): adata_qc.obs[str(LEIDEN_KEY)].astype(str).values,
            "n_hubs_with_call": pd.to_numeric(adata_qc.obs["n_hubs_with_call"], errors="coerce").values if "n_hubs_with_call" in adata_qc.obs.columns else np.nan,
            "n_distinct_subconsensuses": pd.to_numeric(adata_qc.obs["n_distinct_subconsensuses"], errors="coerce").values if "n_distinct_subconsensuses" in adata_qc.obs.columns else np.nan,
            "UMAP1": X_umap[:, 0],
            "UMAP2": X_umap[:, 1],
        })
        umap_df.to_csv(os.path.join(TABLE_DIR, "leiden_umap_coordinates.tsv"), sep="\t", index=False)
        print("[UMAP] Wrote table: leiden_umap_coordinates.tsv")
    except Exception as e:
        print(f"[UMAP] Warning: failed writing UMAP coordinates table ({e}).")

    print("\n--- Additional UMAP outputs: developmental age and sub-consensus count ---")
    create_figure_umap_by_age(adata_qc, outpath=os.path.join(FIG_DIR, str(UMAP_AGE_FIG)), age_col="age_hpf")
    create_figure_umap_by_subconsensus_count(adata_qc, outpath=os.path.join(FIG_DIR, str(UMAP_SUBCONS_FIG)), value_col="n_distinct_subconsensuses")

    de_all: Optional[pd.DataFrame] = None
    de_summary: Optional[pd.DataFrame] = None
    if (not args.skip_de) and DE_ENABLE:
        print("\n--- Balanced DE on Leiden clusters ---")
        n_samples_for_de = (
            int(adata_qc.obs["sample"].astype(str).nunique())
            if "sample" in adata_qc.obs.columns
            else 1
        )
        use_descriptive_marker_map_de = (
            bool(MARKER_MAP_SINGLE_SAMPLE_DESCRIPTIVE)
            and n_samples_for_de < int(DE_MIN_SAMPLES)
            and bool(MARKER_MAP_ENABLE)
            and not bool(args.skip_marker_map)
        )
        if use_descriptive_marker_map_de:
            print(
                f"[MarkerMap] Only {n_samples_for_de} sample/specimen available; "
                "using descriptive cell-level Leiden cluster-vs-rest marker summaries "
                "so FigureS2_leiden_expression_cluster_map can be written. "
                "The p-value/FDR columns in these rows are placeholders and are not "
                "valid cross-specimen DE evidence."
            )
            de_all, de_summary = run_descriptive_leiden_marker_map_de(
                adata_qc,
                leiden_key=str(LEIDEN_KEY),
                sample_col="sample",
                layer="counts",
                top_n=int(DE_TOP_N_CLUSTERS),
                target_sum=int(DE_TARGET_COUNTS_PER_CELL),
                pseudocount=float(DE_PSEUDOCOUNT),
            )
        else:
            de_all, de_summary = run_balanced_leiden_de(
                adata_qc,
                leiden_key=str(LEIDEN_KEY),
                sample_col="sample",
                layer="counts",
                top_n=int(DE_TOP_N_CLUSTERS),
                cells_per_sample=int(DE_CELLS_PER_SAMPLE),
                target_sum=int(DE_TARGET_COUNTS_PER_CELL),
                min_samples=int(DE_MIN_SAMPLES),
                pseudocount=float(DE_PSEUDOCOUNT),
                random_seed=int(RANDOM_SEED),
            )
        de_all.to_csv(os.path.join(TABLE_DIR, str(DE_TABLE_ALL)), sep="\t", index=False)
        de_summary.to_csv(os.path.join(TABLE_DIR, str(DE_TABLE_SUMMARY)), sep="\t", index=False)
        print(f"[DE] Wrote tables: {DE_TABLE_ALL}, {DE_TABLE_SUMMARY}")
        if "status" in de_summary.columns:
            ok_statuses = {"ok", str(MARKER_MAP_DESCRIPTIVE_STATUS)}
            n_de_ok_clusters = int(de_summary["status"].astype(str).isin(ok_statuses).sum())
        else:
            n_de_ok_clusters = 0
        if args.expected_de_ok_clusters is not None and n_de_ok_clusters != int(args.expected_de_ok_clusters):
            raise RuntimeError(
                f"Balanced-DE cluster count changed: expected {int(args.expected_de_ok_clusters)}, got {n_de_ok_clusters}. "
                "This would change the marker-map data; rerun with the intended fixed settings."
            )

        if MARKER_MAP_ENABLE and not bool(args.skip_marker_map):
            create_figure_leiden_expression_cluster_map(
                de_summary,
                de_all,
                outpath=os.path.join(FIG_DIR, str(MARKER_MAP_FIG)),
                leiden_palette=leiden_palette,
            )
        else:
            print("[MarkerMap] Skipped by request or MARKER_MAP_ENABLE=False.")

        _write_json_safely(
            {
                "selected_leiden_resolution": float(leiden_res),
                "n_leiden_clusters": int(n_leiden_clusters),
                "n_de_ok_clusters": int(n_de_ok_clusters),
                "marker_map_descriptive_cellwise": bool(use_descriptive_marker_map_de),
                "leiden_key": str(LEIDEN_KEY),
                "global_n_neighbors": int(GLOBAL_N_NEIGHBORS),
                "global_n_pcs": int(GLOBAL_N_PCS),
                "qc_min_counts": int(QC_MIN_COUNTS),
                "qc_min_genes": int(QC_MIN_GENES),
                "downsample_subconsensus_gene_calls": bool(DOWNSAMPLE_SUBCONSENSUS_GENE_CALLS),
                "downsample_target_counts_per_cell": int(DOWNSAMPLE_TARGET_COUNTS_PER_CELL),
                "de_cells_per_sample": int(DE_CELLS_PER_SAMPLE),
                "de_target_counts_per_cell": int(DE_TARGET_COUNTS_PER_CELL),
                "gse_scale_bar_enable": bool(GSE_SCALE_BAR_ENABLE),
                "gse_scale_bar_length": float(GSE_SCALE_BAR_LENGTH),
                "gse_scale_bar_margin_frac": [float(GSE_SCALE_BAR_MARGIN_FRAC[0]), float(GSE_SCALE_BAR_MARGIN_FRAC[1])],
                "gse_leiden_scatter_exclude_labels": [str(x) for x in GSE_EXCLUDED_LEIDEN_LABELS],
                "gse_cell_surface_render_enable": bool(GSE_CELL_SURFACE_RENDER_ENABLE and not bool(args.skip_cell_surfaces_pyvista)),
                "gse_cell_surface_method": str(GSE_CELL_SURFACE_METHOD),
                "gse_cell_surface_cluster_id_col": str(GSE_CELL_SURFACE_CLUSTER_ID_COL),
                "gse_cell_surface_require_postfilter_cells": bool(GSE_CELL_SURFACE_REQUIRE_POSTFILTER_CELLS),
                "gse_cell_surface_filter_to_leiden_colored_cells": bool(GSE_CELL_SURFACE_FILTER_TO_LEIDEN_COLORED_CELLS),
                "gse_cell_surface_leiden_filter_label_col": str(GSE_CELL_SURFACE_LEIDEN_FILTER_LABEL_COL),
                "gse_cell_surface_direction_count": int(GSE_CELL_SURFACE_DIRECTION_COUNT),
                "gse_cell_surface_max_points_per_cell": int(GSE_CELL_SURFACE_MAX_POINTS_PER_CELL),
                "gse_cell_surface_alpha_min_points": int(GSE_CELL_SURFACE_ALPHA_MIN_POINTS),
                "gse_cell_surface_alpha_max_points": int(GSE_CELL_SURFACE_ALPHA_MAX_POINTS),
                "gse_cell_surface_alpha_radius_multiplier": float(GSE_CELL_SURFACE_ALPHA_RADIUS_MULTIPLIER),
                "gse_cell_surface_alpha_relaxation_factors": [float(x) for x in GSE_CELL_SURFACE_ALPHA_RELAXATION_FACTORS],
                "gse_cell_surface_alpha_expansion": float(GSE_CELL_SURFACE_ALPHA_EXPANSION),
                "gse_cell_surface_hull_min_points": int(GSE_CELL_SURFACE_HULL_MIN_POINTS),
                "gse_cell_surface_hull_max_witness_points": int(GSE_CELL_SURFACE_HULL_MAX_WITNESS_POINTS),
                "gse_cell_surface_hull_expansion": float(GSE_CELL_SURFACE_HULL_EXPANSION),
                "gse_cell_surface_radius_quantile": float(GSE_CELL_SURFACE_RADIUS_QUANTILE),
                "gse_cell_surface_radius_expansion": float(GSE_CELL_SURFACE_RADIUS_EXPANSION),
                "gse_cell_surface_tangential_drift_fraction": float(GSE_CELL_SURFACE_TANGENTIAL_DRIFT_FRACTION),
                "gse_cell_surface_ssao_required": True,
                "gse_cell_surface_ssao_radius": float(GSE_CELL_SURFACE_SSAO_RADIUS),
                "gse_cell_surface_cache_enable": bool(GSE_CELL_SURFACE_CACHE_ENABLE),
                "marker_map_enabled": bool(MARKER_MAP_ENABLE and not bool(args.skip_marker_map)),
                "marker_map_top_genes_per_leiden": int(MARKER_MAP_TOP_GENES_PER_LEIDEN),
                "marker_map_min_log2fc": float(MARKER_MAP_MIN_LOG2FC),
                "marker_map_min_pct_in": float(MARKER_MAP_MIN_PCT_IN),
                "marker_map_min_pct_delta": float(MARKER_MAP_MIN_PCT_DELTA),
                "marker_map_max_modules": int(MARKER_MAP_MAX_MODULES),
                "umap_random_state": int(UMAP_RANDOM_STATE),
                "age_by_sample": {str(k): str(v) for k, v in AGE_BY_SAMPLE.items()},
                "random_seed": int(RANDOM_SEED),
                "file_paths": [str(x) for x in file_paths],
            },
            os.path.join(TABLE_DIR, str(ANALYSIS_MANIFEST)),
        )
        print(f"[Manifest] Wrote: {ANALYSIS_MANIFEST}")
    else:
        print("[DE] Skipped by request or DE_ENABLE=False.")
        _write_json_safely(
            {
                "selected_leiden_resolution": float(leiden_res),
                "n_leiden_clusters": int(n_leiden_clusters),
                "leiden_key": str(LEIDEN_KEY),
                "global_n_neighbors": int(GLOBAL_N_NEIGHBORS),
                "global_n_pcs": int(GLOBAL_N_PCS),
                "downsample_subconsensus_gene_calls": bool(DOWNSAMPLE_SUBCONSENSUS_GENE_CALLS),
                "downsample_target_counts_per_cell": int(DOWNSAMPLE_TARGET_COUNTS_PER_CELL),
                "gse_scale_bar_enable": bool(GSE_SCALE_BAR_ENABLE),
                "gse_scale_bar_length": float(GSE_SCALE_BAR_LENGTH),
                "gse_scale_bar_margin_frac": [float(GSE_SCALE_BAR_MARGIN_FRAC[0]), float(GSE_SCALE_BAR_MARGIN_FRAC[1])],
                "gse_cell_surface_render_enable": bool(GSE_CELL_SURFACE_RENDER_ENABLE and not bool(args.skip_cell_surfaces_pyvista)),
                "gse_cell_surface_method": str(GSE_CELL_SURFACE_METHOD),
                "gse_cell_surface_cluster_id_col": str(GSE_CELL_SURFACE_CLUSTER_ID_COL),
                "gse_cell_surface_require_postfilter_cells": bool(GSE_CELL_SURFACE_REQUIRE_POSTFILTER_CELLS),
                "gse_cell_surface_filter_to_leiden_colored_cells": bool(GSE_CELL_SURFACE_FILTER_TO_LEIDEN_COLORED_CELLS),
                "gse_cell_surface_leiden_filter_label_col": str(GSE_CELL_SURFACE_LEIDEN_FILTER_LABEL_COL),
                "gse_cell_surface_direction_count": int(GSE_CELL_SURFACE_DIRECTION_COUNT),
                "gse_cell_surface_max_points_per_cell": int(GSE_CELL_SURFACE_MAX_POINTS_PER_CELL),
                "gse_cell_surface_alpha_min_points": int(GSE_CELL_SURFACE_ALPHA_MIN_POINTS),
                "gse_cell_surface_alpha_max_points": int(GSE_CELL_SURFACE_ALPHA_MAX_POINTS),
                "gse_cell_surface_alpha_radius_multiplier": float(GSE_CELL_SURFACE_ALPHA_RADIUS_MULTIPLIER),
                "gse_cell_surface_alpha_relaxation_factors": [float(x) for x in GSE_CELL_SURFACE_ALPHA_RELAXATION_FACTORS],
                "gse_cell_surface_alpha_expansion": float(GSE_CELL_SURFACE_ALPHA_EXPANSION),
                "gse_cell_surface_hull_min_points": int(GSE_CELL_SURFACE_HULL_MIN_POINTS),
                "gse_cell_surface_hull_max_witness_points": int(GSE_CELL_SURFACE_HULL_MAX_WITNESS_POINTS),
                "gse_cell_surface_hull_expansion": float(GSE_CELL_SURFACE_HULL_EXPANSION),
                "gse_cell_surface_radius_quantile": float(GSE_CELL_SURFACE_RADIUS_QUANTILE),
                "gse_cell_surface_radius_expansion": float(GSE_CELL_SURFACE_RADIUS_EXPANSION),
                "gse_cell_surface_tangential_drift_fraction": float(GSE_CELL_SURFACE_TANGENTIAL_DRIFT_FRACTION),
                "gse_cell_surface_ssao_required": True,
                "gse_cell_surface_ssao_radius": float(GSE_CELL_SURFACE_SSAO_RADIUS),
                "gse_cell_surface_cache_enable": bool(GSE_CELL_SURFACE_CACHE_ENABLE),
                "umap_random_state": int(UMAP_RANDOM_STATE),
                "age_by_sample": {str(k): str(v) for k, v in AGE_BY_SAMPLE.items()},
                "random_seed": int(RANDOM_SEED),
                "file_paths": [str(x) for x in file_paths],
            },
            os.path.join(TABLE_DIR, str(ANALYSIS_MANIFEST)),
        )

    if not bool(args.skip_coarsened_cell_graphs):
        print("\n--- Ann12 cell-coarsened graph/count export ---")
        try:
            from caller_module import CoarsenedGraphConfig, run_coarsened_graph_export

            coarsened_out_dir = (
                str(args.coarsened_graph_dir)
                if getattr(args, "coarsened_graph_dir", None)
                else os.path.join(FIG_DIR, "sample_cell_connectivity_h5ad")
            )
            coarsened_cfg = CoarsenedGraphConfig(
                fig_dir=str(FIG_DIR),
                table_dir=str(TABLE_DIR),
                out_dir=coarsened_out_dir,
                filename_suffix="ann12_leiden_cell_connectivity",
                connection_basename=str(CONNECTIVITY_NPZ_BASENAME),
                sidecar_basename=str(REFINED_HUB_LABEL_SIDECAR),
                cluster_id_col="cluster_id",
                cluster_key=str(CLUSTER_KEY),
                coordinate_keys=tuple(str(x) for x in GSE_COORD_KEYS),
                random_seed=int(RANDOM_SEED),
                write_pooled=not bool(args.skip_coarsened_graph_pool),
                pooled_out_path=getattr(args, "coarsened_graph_pooled_path", None),
                top_genes_per_cluster=int(args.coarsened_graph_top_genes),
                transport_enable=not bool(args.coarsened_graph_disable_transport),
                binarize_for_cell_graph=bool(args.coarsened_graph_binarize_raw_uei),
                include_cell_gene_counts=not bool(args.skip_coarsened_graph_cell_gene_counts),
                cell_gene_count_layers=tuple(x.strip() for x in str(args.coarsened_graph_cell_gene_layers).split(",") if x.strip()),
            )
            if getattr(args, "coarsened_graph_primary", None):
                coarsened_cfg.transport_primary_graph = str(args.coarsened_graph_primary).strip().lower()

            coarsened_manifest = run_coarsened_graph_export(
                adata_qc,
                file_paths=loaded_file_paths,
                leiden_keys=[str(LEIDEN_KEY)],
                cfg=coarsened_cfg,
                copy_obsm_keys=("X_pca", "X_umap"),
            )
            try:
                manifest_now = _read_json_safely(os.path.join(TABLE_DIR, str(ANALYSIS_MANIFEST)))
                if isinstance(manifest_now, dict):
                    manifest_now["coarsened_cell_graphs_enabled"] = True
                    manifest_now["coarsened_cell_graphs_dir"] = str(coarsened_out_dir)
                    manifest_now["coarsened_cell_graphs_filename_suffix"] = "ann12_leiden_cell_connectivity"
                    manifest_now["coarsened_cell_graphs_primary_requested"] = str(coarsened_cfg.transport_primary_graph)
                    manifest_now["coarsened_cell_graphs_transport_enabled"] = bool(coarsened_cfg.transport_enable)
                    manifest_now["coarsened_cell_graphs_include_cell_gene_counts"] = bool(coarsened_cfg.include_cell_gene_counts)
                    manifest_now["coarsened_cell_graphs_cell_gene_count_layers"] = [str(x) for x in coarsened_cfg.cell_gene_count_layers]
                    manifest_now["coarsened_cell_graphs_write_pooled"] = bool(coarsened_cfg.write_pooled)
                    manifest_now["coarsened_cell_graphs_manifest"] = "coarsened_graph_export_manifest.json"
                    manifest_now["coarsened_cell_graphs_n_written_samples"] = int(
                        sum(1 for r in coarsened_manifest.get("summary_rows", []) if str(r.get("status", "")) == "written")
                    ) if isinstance(coarsened_manifest, dict) else 0
                    _write_json_safely(manifest_now, os.path.join(TABLE_DIR, str(ANALYSIS_MANIFEST)))
            except Exception as e:
                print(f"[CellGraph] Warning: failed to augment manifest with coarsened graph settings ({e}).")
        except Exception as e:
            if bool(getattr(args, "coarsened_graph_strict", False)):
                raise
            print(f"[CellGraph] Coarsened graph export skipped/failed: {e}")
    else:
        print("[CellGraph] Coarsened graph analysis skipped by request.")

    if (not args.skip_pyvista) and GSE_RENDER_ENABLE:
        print("\n--- PyVista: rendering hub-level GSE clouds colored by Leiden ---")
        render_gse_pyvista_snapshots_for_all_samples(
            adata_qc,
            file_paths=file_paths,
            out_dir=out_leiden_dir,
            coord_keys=GSE_COORD_KEYS,
            cluster_key=CLUSTER_KEY,
            label_col=LEIDEN_KEY,
            label_palette_hex=leiden_palette,
        )

        _render_coarsen_align_slice_arcs_from_args(
            args,
            file_paths=file_paths,
            fig_dir=str(FIG_DIR),
        )

        if bool(GSE_CELL_SURFACE_RENDER_ENABLE) and not bool(args.skip_cell_surfaces_pyvista):
            print("\n--- PyVista: rendering ambient-occluded surfaces for infomap cell IDs ---")
            render_gse_pyvista_infomap_cell_surface_snapshots_for_all_samples(
                adata_cells=adata_qc,
                file_paths=file_paths,
                out_dir=os.path.join(FIG_DIR, str(GSE_CELL_SURFACE_PNG_DIR_NAME)),
                coord_keys=GSE_COORD_KEYS,
                cluster_key=CLUSTER_KEY,
                drop_cluster_values=DROP_CLUSTER_VALUES,
                label_col=str(GSE_CELL_SURFACE_LEIDEN_FILTER_LABEL_COL),
                force_rebuild=bool(args.force_rebuild_cell_surfaces_pyvista),
            )
        else:
            print("[PyVistaCellSurface] Skipped by request or GSE_CELL_SURFACE_RENDER_ENABLE=False.")


    else:
        print("[PyVista] Skipped by request or GSE_RENDER_ENABLE=False.")

    print("\n" + "=" * 72)
    print("Complete.")
    print(f"Figures saved to: {FIG_DIR}/")
    print(f"  - {PCA_FIG}")
    print(f"  - {PCA_FIG_PNG}")
    print(f"  - {UMAP_FIG}")
    print(f"  - {UMAP_FIG_PNG}")
    print(f"  - {UMAP_AGE_FIG}")
    print(f"  - {UMAP_AGE_FIG_PNG}")
    print(f"  - {UMAP_SUBCONS_FIG}")
    print(f"  - {UMAP_SUBCONS_FIG_PNG}")
    print(f"  - {MARKER_MAP_FIG}")
    print(f"  - {os.path.splitext(MARKER_MAP_FIG)[0] + '.png'}")
    print(f"  - pyvista_gse_scatter_leiden/")
    if (not bool(getattr(args, "skip_coarsen_align_slice_arcs", False))) and (not bool(getattr(args, "skip_pyvista", False))):
        print(f"  - pyvista_coarsen_align_slice_arcs/")
    if (not bool(getattr(args, "skip_cell_surfaces_pyvista", False))) and (not bool(getattr(args, "skip_pyvista", False))) and bool(GSE_CELL_SURFACE_RENDER_ENABLE):
        print(f"  - {GSE_CELL_SURFACE_PNG_DIR_NAME}/")
    if not bool(getattr(args, "skip_coarsened_cell_graphs", False)):
        print(f"  - {getattr(args, 'coarsened_graph_dir', None) or os.path.join(FIG_DIR, 'sample_cell_connectivity_h5ad')}/  (ann12 cell-coarsened connectivity/count H5ADs)")
        print("  - tables/pseudobulk_top_genes_all_leiden.tsv")
        print("  - tables/coarsened_graph_export_manifest.json")
    if DIAG_ENABLE:
        print(f"  - {DIAG_DIR_NAME}/  (cell-filtering diagnostics; rank-order plots + per-stage TSV)")
    print(f"Tables saved to:  {TABLE_DIR}/")
    print("=" * 72)


if __name__ == "__main__":
    warnings.filterwarnings("once", category=UserWarning)
    main()
