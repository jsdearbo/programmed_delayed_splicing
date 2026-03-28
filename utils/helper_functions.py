"""
Seqlet extraction from MoDISco HDF5, coordinate utilities, and
per-sequence motif-hit plotting (logo + gene map).
"""
import os
import logging
import numpy as np
import pandas as pd
import h5py
from tangermeme.tools.fimo import fimo

from .plotting_functions import (
    plot_logo_and_optional_gene_map,
    plot_logo_gene_map_and_read_densities,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Seqlet Extraction
# ---------------------------------------------------------------------------

def extract_seqlets_to_bed_modiscolite(
    h5_file: str,
    pattern_group: str = 'both',
    example_names=None,
    indexing_df=None,
) -> pd.DataFrame:
    """
    Extract seqlets from a MoDISco-lite HDF5 file as a BED-style DataFrame.

    Parameters
    ----------
    h5_file : str
        Path to modisco_results.h5.
    pattern_group : str
        Which pattern groups to extract: 'both' (default), 'pos', or 'neg'.
    example_names : list, optional
        Sequence names aligned with example indices in the H5.
    indexing_df : pd.DataFrame, optional
        Used to look up the native strand per sequence.
    """
    bed_rows = []
    with h5py.File(h5_file, 'r') as f:
        available = set(f.keys())

        def resolve_groups(pg):
            if pg in (None, 'both', 'all', 'patterns'):
                groups = [g for g in ('pos_patterns', 'neg_patterns') if g in available]
                if not groups:
                    groups = [g for g in available if g.endswith('_patterns')]
                return groups
            if pg in ('pos', 'pos_patterns'):
                return [g for g in ('pos_patterns',) if g in available]
            if pg in ('neg', 'neg_patterns'):
                return [g for g in ('neg_patterns',) if g in available]
            return [pg] if pg in available else []

        groups = resolve_groups(pattern_group)
        if not groups:
            raise KeyError(
                f"No pattern groups found for '{pattern_group}'. "
                f"Available: {sorted(available)}"
            )

        for group_name in groups:
            group = f[group_name]
            prefix = 'pos' if group_name.startswith('pos') else 'neg'
            for pattern_name in group.keys():
                seqlets    = group[pattern_name]['seqlets']
                starts     = seqlets['start'][:]
                ends       = seqlets['end'][:]
                strands    = seqlets['is_revcomp'][:]
                example_idxs = seqlets['example_idx'][:]
                for i in range(len(starts)):
                    example = int(example_idxs[i])
                    chrom   = example_names[example] if example_names else example
                    if indexing_df is not None:
                        native_strand = indexing_df.loc[
                            indexing_df['index'] == example, 'strand'
                        ].values[0]
                    else:
                        native_strand = '+'
                    strand = (
                        ('-' if native_strand == '+' else '+')
                        if strands[i] else native_strand
                    )
                    bed_rows.append([
                        chrom, int(starts[i]), int(ends[i]),
                        f"{pattern_name}_seqlet_{i}", 0, strand,
                        f"{prefix}_{pattern_name}", bool(strands[i]),
                    ])

    return pd.DataFrame(
        bed_rows,
        columns=['example_index', 'start', 'end', 'name', 'score',
                 'strand', 'pattern_label', 'is_revcomp'],
    )


# ---------------------------------------------------------------------------
# Coordinate Utilities
# ---------------------------------------------------------------------------

def update_coordinates(df: pd.DataFrame, tensor_start: int) -> pd.DataFrame:
    """Shift seqlet start/end by tensor_start."""
    df = df.copy()
    df["start"] = df["start"] + tensor_start
    df["end"]   = df["end"]   + tensor_start
    return df


def filter_by_strand(df: pd.DataFrame, seq_name: str,
                      indexing_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows matching the native strand for seq_name."""
    correct_strand = indexing_df.loc[
        indexing_df["unique_ID"] == seq_name, "strand"
    ].values[0]
    return df[df["strand"] == correct_strand]


def reorder_modiscolite_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column order for MoDISco-lite output."""
    cols = ['pattern_label', 'example_index', 'start', 'end',
            'name', 'score', 'strand', 'unique_ID']
    return df[cols]


# ---------------------------------------------------------------------------
# Motif Hit Plotting — FIMO
# ---------------------------------------------------------------------------

def handle_fimo(
    motifs_of_interest_dict: dict,
    primary_seq: str,
    indexing_df: pd.DataFrame,
    pt_attributions,
    logo_plot_dir: str,
    elements_df: pd.DataFrame,
):
    """
    Run FIMO on primary_seq and generate a logo+gene-map plot for each hit.
    """
    annotations_list = fimo(motifs=motifs_of_interest_dict, sequences=primary_seq, dim=1)
    all_rows = []
    for df in annotations_list:
        if "sequence_name" in df.columns and not df.empty:
            seq_name = df["sequence_name"].iloc[0]
            tensor_start = indexing_df.loc[
                indexing_df["unique_ID"] == seq_name, "tensor_start"
            ]
            if not tensor_start.empty:
                df = update_coordinates(df, tensor_start.values[0])
                df = filter_by_strand(df, seq_name, indexing_df)
                df['unique_ID'] = seq_name
                if "motif_name" not in df.columns:
                    df['motif_name'] = "unknown"
                all_rows.append(df)

    if not all_rows:
        print("No FIMO hits found.")
        return

    annotations = pd.concat(all_rows, ignore_index=True)
    for de_novo_motif in annotations['motif_name'].unique():
        motif_df = annotations[annotations['motif_name'] == de_novo_motif]
        motif_logo_plot_dir = os.path.join(logo_plot_dir, de_novo_motif)
        os.makedirs(motif_logo_plot_dir, exist_ok=True)
        for seq_name in motif_df['unique_ID'].unique():
            df = motif_df[motif_df['unique_ID'] == seq_name]
            if not df.empty:
                top_row  = df.loc[df['score'].idxmax()]
                attr_idx = indexing_df.loc[
                    indexing_df["unique_ID"] == seq_name, 'index'
                ].values[0]
                arr = pt_attributions[attr_idx]
                plot_logo_and_optional_gene_map(
                    arr, top_row, df, motif_logo_plot_dir, elements_df, indexing_df
                )


# ---------------------------------------------------------------------------
# Motif Hit Plotting — MoDISco-lite
# ---------------------------------------------------------------------------

def handle_modiscolite(
    h5_file: str,
    motifs_of_interest: list,
    indexing_df: pd.DataFrame,
    pt_attributions,
    logo_plot_dir: str,
    elements_df: pd.DataFrame,
    MODEL_SEQ_LEN: int = 524288,
    MODISCO_WINDOW: int = 5000,
    intron_of_interest: str = None,
    bw_files: dict = None,
    y_scale: str = 'log1p',
    figsize: tuple = (20, 1.5),
    **kwargs,
):
    """
    Extract MoDISco seqlets, shift to genomic coordinates, filter by strand,
    and generate a logo+gene-map plot per motif per sequence.

    Parameters
    ----------
    h5_file : str
        Path to modisco_results.h5.
    motifs_of_interest : list of str
        Pattern labels to plot (e.g. ['pos_pattern_0', 'neg_pattern_1']).
    indexing_df : pd.DataFrame
        Per-sequence indexing metadata (output of _prep_indexing_df).
    pt_attributions : array-like, shape (N, 4, L)
        Attribution arrays aligned with indexing_df rows.
    logo_plot_dir : str
        Root directory for output plots.
    elements_df : pd.DataFrame
        Gene/transcript element table for gene-map visualisation.
    MODEL_SEQ_LEN : int
        Full input sequence length (default 524,288 for Borzoi).
    MODISCO_WINDOW : int
        MoDISco seqlet search window used during run_modisco (default 5,000).
    intron_of_interest : str, optional
        If set, restrict plots to this sequence name only.
    bw_files : dict, optional
        {track_name: path_to_bigwig} for read-density tracks.
    y_scale : str
        Y-axis scaling for read-density tracks ('log1p' or 'linear').
    figsize : tuple
        Figure size (width, height) per row of the gene-map panel.
    """
    attr_respect_to = kwargs.pop("attr_respect_to", "intron_only")
    for k in ("bw_files", "y_scale", "figsize"):
        kwargs.pop(k, None)

    annotations = extract_seqlets_to_bed_modiscolite(
        h5_file, 'both', indexing_df=indexing_df
    )
    index_to_name = dict(zip(indexing_df['index'], indexing_df['unique_ID']))
    annotations['unique_ID'] = (
        annotations['example_index'].map(index_to_name).fillna('unknown')
    )

    # Shift seqlet coords from modisco-window space to full input-tensor space
    tensor_window_start = (MODEL_SEQ_LEN // 2) - (MODISCO_WINDOW // 2)
    annotations = update_coordinates(annotations, tensor_window_start)

    # Filter to valid sequences and correct strand
    annotations = annotations[
        annotations.apply(lambda row: (
            row['unique_ID'] in indexing_df['unique_ID'].values and
            row['strand'] == indexing_df.loc[
                indexing_df['unique_ID'] == row['unique_ID'], 'strand'
            ].values[0]
        ), axis=1)
    ]

    annotations['score'] = annotations['score'].astype(float)
    annotations = annotations[annotations['pattern_label'].isin(motifs_of_interest)]
    annotations = reorder_modiscolite_columns(annotations)

    if intron_of_interest:
        annotations = annotations[annotations['unique_ID'] == intron_of_interest]

    for de_novo_motif in annotations['pattern_label'].unique():
        motif_df = annotations[annotations['pattern_label'] == de_novo_motif]
        motif_df.to_csv(
            os.path.join(logo_plot_dir, f"{de_novo_motif}_hits.csv"), index=False
        )
        motif_logo_plot_dir = os.path.join(logo_plot_dir, de_novo_motif)
        os.makedirs(motif_logo_plot_dir, exist_ok=True)

        for seq_name in motif_df['unique_ID'].unique():
            df = motif_df[motif_df['unique_ID'] == seq_name]
            if df.empty:
                continue
            top_row  = df.loc[df['score'].idxmax()]
            attr_idx = indexing_df.loc[
                indexing_df["unique_ID"] == seq_name, 'index'
            ].values[0]
            arr = pt_attributions[attr_idx]
            logger.info(f"Plotting {seq_name} | pattern {de_novo_motif} | "
                        f"attr_sum={float(np.sum(arr)):.4e}")
            plot_logo_gene_map_and_read_densities(
                arr=arr,
                row=top_row,
                seq_annotations=df,
                elements_df=elements_df,
                indexing_df=indexing_df,
                bw_files=bw_files,
                logo_plot_dir=motif_logo_plot_dir,
                y_scale=y_scale,
                figsize=figsize,
                attr_respect_to=attr_respect_to,
                **kwargs,
            )
