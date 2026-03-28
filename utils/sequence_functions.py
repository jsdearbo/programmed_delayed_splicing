"""
Sequence preparation, coordinate handling, attribution processing,
and GTF-derived element utilities used by the PABPC1 attribution pipeline.
"""
import logging
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

import grelu.sequence.format

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class AttributionInput:
    sequence: str
    input_intervals: pd.DataFrame
    eval_intervals: pd.DataFrame
    name: str
    meta: pd.Series


# ---------------------------------------------------------------------------
# Interval & Coordinate Helpers
# ---------------------------------------------------------------------------

def get_element_coords(row: pd.Series, measurement: str) -> tuple:
    """
    Return (chrom, start, end, strand) for the requested measurement mode.

    measurement='element_only'    : uses element bounds (start/end columns).
    measurement='whole_transcript': uses transcript bounds (tscript_start/tscript_end).
    """
    chrom = row['chrom']
    strand = row.get('strand', '+')

    if measurement in ['intron_only', 'element_only']:
        start, end = row['start'], row['end']
    elif measurement == 'whole_transcript':
        if 'tscript_start' not in row or 'tscript_end' not in row:
            raise ValueError(
                "measurement='whole_transcript' requires 'tscript_start' and "
                "'tscript_end' columns in the coordinate file."
            )
        start, end = row['tscript_start'], row['tscript_end']
    else:
        raise ValueError(f"Unknown measurement: {measurement}")
    return chrom, start, end, strand


def make_eval_interval(row: pd.Series, measurement: str) -> pd.DataFrame:
    """Return a single-row DataFrame for the evaluation interval."""
    chrom, start, end, _ = get_element_coords(row, measurement)
    return pd.DataFrame({'chrom': [chrom], 'start': [start], 'end': [end]})


def make_input_interval(chrom: str, start: int, end: int,
                         seq_len: int, strand: str) -> pd.DataFrame:
    """Return a single-row DataFrame for the model input window, centered on start/end."""
    center = (start + end) // 2
    input_start = center - seq_len // 2
    input_end = center + seq_len // 2
    return pd.DataFrame({
        'chrom': [chrom], 'start': [input_start],
        'end': [input_end], 'strand': [strand]
    })


def fetch_sequence(intervals: pd.DataFrame, species: str,
                   genome_version: Optional[str] = None) -> str:
    """Fetch the DNA sequence for a single input window via grelu.

    genome_version overrides the default species→genome mapping when set
    (e.g. ``genome_version="GRCh38"`` if genomepy is configured with that name
    instead of the default ``hg38``).
    """
    ivals = intervals.copy()

    if species == "human":
        if genome_version is None:
            genome_version = "hg38"
        ivals['chrom'] = ivals['chrom'].apply(
            lambda x: 'chr' + str(x) if not str(x).startswith('chr') else str(x)
        )
    elif species == "mouse":
        if genome_version is None:
            genome_version = "mm10"
        if genome_version == "GRCm39":
            ivals['chrom'] = ivals['chrom'].apply(
                lambda x: str(x).replace('chr', '') if str(x).startswith('chr') else str(x)
            )

    seq = grelu.sequence.format.convert_input_type(
        ivals, output_type="strings", genome=genome_version
    )[0]
    return seq


# ---------------------------------------------------------------------------
# Input Preparation
# ---------------------------------------------------------------------------

def prepare_inputs(
    df: pd.DataFrame,
    model,
    species: str,
    centering_mode: str,
    attr_respect_to: str,
    name_col: str = None,
    genome_name: str = None,
) -> list:
    """
    Build a list of AttributionInput objects, one per row in df.

    Parameters
    ----------
    df : pd.DataFrame
        Coordinate table (one row per genomic element/peak).
    model :
        Loaded gReLU/Borzoi model (used to read seq_len).
    species : str
        'human' or 'mouse'.
    centering_mode : str
        How to center the input window: 'element_only' or 'whole_transcript'.
    attr_respect_to : str
        Genomic region over which predictions are aggregated for attribution:
        'element_only' or 'whole_transcript'.
    name_col : str, optional
        Column in df to use as element name. Auto-detected if None.
    """
    seq_len = model.data_params['train']['seq_len']
    inputs = []
    logger.info(
        f"Preparing inputs: centering_mode={centering_mode}, "
        f"attr_respect_to={attr_respect_to}"
    )

    if name_col is None:
        for candidate in ('unique_ID', 'intron_name', 'name', 'gene_name'):
            if candidate in df.columns:
                name_col = candidate
                break

    for idx, row in df.iterrows():
        name = str(row[name_col]) if name_col and pd.notna(row.get(name_col)) else f"seq_{idx}"
        chrom, start, end, strand = get_element_coords(row, centering_mode)
        input_intervals = make_input_interval(chrom, start, end, seq_len, strand)
        eval_intervals = make_eval_interval(row, attr_respect_to)
        sequence = fetch_sequence(input_intervals, species, genome_version=genome_name)
        inputs.append(AttributionInput(sequence, input_intervals, eval_intervals, name, meta=row))

    return inputs


# ---------------------------------------------------------------------------
# Output Bin Mapping
# ---------------------------------------------------------------------------

def get_eval_bins(
    model,
    input_intervals: pd.DataFrame,
    eval_intervals: pd.DataFrame,
    output_window: int = 196_608,
) -> list:
    """
    Convert genome-space eval_intervals to Borzoi output bin indices.

    Parameters
    ----------
    model :
        Loaded Borzoi model (provides bin_size via data_params).
    input_intervals : pd.DataFrame
        Single-row DataFrame with the model input window (chrom, start, end).
    eval_intervals : pd.DataFrame
        Single-row DataFrame with the genomic region to aggregate over.
    output_window : int
        Borzoi central output window size in bp (default 196,608).
    """
    input_start  = int(input_intervals.start.iloc[0])
    input_end    = int(input_intervals.end.iloc[0])
    input_center = (input_start + input_end) // 2

    output_half  = output_window // 2
    output_start = max(input_center - output_half, input_start)
    output_end   = min(input_center + output_half, input_end)

    eval_start = max(int(eval_intervals.start.iloc[0]), output_start)
    eval_end   = min(int(eval_intervals.end.iloc[0]),   output_end)

    if eval_start >= eval_end:
        raise ValueError(
            f"Eval interval [{eval_start}, {eval_end}) does not overlap the "
            f"Borzoi output window [{output_start}, {output_end})."
        )

    rel_start = eval_start - output_start
    rel_end   = eval_end   - output_start
    bin_size  = model.data_params["train"]["bin_size"]
    n_bins    = output_window // bin_size

    bin_start = max(0, min(rel_start // bin_size,                     n_bins))
    bin_end   = max(0, min((rel_end + bin_size - 1) // bin_size, n_bins))

    if bin_start >= bin_end:
        raise ValueError(
            f"Empty bin range for eval interval: rel [{rel_start}, {rel_end}), "
            f"bins [{bin_start}, {bin_end})."
        )

    return list(range(bin_start, bin_end))


# ---------------------------------------------------------------------------
# GTF-Derived Element DataFrames
# ---------------------------------------------------------------------------

def create_introns_dataframe(gtf_df: pd.DataFrame) -> pd.DataFrame:
    """Build an intron coordinate table from a processed GTF DataFrame."""
    exon_df = gtf_df[gtf_df['feature'] == 'exon']
    logger.info("Creating introns DataFrame")
    intron_dfs = []
    for transcript_id, exon_group in exon_df.groupby('transcript_id'):
        num_introns = len(exon_group) - 1
        if num_introns < 1:
            continue
        if exon_group['strand'].iloc[0] == '+':
            intron_start = exon_group['end'].iloc[:num_introns].values + 1
            intron_end   = exon_group['start'].iloc[1:num_introns + 1].values - 1
        else:
            intron_start = exon_group['end'].iloc[1:num_introns + 1].values + 1
            intron_end   = exon_group['start'].iloc[:num_introns].values - 1
        intron_length = intron_end - intron_start + 1
        intron_dfs.append(pd.DataFrame({
            'chrom':           exon_group['chrom'].iloc[:num_introns].values,
            'start':           intron_start,
            'end':             intron_end,
            'gene_name':       exon_group['gene_name'].iloc[:num_introns].values,
            'transcript_name': exon_group['transcript_name'].iloc[:num_introns].values,
            'transcript_id':   exon_group['transcript_id'].iloc[:num_introns].values,
            'strand':          exon_group['strand'].iloc[:num_introns].values,
            'intron_number':   range(1, num_introns + 1),
            'intron_length':   intron_length,
        }))
    return pd.concat(intron_dfs, ignore_index=True)


def create_elements_dataframe(intron_df: pd.DataFrame,
                               feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a combined elements DataFrame (exons, introns, UTRs) for each
    transcript in feature_df. Used for gene-map visualisation.
    """
    logger.info("Creating elements DataFrame (exons, introns, UTRs)")
    elements_dfs = []
    for transcript_id in feature_df['transcript_id'].dropna().unique():
        transcript_exons = feature_df[
            (feature_df['transcript_id'] == transcript_id) &
            (feature_df['feature'] == 'exon')
        ].copy()
        transcript_introns = intron_df[intron_df['transcript_id'] == transcript_id].copy()
        transcript_utrs = feature_df[
            (feature_df['transcript_id'] == transcript_id) &
            (feature_df['feature'].isin(['UTR', 'five_prime_UTR', 'three_prime_UTR']))
        ].copy()

        if ('exon_number' in transcript_exons.columns and
                transcript_exons['exon_number'].notna().any()):
            transcript_exons = transcript_exons.assign(
                element=lambda x: 'exon_' + x['exon_number'].astype(int).astype(str),
                element_number=lambda x: x['exon_number'].astype(int)
            )
        else:
            transcript_exons = transcript_exons.sort_values(['start', 'end']).copy()
            transcript_exons['element_number'] = range(1, len(transcript_exons) + 1)
            transcript_exons['element'] = ('exon_' +
                                            transcript_exons['element_number'].astype(str))

        transcript_introns = transcript_introns.assign(
            element=lambda x: 'intron_' + x['intron_number'].astype(str),
            element_number=transcript_introns['intron_number']
        ).rename(columns={'intron_length': 'length'})

        if not transcript_utrs.empty:
            transcript_utrs = transcript_utrs.sort_values(['start', 'end'])
            transcript_utrs = transcript_utrs.assign(
                element=lambda x: (
                    x['feature'].str.replace('_', '').str.lower() + '_' +
                    x.groupby('transcript_id').cumcount().astype(str)
                ),
                element_number=(
                    transcript_utrs.groupby('transcript_id').cumcount() + 100
                )
            )

        combined = pd.concat(
            [transcript_exons, transcript_introns, transcript_utrs],
            ignore_index=True, sort=False
        )
        combined['transcript_id'] = transcript_id
        combined['element_type'] = combined['element'].str.extract(r'(\w+)_')
        combined = combined.sort_values(['start', 'end']).reset_index(drop=True)
        combined['length'] = combined['end'] - combined['start'] + 1
        elements_dfs.append(combined)

    elements_df = pd.concat(elements_dfs, ignore_index=True)
    return elements_df[[
        'chrom', 'start', 'end', 'strand', 'element', 'element_number', 'length',
        'gene_name', 'transcript_id', 'transcript_name', 'element_type'
    ]]


# ---------------------------------------------------------------------------
# Attribution Utilities
# ---------------------------------------------------------------------------

def attribution_native_only(attrs: np.ndarray, seq: str,
                             on_unknown: str = "zero") -> np.ndarray:
    """
    Zero out non-native-base channels in an attribution array.

    attrs can be shaped (L, 4) or (4, L); the same shape is returned.
    on_unknown controls how ambiguous bases (N, etc.) are handled:
    'zero' (default), 'ignore' (keep all channels), or 'error'.
    """
    attrs = np.asarray(attrs)
    if attrs.ndim != 2:
        raise ValueError(f"Expected 2D attrs, got shape {attrs.shape}")

    L = len(seq)
    if attrs.shape == (4, L):
        channel_first = True
        A = attrs
    elif attrs.shape == (L, 4):
        channel_first = False
        A = attrs.T
    else:
        raise ValueError(
            f"Expected attrs shape (4, L) or (L, 4); got {attrs.shape} for seq len {L}"
        )

    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    out = np.zeros_like(A)

    for i, base in enumerate(seq):
        idx = base_to_idx.get(base.upper())
        if idx is not None:
            out[idx, i] = A[idx, i]
        elif on_unknown == "ignore":
            out[:, i] = A[:, i]
        elif on_unknown == "error":
            raise ValueError(f"Unknown base '{base}' at position {i}")

    return out if channel_first else out.T


# ---------------------------------------------------------------------------
# Attribution Masking
# ---------------------------------------------------------------------------

def parse_unique_id(unique_id: str) -> tuple:
    """Split 'GeneName_PeakNumber' → (gene_name, peak_number)."""
    tname, num = unique_id.rsplit('_', 1)
    return tname, int(num)


def _transcript_block(elements_df: pd.DataFrame, transcript_name: str) -> pd.DataFrame:
    """Return position-sorted rows for one transcript."""
    return (elements_df.loc[elements_df['transcript_name'] == transcript_name]
            .sort_values(['start', 'end'])
            .reset_index(drop=True))


def adjacent_rows_transcriptional(elements_df: pd.DataFrame,
                                   transcript_name: str,
                                   intron_number: int,
                                   strand: str):
    """
    Return (upstream_row, intron_row, downstream_row) in transcriptional order.
    Accounts for strand when selecting upstream/downstream neighbors.
    """
    tdf = _transcript_block(elements_df, transcript_name)
    mask = (tdf['element_type'] == 'intron') & (tdf['element_number'] == intron_number)
    if not mask.any():
        raise KeyError(
            f"intron_{intron_number} not found for transcript '{transcript_name}'"
        )
    pos = int(mask[mask].index[0])
    if strand == '+':
        return tdf.iloc[pos - 1], tdf.iloc[pos], tdf.iloc[pos + 1]
    elif strand == '-':
        return tdf.iloc[pos + 1], tdf.iloc[pos], tdf.iloc[pos - 1]
    else:
        raise ValueError(f"Unexpected strand: {strand!r}")


def mask_attributions(
    attributions: np.ndarray,
    coords: pd.DataFrame,
    model=None,
    mode: str = "element_only",
    flank: int = 0,
    seq_len: int = None,
    elements_df: pd.DataFrame = None,
    in_place: bool = False,
) -> np.ndarray:
    """
    Zero out attribution signal outside the region of interest.

    Parameters
    ----------
    attributions : np.ndarray, shape (N, 4, L)
        Raw attribution arrays, one per sequence.
    coords : pd.DataFrame
        Coordinate table aligned with attributions (row i ↔ attribution i).
    model :
        Borzoi model instance (used to read seq_len if not provided directly).
    mode : str
        Masking strategy:
        - 'element_only' / 'peak_only': keep element ± flank.
        - 'context_only': mask element ± flank, keep flanking context.
        - 'upstream_exon', 'downstream_exon', 'adjacent_exons',
          'intron_and_adjacent_exons': exon-aware modes (require elements_df).
    flank : int
        Bp to extend the kept window beyond the element boundary.
    seq_len : int, optional
        Total input sequence length. Inferred from model if not given.
    elements_df : pd.DataFrame, optional
        Required for exon-aware modes.
    in_place : bool
        If True, modify attributions in place (saves memory).
    """
    if not in_place:
        attributions = attributions.copy()

    if seq_len is None:
        if model is not None:
            seq_len = int(model.data_params['train']['seq_len'])
        else:
            raise ValueError("Either model or seq_len must be provided.")

    if not np.issubdtype(attributions.dtype, np.floating):
        attributions = attributions.astype(np.float32, copy=True)

    masked_value = 1e-20

    def _clamp(lo, hi):
        lo = max(0, int(lo))
        hi = min(seq_len, int(hi))
        return (lo, hi) if lo < hi else (None, None)

    for i, row in coords.iterrows():
        elem_len       = abs(int(row['end']) - int(row['start']))
        elem_attr_start = (seq_len - elem_len) // 2
        elem_attr_end   = elem_attr_start + elem_len
        intron_5p_idx   = elem_attr_start
        intron_3p_idx   = elem_attr_end

        if mode in ["element_only", "intron_only", "peak_only"]:
            lo, hi = _clamp(elem_attr_start - flank, elem_attr_end + flank)
            if lo is None:
                attributions[i, :, :] = masked_value
            else:
                attributions[i, :, :lo] = masked_value
                attributions[i, :, hi:] = masked_value

        elif mode == "context_only":
            lo, hi = _clamp(elem_attr_start - flank, elem_attr_end + flank)
            if lo is not None:
                attributions[i, :, lo:hi] = masked_value

        elif mode in {"upstream_exon", "downstream_exon",
                      "adjacent_exons", "intron_and_adjacent_exons"}:
            if elements_df is None:
                raise ValueError("elements_df is required for exon-based modes.")
            if 'unique_ID' not in row:
                raise ValueError(
                    f"Mode '{mode}' requires 'unique_ID' in coord data."
                )
            tname, inum = parse_unique_id(str(row['unique_ID']))
            strand = str(row.get('strand', '+'))
            try:
                up_row, _, dn_row = adjacent_rows_transcriptional(
                    elements_df, tname, inum, strand
                )
            except Exception as e:
                logger.warning(f"Skipping {mode} mask for {tname}: {e}")
                attributions[i, :, :] = masked_value
                continue

            up_len = abs(int(up_row['end']) - int(up_row['start']))
            dn_len = abs(int(dn_row['end']) - int(dn_row['start']))
            up_lo, up_hi = intron_5p_idx - up_len, intron_5p_idx
            dn_lo, dn_hi = intron_3p_idx,          intron_3p_idx + dn_len

            if mode == "upstream_exon":
                lo, hi = _clamp(up_lo - flank, up_hi + flank)
                if lo is None:
                    attributions[i, :, :] = masked_value
                else:
                    attributions[i, :, :lo] = masked_value
                    attributions[i, :, hi:] = masked_value

            elif mode == "downstream_exon":
                lo, hi = _clamp(dn_lo - flank, dn_hi + flank)
                if lo is None:
                    attributions[i, :, :] = masked_value
                else:
                    attributions[i, :, :lo] = masked_value
                    attributions[i, :, hi:] = masked_value

            elif mode == "intron_and_adjacent_exons":
                lo, hi = _clamp(
                    (intron_5p_idx - up_len) - flank,
                    (intron_3p_idx + dn_len) + flank
                )
                if lo is None:
                    attributions[i, :, :] = masked_value
                else:
                    attributions[i, :, :lo] = masked_value
                    attributions[i, :, hi:] = masked_value

            elif mode == "adjacent_exons":
                keep = np.zeros(seq_len, dtype=bool)
                up_clamped = _clamp(up_lo - flank, up_hi + flank)
                dn_clamped = _clamp(dn_lo - flank, dn_hi + flank)
                if up_clamped[0] is not None:
                    keep[up_clamped[0]:up_clamped[1]] = True
                if dn_clamped[0] is not None:
                    keep[dn_clamped[0]:dn_clamped[1]] = True
                attributions[i, :, ~keep] = masked_value

        else:
            raise ValueError(f"Unknown mode: {mode}")

    return attributions


# ---------------------------------------------------------------------------
# FASTA Utilities
# ---------------------------------------------------------------------------

def load_fasta_as_dict(fasta_path: str) -> dict:
    """Parse a FASTA file into {header: sequence} dict."""
    fasta_dict = {}
    with open(fasta_path, 'r') as f:
        seq_name = None
        seq_chunks = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq_name:
                    fasta_dict[seq_name] = ''.join(seq_chunks)
                seq_name = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if seq_name:
            fasta_dict[seq_name] = ''.join(seq_chunks)
    return fasta_dict


def remove_fasta_overlaps(primary_fasta: str, ctrl_fasta: str,
                           cosi_group: str = "primary") -> str:
    """
    Remove sequences from ctrl_fasta whose headers appear in primary_fasta.

    Writes the filtered control to
    ``<ctrl_base>_no_overlaps_<cosi_group>.fa`` alongside the original
    and returns its path.  Safe to call repeatedly — existing output is
    overwritten so results stay current.
    """
    primary_sequences = load_fasta_as_dict(primary_fasta)
    logger.info(
        f"Loaded {len(primary_sequences)} sequences from primary: {primary_fasta}"
    )

    ctrl_sequences = load_fasta_as_dict(ctrl_fasta)
    filtered = {h: s for h, s in ctrl_sequences.items()
                if h not in primary_sequences}
    overlaps_found = len(ctrl_sequences) - len(filtered)

    if overlaps_found:
        logger.info(
            f"Removed {overlaps_found} overlapping sequences from control "
            f"({len(ctrl_sequences)} → {len(filtered)})"
        )
    else:
        logger.info("No overlapping sequences found between primary and control.")

    base_name = os.path.splitext(ctrl_fasta)[0]
    output_fasta = f"{base_name}_no_overlaps_{cosi_group}.fa"
    with open(output_fasta, 'w') as f:
        for header, sequence in filtered.items():
            f.write(f">{header}\n")
            for i in range(0, len(sequence), 80):
                f.write(sequence[i:i + 80] + '\n')

    logger.info(f"Filtered control written to: {output_fasta}")
    return output_fasta
