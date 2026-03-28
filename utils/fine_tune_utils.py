"""
Utilities for Step 0 (fine-tuning): chromosome interval generation,
dataset caching, and path management.

Adapted from gReLU_fine_tuning/utils.py with hardcoded local paths removed.
Chromosome lengths are resolved via genomepy's API rather than a fixed
assembly_report.txt location.
"""
import os
import pickle
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_chromosome_lengths(genome: str) -> pd.DataFrame:
    """
    Return a DataFrame with columns ['chrom', 'length'] for all sequences
    in the given genome assembly (as registered with genomepy).

    Attempts:
      1. genomepy Genome object → .sizes file (two-column TSV, no header)
      2. Fallback: assembly_report.txt in the default genomepy cache dir
         (~/.local/share/genomes/<genome>/assembly_report.txt)
    """
    try:
        import genomepy
        g = genomepy.Genome(genome)
        sizes_file = g.sizes_file
        df = pd.read_csv(sizes_file, sep="\t", header=None, names=["chrom", "length"])
        logger.info(f"Loaded {len(df)} chromosome lengths for {genome} via genomepy")
        return df
    except Exception as e:
        logger.warning(f"genomepy Genome API failed ({e}), trying assembly_report.txt fallback")

    # Fallback: assembly_report.txt in the default cache dir
    cache_dir = os.path.join(os.path.expanduser("~"), ".local", "share", "genomes")
    assembly_report_path = os.path.join(cache_dir, genome, "assembly_report.txt")
    if not os.path.exists(assembly_report_path):
        raise FileNotFoundError(
            f"Could not find chromosome lengths for '{genome}'. "
            f"Checked genomepy API and {assembly_report_path}. "
            "Run: genomepy install <genome>  to download the assembly."
        )
    assembly_report = pd.read_csv(assembly_report_path, sep="\t")
    df = assembly_report[["UCSC-style-name", "Sequence-Length"]].copy()
    df.columns = ["chrom", "length"]
    logger.info(f"Loaded {len(df)} chromosome lengths for {genome} from assembly_report.txt")
    return df


def generate_intervals(
    chromosome_lengths: pd.DataFrame,
    sequence_length: int,
    tstride: bool = False,
) -> pd.DataFrame:
    """
    Tile the genome with non-overlapping (or overlapping, if tstride=True) windows.

    Parameters
    ----------
    chromosome_lengths : pd.DataFrame
        Columns: ['chrom', 'length']
    sequence_length : int
        Window size in bp.
    tstride : bool
        If True, use a stride of sequence_length // 3 - 15 (t-stride walking window).
        If False (default), use non-overlapping tiling.
    """
    intervals_list = []
    for _, row in chromosome_lengths.iterrows():
        chrom  = row["chrom"]
        length = row["length"]
        stride = (sequence_length // 3 - 15) if tstride else sequence_length
        for start in range(0, length, stride):
            end = start + sequence_length
            if end <= length:
                intervals_list.append({"chrom": chrom, "start": start, "end": end})
    df = pd.DataFrame(intervals_list)
    logger.info(f"Generated {len(df)} intervals (tstride={tstride})")
    return df


def make_data_path(
    blacklist: str,
    tstride: bool = False,
    shift: int = None,
    bw_name: str = None,
    trans_func: str = None,
    dataset_cache_dir: str = None,
) -> str:
    """
    Build a deterministic cache path for pre-processed datasets.

    Parameters
    ----------
    blacklist : str
        Blacklist strategy used ('grelu' or 'calico').
    tstride : bool
        Whether t-stride windowing was used.
    shift : int or None
        Max sequence shift used during augmentation.
    bw_name : str
        Identifier string derived from the BigWig file names.
    trans_func : str or None
        Label transform function used ('log1p', 'log', 'squashed', or None).
    dataset_cache_dir : str
        Root directory for caching datasets. Must be provided.
    """
    if dataset_cache_dir is None:
        raise ValueError(
            "dataset_cache_dir must be provided. "
            "Set fine_tuning.dataset_cache_dir in your config."
        )
    stride_status = "stride" if tstride else "no_stride"
    shift_status  = f"shift_{shift}" if shift else "no_shift"
    base_path     = os.path.join(dataset_cache_dir, blacklist, stride_status, shift_status, bw_name)
    full_path     = os.path.join(base_path, trans_func) if trans_func else os.path.join(base_path, "no_scaling")
    return full_path


def load_datasets(dataset_files: dict) -> dict:
    """Load pickled dataset objects from disk."""
    datasets = {}
    for name, file_path in dataset_files.items():
        with open(file_path, "rb") as f:
            datasets[name] = pickle.load(f)
        logger.info(f"Loaded {name} from {file_path}")
    return datasets


def save_datasets(datasets: dict, dataset_files: dict):
    """Pickle dataset objects to disk, creating parent directories as needed."""
    first_path = list(dataset_files.values())[0]
    os.makedirs(os.path.dirname(first_path), exist_ok=True)
    for name, dataset in datasets.items():
        with open(dataset_files[name], "wb") as f:
            pickle.dump(dataset, f)
        logger.info(f"Saved {name} → {dataset_files[name]}")
