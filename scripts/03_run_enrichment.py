"""
Step 3 — Motif enrichment analysis (SEA + FIMO).

Reads the input sequences and mapping produced by 01_get_attributions.py,
generates per-subset FASTA files, and runs pairwise enrichment comparisons
using motifs from a specified MoDISco run.

Usage
-----
    python scripts/03_run_enrichment.py --config config/example_config.yaml

Outputs (written under experiment_dir/enrichment/<primary>_vs_<control>/)
--------------------------------------------------------------------------
    fasta_files/            — one .fa per subset
    sea/sea.tsv             — SEA results
    sea/sea_enrichment.csv  — tidy SEA summary
    fimo/fimo_enrichment.csv — FIMO per-motif hit percentages
    */sea_enrichment_scatter.png, */fimo_enrichment_scatter.png
"""
import os
import re
import sys
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utility_functions import setup_experiment_directory, validate_file
from utils.enrichment_functions import (
    run_enrichment_analysis, write_fasta, check_dependencies
)
from utils.sequence_functions import remove_fasta_overlaps

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Run motif enrichment analysis")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    return p.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def crop_sequence(seq: str, element_len: int, flank: int) -> str:
    """Crop a sequence to element_len + flank bp on each side, centered."""
    total_len = len(seq)
    mid = total_len // 2
    half_len = element_len // 2
    start = max(0, mid - half_len - flank)
    end   = min(total_len, mid + (element_len - half_len) + flank)
    return seq[start:end]


def get_subset_sequences(subset_mask, input_seqs, mapping_df, coord_data, flank=None):
    """Extract sequences and names for a boolean row mask over coord_data.

    If flank is given, each sequence is cropped to element_len + flank bp on
    each side (matching the window used by MoDISco). Without cropping, long
    sequences cause FIMO to find chance hits in nearly every entry.
    """
    valid_indices  = np.where(subset_mask)[0]
    subset_mapping = mapping_df[mapping_df["coord_index"].isin(valid_indices)].copy()
    subset_mapping = subset_mapping.dropna(subset=["coord_index", "attribution_index"])
    subset_mapping["coord_index"]      = subset_mapping["coord_index"].astype(int)
    subset_mapping["attribution_index"] = subset_mapping["attribution_index"].astype(int)
    subset_mapping = subset_mapping.drop_duplicates(subset=["attribution_index"], keep="first")

    attr_indices  = subset_mapping["attribution_index"].values
    attr_to_coord = dict(zip(subset_mapping["attribution_index"],
                             subset_mapping["coord_index"]))
    names = (subset_mapping["name"].astype(str).tolist()
             if "name" in subset_mapping.columns
             else [f"seq_{i}" for i in attr_indices])

    seqs = []
    for attr_idx in attr_indices:
        seq = input_seqs[attr_idx]
        if flank is not None:
            coord_idx  = attr_to_coord[attr_idx]
            row        = coord_data.iloc[coord_idx]
            element_len = int(abs(row["end"] - row["start"]))
            seq = crop_sequence(seq, element_len, flank)
        seqs.append(seq)

    return seqs, names


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    experiment_dir  = cfg["experiment_dir"]
    coord_file_path = cfg["coord_file_path"]
    groupby_col     = cfg.get("groupby_column")

    # Which MoDISco run to take motifs from
    enrich_cfg   = cfg.get("enrichment", {})
    source_subset = enrich_cfg.get("source_subset", "all_seqs_modisco")
    source_flank  = enrich_cfg.get("source_flank", "masked_50bp_flank")
    comparisons   = enrich_cfg.get("comparisons", [])
    run_sea         = enrich_cfg.get("run_sea", True)
    run_fimo        = enrich_cfg.get("run_fimo", True)
    dedupe_overlaps = cfg.get("dedupe_overlaps", True)

    # Derive FIMO crop flank from source_flank name (e.g. "masked_50bp_flank" → 50)
    _m = re.search(r"masked_(\d+)bp_flank", source_flank)
    fimo_flank = int(_m.group(1)) if _m else None
    if fimo_flank is not None:
        logger.info(f"FIMO sequences will be cropped to element + {fimo_flank}bp flank")
    else:
        logger.warning("Could not parse flank from source_flank; passing full sequences to FIMO")

    check_dependencies()
    validate_file(coord_file_path, "Coordinate file")
    setup_experiment_directory(experiment_dir)

    # Locate motif file
    motif_file = os.path.join(experiment_dir, source_subset, source_flank, "forward.meme")
    if not os.path.exists(motif_file):
        alt = os.path.join(experiment_dir, source_subset, source_flank, "meme.txt")
        if os.path.exists(alt):
            motif_file = alt
        else:
            raise FileNotFoundError(
                f"Motif file not found at {motif_file} (or meme.txt). "
                "Run 02_run_modisco.py first."
            )
    logger.info(f"Using motifs from: {motif_file}")

    # Load data produced by step 1
    with open(os.path.join(experiment_dir, "input_seqs.pkl"), "rb") as f:
        input_seqs = pickle.load(f)
    mapping_df  = pd.read_csv(os.path.join(experiment_dir, "attribution_mapping.csv"))
    coord_data  = pd.read_csv(coord_file_path)
    logger.info(f"Loaded {len(coord_data)} coordinate rows")

    # Build subset masks (must match logic from 02_run_modisco.py)
    subset_masks = {}
    if groupby_col and groupby_col in coord_data.columns:
        for val in coord_data[groupby_col].unique():
            label = f"{val}_seqs"
            subset_masks[label] = (coord_data[groupby_col] == val).values
    subset_masks["all_seqs"] = np.ones(len(coord_data), dtype=bool)

    # Generate FASTA files for every subset
    fasta_dir   = os.path.join(experiment_dir, "fasta_files")
    os.makedirs(fasta_dir, exist_ok=True)
    fasta_paths = {}
    for name, mask in subset_masks.items():
        seqs, names = get_subset_sequences(mask, input_seqs, mapping_df, coord_data,
                                           flank=fimo_flank)
        out_path    = os.path.join(fasta_dir, f"{name}.fa")
        write_fasta(seqs, names, out_path)
        fasta_paths[name] = out_path
        logger.info(f"FASTA written for '{name}': {len(seqs)} sequences → {out_path}")

    # Register any pre-built control FASTA (e.g. TNF gene control set generated by
    # generate_control_fasta.py).  The file is referenced directly; no copying needed.
    prebuilt_ctrl = enrich_cfg.get("control_fasta")
    if prebuilt_ctrl:
        if not os.path.exists(prebuilt_ctrl):
            raise FileNotFoundError(
                f"enrichment.control_fasta not found: {prebuilt_ctrl}\n"
                "Generate it first with: python scripts/generate_control_fasta.py "
                "--config <cfg> --use-tnf-list --output <path>"
            )
        # Derive the label from the path stem (e.g. tnf_controls_seqs.fa → tnf_controls_seqs)
        ctrl_stem = os.path.splitext(os.path.basename(prebuilt_ctrl))[0]
        if not ctrl_stem.endswith("_seqs"):
            ctrl_stem = ctrl_stem + "_seqs"
        fasta_paths[ctrl_stem] = prebuilt_ctrl
        logger.info(f"Registered pre-built control FASTA '{ctrl_stem}' → {prebuilt_ctrl}")

    # Run comparisons
    if not comparisons:
        logger.warning(
            "No comparisons defined in config enrichment.comparisons. "
            "Nothing to analyse."
        )
        return

    for comp in comparisons:
        prim_label = f"{comp['primary']}_seqs"
        ctrl_label = f"{comp['control']}_seqs"
        if prim_label not in fasta_paths or ctrl_label not in fasta_paths:
            logger.warning(
                f"Skipping {prim_label} vs {ctrl_label}: subset FASTA not found."
            )
            continue

        logger.info(f"Enrichment: {prim_label} vs {ctrl_label}")

        ctrl_fasta = fasta_paths[ctrl_label]
        if dedupe_overlaps:
            logger.info(f"Removing overlaps from control ({ctrl_label})...")
            try:
                ctrl_fasta = remove_fasta_overlaps(
                    primary_fasta=fasta_paths[prim_label],
                    ctrl_fasta=ctrl_fasta,
                    cosi_group=prim_label,
                )
            except Exception as e:
                logger.error(
                    f"Overlap removal failed for {prim_label} vs {ctrl_label}: {e}. "
                    "Proceeding with original control."
                )
                ctrl_fasta = fasta_paths[ctrl_label]

        comp_dir = os.path.join(
            experiment_dir, "enrichment", source_subset, f"{prim_label}_vs_{ctrl_label}"
        )
        run_enrichment_analysis(
            primary_fasta=fasta_paths[prim_label],
            control_fasta=ctrl_fasta,
            meme_file=motif_file,
            output_dir=comp_dir,
            run_sea_tool=run_sea,
            run_fimo_tool=run_fimo,
        )

    logger.info("Enrichment analysis complete.")


if __name__ == "__main__":
    main()
    