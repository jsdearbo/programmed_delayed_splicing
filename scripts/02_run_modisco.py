"""
Step 2 — Run TF-MoDISco motif discovery on masked attributions.

Reads attributions and input sequences produced by 01_get_attributions.py,
optionally splits them into expression-based subsets, masks the attributions
to the region of interest (element ± flank), and runs MoDISco for each
subset/flank combination.

Usage
-----
    python scripts/02_run_modisco.py --config config/example_config.yaml

Outputs (written under experiment_dir/<subset>/<masked_Xbp_flank>/)
---------------------------------------------------------------------
    modisco_report.h5     — raw MoDISco output
    forward.meme          — de-novo motifs (forward strand)
    combined.meme         — forward + reverse-complement motifs
"""
import os
import sys
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.sequence_functions import attribution_native_only, mask_attributions
from utils.utility_functions import (
    setup_experiment_directory, validate_file, load_model, load_saved_attributions
)
from utils.modeling_functions import run_modisco_analysis
from utils.enrichment_functions import convert_modisco_h5_to_meme

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Run TF-MoDISco motif discovery")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    return p.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    experiment_dir  = cfg["experiment_dir"]
    coord_file_path = cfg["coord_file_path"]
    species         = cfg.get("species", "human")
    model_selection = cfg.get("model_selection", "pretrained")
    checkpoint_path = cfg.get("checkpoint_path")
    mask_mode       = cfg.get("mask_mode", "element_only")
    flanks          = cfg.get("flank", [500, 50])
    modisco_len     = cfg.get("modisco_len", 100)
    groupby_col     = cfg.get("groupby_column")  # e.g. 'expression' → splits up/down/all

    if isinstance(flanks, int):
        flanks = [flanks]

    validate_file(coord_file_path, "Coordinate file")
    setup_experiment_directory(experiment_dir)

    model, device = load_model(
        species=species,
        model_selection=model_selection,
        checkpoint_path=checkpoint_path,
    )
    genome = "hg38" if species == "human" else "mm10"

    # Load attributions and sequences saved by step 1
    final_attributions = load_saved_attributions(
        os.path.join(experiment_dir, "attributions.pkl")
    )
    with open(os.path.join(experiment_dir, "input_seqs.pkl"), "rb") as f:
        input_seqs = pickle.load(f)
    mapping_df = pd.read_csv(os.path.join(experiment_dir, "attribution_mapping.csv"))
    coord_data = pd.read_csv(coord_file_path)
    logger.info(f"Loaded {len(coord_data)} coordinate rows")

    # Convert to native-only attributions
    native_only = []
    for i, seq in enumerate(input_seqs):
        native_only.append(attribution_native_only(final_attributions[i], seq))
    final_attributions = np.array(native_only)
    logger.info("Converted to native-only attributions")

    # Build subsets
    subsets = []
    if groupby_col and groupby_col in coord_data.columns:
        for val in coord_data[groupby_col].unique():
            subsets.append((f"{val}_peaks_modisco", coord_data[groupby_col] == val))
    subsets.append(("all_peaks_modisco", np.ones(len(coord_data), dtype=bool)))

    for subset_name, mask in subsets:
        subset_coord_full = coord_data[mask].copy()

        # Map subset rows → attribution indices via an inner join to discard failed sequences
        subset_mapping = pd.merge(
            subset_coord_full, mapping_df,
            left_index=True, right_on="coord_index", how="inner"
        )
        subset_indices      = subset_mapping["attribution_index"].astype(int).values
        subset_attributions = final_attributions[subset_indices]
        subset_input_seqs   = [input_seqs[i] for i in subset_indices]
        
        # Override subset_coord with intersected rows so mask_attributions gets matching lengths
        subset_coord = subset_mapping.reset_index(drop=True)

        if len(subset_input_seqs) == 0:
            logger.warning(f"Subset '{subset_name}' has 0 valid sequences. Skipping to avoid length assertion errors.")
            continue

        for flank in flanks:
            logger.info(f"Running MoDISco: {subset_name} | flank={flank}")
            out_dir = os.path.join(experiment_dir, subset_name, f"masked_{flank}bp_flank")
            os.makedirs(out_dir, exist_ok=True)

            masked_attrs = mask_attributions(
                subset_attributions,
                subset_coord,
                model,
                mode=mask_mode,
                flank=flank,
            )

            # QA: report zero-variance windows
            window_var = np.var(masked_attrs, axis=(1, 2))
            logger.info(f"Zero-variance windows: {np.sum(window_var == 0)}")

            run_modisco_analysis(
                model=model,
                input_seqs=subset_input_seqs,
                experiment_dir=out_dir,
                attributions=masked_attrs,
                device=device,
                genome=genome,
                window=modisco_len,
            )

            # Convert H5 → MEME
            h5_path = os.path.join(out_dir, "modisco_report.h5")
            convert_modisco_h5_to_meme(h5_path, out_dir)

    logger.info("MoDISco analysis complete.")


if __name__ == "__main__":
    main()
    