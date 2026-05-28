"""
Step 1 — Compute per-nucleotide attributions for each element/peak.

Usage
-----
    python scripts/01_get_attributions.py --config config/example_config.yaml

Outputs (written to experiment_dir/)
-------------------------------------
    attributions.pkl        — numpy array (N, 4, L)
    input_seqs.pkl          — list of input sequences
    element_names_list.pkl  — list of element names
    attribution_mapping.csv — maps coord_index → attribution_index
"""
import os
import sys
import gc
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import yaml

# Make utils/ importable regardless of where the script is called from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.sequence_functions import prepare_inputs, get_eval_bins
from utils.utility_functions import (
    setup_experiment_directory, validate_file, load_model, load_tasks
)
from utils.modeling_functions import get_attributions_for_element

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Compute Borzoi attributions")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    return p.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def main():
    args   = parse_args()
    cfg    = load_config(args.config)

    experiment_dir  = cfg["experiment_dir"]
    coord_file_path = cfg["coord_file_path"]
    species         = cfg.get("species", "human")
    genome_name     = cfg.get("genome_name")          # optional override (e.g. "GRCh38")
    task_id         = cfg["task_id"]
    model_selection = cfg.get("model_selection", "pretrained")
    checkpoint_path = cfg.get("checkpoint_path")
    # centering_mode: how to center the input window
    centering_mode  = cfg.get("centering_mode", cfg.get("attr_respect_to", "element_only"))
    # attr_respect_to: region over which predictions are aggregated
    attr_respect_to = cfg.get("attr_respect_to", "element_only")
    method          = cfg.get("attribution_method", "inputxgradient")
    name_col        = cfg.get("name_col")

    validate_file(coord_file_path, "Coordinate file")
    setup_experiment_directory(experiment_dir)

    # Define genome
    genome = genome_name if genome_name is not None else ("hg38" if species == "human" else "mm10")
    logger.info(f"Using genome: {genome}")

    # Load model
    model, device = load_model(
        species=species,
        model_selection=model_selection,
        checkpoint_path=checkpoint_path,
    )

    # Normalise device to string
    if isinstance(device, int):
        device_str = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    elif isinstance(device, torch.device):
        device_str = str(device)
    else:
        device_str = device or "cpu"
    logger.info(f"Using device: {device_str}")

    # Select RNA tasks for this experiment
    if model_selection == "fine_tuned":
        # Fine-tuned models have a custom head; data_params["tasks"] reflects the
        # pretrained model and cannot be used for name lookup.  Use integer indices.
        from lora_patch import _unwrap_borzoi_model
        head = _unwrap_borzoi_model(model).head
        n_tasks = getattr(head, "n_tasks", 3)
        tasks_list = list(range(n_tasks))
    else:
        tasks = load_tasks(model)
        tasks_list = tasks[tasks["name"].str.contains(task_id, na=False)]["name"].tolist()
    logger.info(f"Tasks selected ({len(tasks_list)}): {tasks_list[:5]} ...")

    # Load coordinates
    coord_data = pd.read_csv(coord_file_path)
    logger.info(f"Loaded {len(coord_data)} coordinate rows")

    # Build AttributionInput objects
    element_inputs = prepare_inputs(
        coord_data, model, species,
        centering_mode=centering_mode,
        attr_respect_to=attr_respect_to,
        name_col=name_col,
        genome_name=genome_name,
    )

    # Compute attributions — track only successful elements so all outputs stay in sync
    seq_len = model.data_params['train']['seq_len']
    attributions_list = []
    successful_inputs = []
    for item in element_inputs:
        try:
            logger.info(f"Processing: {item.name}")
            # Pad short sequences (e.g. near chromosome boundary) to seq_len with Ns
            seq = item.sequence
            if len(seq) < seq_len:
                logger.warning(
                    f"{item.name}: sequence length {len(seq)} < {seq_len}; "
                    f"right-padding with Ns"
                )
                seq = seq + 'N' * (seq_len - len(seq))
            selected_bins = get_eval_bins(model, item.input_intervals, item.eval_intervals)
            attrs = get_attributions_for_element(
                model=model,
                input_seq=seq,
                selected_bins=selected_bins,
                tasks_to_plot_list=tasks_list,
                device=device_str,
                genome=genome,
                method=method,
                batch_size=1,
            )
            attributions_list.append(attrs)
            item = item._replace(sequence=seq) if hasattr(item, '_replace') else item
            successful_inputs.append((item, seq))
        except Exception as e:
            logger.exception(f"Failed for {item.name}: {e}")
        finally:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

    # Save outputs — only successful elements, so attributions/seqs/names are aligned
    final_attributions = np.concatenate(attributions_list, axis=0)
    names_list   = [item.name for item, _ in successful_inputs]
    input_seqs   = [seq       for _, seq  in successful_inputs]

    with open(os.path.join(experiment_dir, "attributions.pkl"), "wb") as f:
        pickle.dump(final_attributions, f)
    with open(os.path.join(experiment_dir, "input_seqs.pkl"), "wb") as f:
        pickle.dump(input_seqs, f)
    with open(os.path.join(experiment_dir, "element_names_list.pkl"), "wb") as f:
        pickle.dump(names_list, f)

    name_to_coord = {str(row[name_col or 'unique_ID']): idx
                     for idx, row in coord_data.iterrows()}
    mapping_df = pd.DataFrame({
        "coord_index":       [name_to_coord.get(n, -1) for n in names_list],
        "name":              names_list,
        "attribution_index": range(len(names_list)),
    })
    mapping_df.to_csv(os.path.join(experiment_dir, "attribution_mapping.csv"), index=False)

    logger.info(
        f"Done. Saved {len(attributions_list)} attributions to {experiment_dir}"
    )


if __name__ == "__main__":
    main()
