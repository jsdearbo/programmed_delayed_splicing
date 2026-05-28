"""
Step 0 — Fine-tune Borzoi on RNA-seq BigWig data.

Loads the pretrained Borzoi model via gReLU, builds genome-tiling train/val/test
datasets from BigWig files, and fine-tunes with the specified loss and optimizer.
The resulting checkpoint is used by all downstream steps as the attribution model.

Usage
-----
    python scripts/00_fine_tune.py --config config/example_config.yaml

Outputs (written to fine_tuning.save_dir)
------------------------------------------
    epoch=N-step=M.ckpt   — PyTorch Lightning checkpoint(s)
    last.ckpt             — checkpoint from final epoch
"""

import os
import sys
import logging
import argparse
import yaml
import wandb
import torch

import grelu.data.preprocess
import grelu.resources
import grelu.resources.wandb as grelu_wandb
from grelu.data.dataset import BigWigSeqDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.fine_tune_utils import (
    get_chromosome_lengths,
    generate_intervals,
    load_datasets,
    save_datasets,
    make_data_path,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def ensure_sizes_file_for_fasta(genome_fasta: str) -> str:
    """
    Ensure <genome_fasta>.sizes exists for gReLU CustomGenome usage.
    Builds it from <genome_fasta>.fai when missing.
    """
    fai_path = f"{genome_fasta}.fai"
    sizes_path = f"{genome_fasta}.sizes"

    if os.path.exists(sizes_path):
        return sizes_path

    if not os.path.exists(fai_path):
        raise FileNotFoundError(
            f"Missing FASTA index required to build sizes file: {fai_path}"
        )

    with open(fai_path, "r") as fin, open(sizes_path, "w") as fout:
        for line in fin:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                continue
            fout.write(f"{fields[0]}\t{fields[1]}\n")

    logger.info(f"Created local chromosome sizes file: {sizes_path}")
    return sizes_path


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Borzoi on BigWig RNA-seq data")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    return p.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def require_usable_cuda_if_requested(train_params: dict):
    """
    If config requests GPU devices, fail early with a clear compatibility message
    when CUDA cannot initialize (common with driver/runtime mismatch).
    """
    devices = train_params.get("devices")
    if devices is None:
        return

    if isinstance(devices, str) and devices.lower() == "cpu":
        return

    # Any explicit device list or integer count implies GPU intent.
    gpu_requested = isinstance(devices, (list, tuple, int))
    if not gpu_requested:
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this environment. Detected torch "
            f"{torch.__version__} (CUDA {torch.version.cuda}) with an incompatible/older NVIDIA driver. "
            "Install a PyTorch build compatible with your driver (e.g. CUDA 12.x build), "
            "or update the NVIDIA driver."
        )


def main():
    args = parse_args()
    cfg = load_config(args.config)

    ft = cfg.get("fine_tuning", {})
    if not ft:
        raise ValueError(
            "No 'fine_tuning' section found in config. "
            "Add a fine_tuning: block — see config/example_config.yaml."
        )

    genome = ft["genome"]
    species = ft["species"]
    sequence_length = int(ft["sequence_length"])
    bw_dir = ft["bw_dir"]
    bw_files = ft["bw_files"]
    lab_len = int(ft["lab_len"])
    bin_len = int(ft["bin_len"])
    model_params = ft["model_params"]
    train_params = dict(ft.get("train_params") or {})
    trans_func = ft.get("trans_func")
    tstride = ft.get("tstride", False)
    blacklist = ft.get("blacklist", "grelu")
    shift = ft.get("shift", 0)
    genome_fasta = ft.get("genome_fasta", cfg.get("genome_fasta"))
    project = ft.get("project", "elife_delay_splice")
    dataset_cache_dir = ft.get(
        "dataset_cache_dir",
        os.path.join(
            ft.get("save_dir", os.path.dirname(cfg.get("experiment_dir", "."))),
            "datasets",
        ),
    )

    # Ensure lr is float (YAML may parse 1e-6 as string in some versions)
    if "lr" in train_params:
        train_params["lr"] = float(train_params["lr"])

    # Fail fast on known driver/runtime mismatch before expensive dataset/model work.
    require_usable_cuda_if_requested(train_params)

    input_bigwig = [os.path.join(bw_dir, f) for f in bw_files]
    bw_name_list = [os.path.basename(f).split(".")[0] for f in bw_files]
    bw_name = "__".join(bw_name_list)

    genome_for_dataset = genome
    if genome_fasta:
        if not os.path.exists(genome_fasta):
            raise FileNotFoundError(
                f"Configured genome_fasta does not exist: {genome_fasta}"
            )
        ensure_sizes_file_for_fasta(genome_fasta)
        genome_for_dataset = genome_fasta
        logger.info(
            f"Using local FASTA for dataset genome checks: {genome_for_dataset}"
        )

    for f_path, f_name in zip(input_bigwig, bw_name_list):
        logger.info(f"Task: {f_name}  →  {f_path}")

    chromosome_lengths = get_chromosome_lengths(genome, genome_fasta=genome_fasta)
    intervals_df = generate_intervals(
        chromosome_lengths, sequence_length, tstride=tstride
    )

    if blacklist == "grelu":
        intervals_df = grelu.data.preprocess.filter_blacklist(
            intervals_df, genome=genome, window=1
        )
    elif blacklist == "calico":
        calico_bed = ft.get("calico_blacklist_path")
        if not calico_bed:
            raise ValueError(
                "blacklist='calico' requires fine_tuning.calico_blacklist_path in config."
            )
        intervals_df = grelu.data.preprocess.filter_blacklist(
            intervals_df, blacklist=calico_bed, window=1
        )

    train_chroms = "autosomes"
    val_chroms = ["chr10"]
    test_chroms = ["chr11"]

    intervals_train, intervals_val, intervals_test = grelu.data.preprocess.split(
        intervals_df,
        train_chroms=train_chroms,
        val_chroms=val_chroms,
        test_chroms=test_chroms,
    )

    intervals_train["length"] = intervals_train["end"] - intervals_train["start"]
    unique_count = len(intervals_train["length"].unique())
    logger.info(f"Unique interval lengths in train: {unique_count}")

    transformed_data_path = make_data_path(
        blacklist,
        tstride,
        shift,
        bw_name,
        trans_func,
        dataset_cache_dir=dataset_cache_dir,
    )

    dataset_files = {
        "train_dataset": os.path.join(transformed_data_path, "train_dataset.pkl"),
        "val_dataset": os.path.join(transformed_data_path, "val_dataset.pkl"),
        "test_dataset": os.path.join(transformed_data_path, "test_dataset.pkl"),
    }

    if all(os.path.exists(p) for p in dataset_files.values()):
        logger.info(f"Loading cached datasets from {transformed_data_path}")
        datasets = load_datasets(dataset_files)
    else:
        logger.info(f"Building datasets → {transformed_data_path}")
        dataset_params = {
            "bw_files": input_bigwig,
            "seq_len": sequence_length,
            "genome": genome_for_dataset,
            "end": "both",
            "rc": True,
            "label_len": lab_len,
            "bin_size": bin_len,
            "seed": 42,
            "augment_mode": "random",
            "label_transform_func": trans_func,
            "max_seq_shift": shift,
        }
        datasets = {
            "train_dataset": BigWigSeqDataset(
                intervals=intervals_train, **dataset_params
            ),
            "val_dataset": BigWigSeqDataset(intervals=intervals_val, **dataset_params),
            "test_dataset": BigWigSeqDataset(
                intervals=intervals_test, **dataset_params
            ),
        }
        save_datasets(datasets, dataset_files)

    data_params = {
        "train_dataset": datasets["train_dataset"],
        "val_dataset": datasets["val_dataset"],
        "test_dataset": datasets["test_dataset"],
    }

    model = grelu_wandb.load_model(
        project="borzoi",
        model_name=f"{species}_rep0",
    )

    logger.info(f"Loaded pretrained Borzoi ({species})")
    logger.info(f"Updating model hyperparameters")
    model.train_params.update(train_params)
    model.data_params.update(data_params)
    model.model_params.update(model_params)
    model.save_hyperparameters()

    run_name = train_params.get("name", ft.get("name", project))
    wandb.init(
        project=project,
        name=run_name,
        settings=wandb.Settings(start_method="thread"),
        config={
            "model_params": model_params,
            "train_params": train_params,
            "data_params": {
                k: str(v) for k, v in data_params.items() if k != "test_dataset"
            },
        },
    )

    model.tune_on_dataset(
        train_dataset=datasets["train_dataset"],
        val_dataset=datasets["val_dataset"],
    )

    logger.info("Fine-tuning complete.")


if __name__ == "__main__":
    main()
