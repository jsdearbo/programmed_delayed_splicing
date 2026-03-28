import os
import yaml
import pandas as pd
import pickle
import logging
import torch

import grelu.resources
import grelu.sequence.format
import grelu.interpret.score
import grelu.transforms.prediction_transforms
import grelu.lightning

logger = logging.getLogger(__name__)


def load_config(config_path: str):
    """Load a config from a YAML file path."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_experiment_directory(experiment_dir: str):
    """Create the experiment directory if it doesn't exist."""
    os.makedirs(experiment_dir, exist_ok=True)
    logger.info(f"Experiment directory set up at {experiment_dir}")


def validate_file(filepath: str, description: str):
    """Validate that a file exists."""
    if not os.path.exists(filepath):
        logger.error(f"{description} not found: {filepath}")
        raise FileNotFoundError(f"{description} not found: {filepath}")


def load_model(species: str, model_selection: str = None, checkpoint_path: str = None, device=None):
    """
    Load a Borzoi model.

    Parameters
    ----------
    species : str
        'human' or 'mouse'. Selects the pretrained model variant.
    model_selection : str
        'pretrained' (default) or 'fine_tuned'. Use 'fine_tuned' to load from a
        local checkpoint.
    checkpoint_path : str, optional
        Path to a fine-tuned checkpoint. Required when model_selection='fine_tuned'.
    device : optional
        torch.device or string. Defaults to CUDA if available, else CPU.
    """
    try:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)

        if model_selection == 'fine_tuned':
            if checkpoint_path is None:
                raise ValueError("checkpoint_path must be provided for fine-tuned models.")
            model = grelu.lightning.LightningModel.load_from_checkpoint(
                checkpoint_path, map_location=device
            )
            logger.info(f"Fine-tuned model loaded from {checkpoint_path} on {device}")
        else:
            model = grelu.resources.load_model(project="borzoi", model_name=f"{species}_rep0")
            logger.info(f"Pretrained Borzoi model for {species} loaded on {device}")

        return model.to(device), device

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def load_saved_attributions(file_path: str):
    """Load saved attributions from a pickle file."""
    try:
        with open(file_path, "rb") as f:
            attributions = pickle.load(f)
        logger.info(f"Loaded attributions from {file_path}")
        return attributions
    except Exception as e:
        logger.error(f"Failed to load attributions: {e}")
        raise


def load_gtf(file_path: str) -> pd.DataFrame:
    """Load a GTF file into a pandas DataFrame."""
    logger.info(f"Loading GTF file from {file_path}")
    column_names = [
        "seqname", "source", "feature", "start", "end",
        "score", "strand", "frame", "attribute"
    ]
    gtf_df = pd.read_csv(file_path, sep='\t', comment='#', names=column_names)
    return gtf_df


def process_gtf(gtf_df: pd.DataFrame) -> pd.DataFrame:
    """Extract gene/transcript attributes from a GTF DataFrame."""
    logger.info("Processing GTF DataFrame")
    gtf_df['gene_name'] = gtf_df['attribute'].str.extract(r'gene_name "(.*?)";')
    gtf_df['transcript_name'] = gtf_df['attribute'].str.extract(r'transcript_name "(.*?)";')
    gtf_df['exon_number'] = gtf_df['attribute'].str.extract(r'exon_number (\d+);')
    gtf_df['transcript_id'] = gtf_df['attribute'].str.extract(r'transcript_id "(.*?)";')
    gtf_df['tags'] = gtf_df['attribute'].str.findall(r'tag "([^"]+)"')
    gtf_df = gtf_df.rename(columns={'seqname': 'chrom'})
    return gtf_df


def load_and_process_gtf(gtf_file: str, gene_list: list = None) -> pd.DataFrame:
    """Load and process a GTF file, optionally subsetting to specified genes."""
    logger.info(f"Loading GTF file: {gtf_file}")
    gtf_df = load_gtf(gtf_file)
    gtf_df = process_gtf(gtf_df)
    if gene_list is not None:
        subset_gtf_df = gtf_df[gtf_df['gene_name'].isin(gene_list)]
        logger.info(f"Subset GTF contains {len(subset_gtf_df)} rows for {len(gene_list)} genes.")
        return subset_gtf_df
    else:
        logger.info(f"Returning all {len(gtf_df)} rows (no gene_list supplied).")
        return gtf_df


def select_longest_basic_transcripts(gtf_df: pd.DataFrame) -> pd.DataFrame:
    """Select the longest 'basic' tag transcript per gene."""
    exons = gtf_df[gtf_df['feature'] == 'exon'].copy()
    exons = exons[exons['tags'].apply(
        lambda tags: 'basic' in tags if isinstance(tags, list) else False
    )]
    exons['length'] = exons['end'] - exons['start']
    transcript_lengths = exons.groupby('transcript_id')['length'].sum()
    transcript_to_gene = exons.drop_duplicates('transcript_id')[['transcript_id', 'gene_name']]
    merged = transcript_lengths.reset_index().merge(transcript_to_gene, on='transcript_id')
    longest = merged.sort_values('length', ascending=False).drop_duplicates('gene_name')
    selected_gtf = gtf_df[gtf_df['transcript_id'].isin(longest['transcript_id'])]
    return selected_gtf


def load_meme(meme_file: str) -> dict:
    """Load motifs from a MEME file into a dict of {name: torch.Tensor (4, L)}."""
    with open(meme_file, "r") as f:
        lines = f.readlines()
    motifs = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("MOTIF"):
            motif_name = line.split()[1]
            while not lines[i].strip().startswith("letter-probability matrix"):
                i += 1
            header = lines[i].strip()
            w = int(header.split("w=")[1].split()[0])
            pwm = []
            i += 1
            for _ in range(w):
                vals = [float(x) for x in lines[i].strip().split()]
                if len(vals) == 4:
                    pwm.append(vals)
                i += 1
            pwm = torch.tensor(pwm, dtype=torch.float32).T
            if pwm.shape[0] == 4:
                motifs[motif_name] = pwm
        else:
            i += 1
    return motifs


def load_tasks(model) -> pd.DataFrame:
    """Load task metadata from a model's data_params."""
    return pd.DataFrame(model.data_params['tasks'])
