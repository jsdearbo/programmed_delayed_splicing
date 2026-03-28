"""
Model inference, attribution calculation, and TF-MoDISco wrappers.
"""
import logging
import numpy as np
from typing import Any, List, Sequence

import grelu.interpret.score
import grelu.interpret.modisco
import grelu.transforms.prediction_transforms
from grelu.interpret.score import get_attributions

logger = logging.getLogger(__name__)


def get_attributions_for_element(
    model: Any,
    input_seq: str,
    selected_bins: Sequence[int],
    tasks_to_plot_list: List[str],
    device: str,
    genome: str = "hg38",
    method: str = "inputxgradient",
    **kwargs: Any,
) -> np.ndarray:
    """
    Compute per-nucleotide attributions for a single input sequence.

    Predictions are aggregated over `selected_bins` (output bins covering the
    element/transcript of interest) before computing gradients.

    Parameters
    ----------
    model :
        Loaded Borzoi model.
    input_seq : str
        One-hot encoded or string sequence of length seq_len.
    selected_bins : list of int
        Output bin indices corresponding to the genomic region of interest.
    tasks_to_plot_list : list of str
        Task names to aggregate over (e.g. RNA-seq track names).
    device : str
        'cpu' or 'cuda:0' etc.
    genome : str
        Genome assembly string passed to gReLU ('hg38' for human, 'mm10' for mouse).
    method : str
        Attribution method: 'inputxgradient' (default) or 'saliency'.
    **kwargs :
        Additional keyword arguments forwarded to get_attributions
        (e.g. batch_size).
    """
    element_specific_average = grelu.transforms.prediction_transforms.Aggregate(
        tasks=tasks_to_plot_list,
        positions=selected_bins,
        length_aggfunc="mean",
        task_aggfunc="mean",
        model=model,
    )
    attrs = get_attributions(
        model,
        seqs=[input_seq],
        genome=genome,
        prediction_transform=element_specific_average,
        device=device,
        method=method,
        seed=0,
        hypothetical=False,
        n_shuffles=10,
        batch_size=kwargs.get('batch_size', 4),
    )
    return attrs


def run_modisco_analysis(
    model: Any,
    input_seqs: Sequence[str],
    experiment_dir: str,
    attributions: np.ndarray,
    device: Any,
    genome: str = "hg38",
    window: int = 5000,
    sliding_window_size: int = 21,
    flank_size: int = 10,
) -> None:
    """
    Run TF-MoDISco motif discovery on a set of attribution arrays.

    Parameters
    ----------
    model :
        Loaded Borzoi model.
    input_seqs : list of str
        Input sequences aligned with attributions.
    experiment_dir : str
        Directory where MoDISco output (modisco_results.h5, forward.meme) is written.
    attributions : np.ndarray, shape (N, 4, L)
        Attribution arrays, one per sequence.
    device :
        Device string or index passed to gReLU.
    genome : str
        Genome assembly for gReLU ('hg38' for human, 'mm10' for mouse).
    window : int
        MoDISco seqlet search window (bp).
    sliding_window_size : int
        MoDISco sliding window size (default 21; seqlet max length = window + 2*flank_size).
    flank_size : int
        MoDISco flank size (default 10; equates to 41 bp max seqlet length).
    """
    grelu.interpret.modisco.run_modisco(
        model,
        seqs=input_seqs,
        genome=genome,
        method="completed",
        out_dir=experiment_dir,
        batch_size=1024,
        devices=device,
        num_workers=16,
        window=window,
        seed=0,
        attributions=attributions,
        sliding_window_size=sliding_window_size,
        flank_size=flank_size,
    )
    logger.info("TF-MoDISco analysis completed")
