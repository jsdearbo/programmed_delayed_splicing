import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re

import grelu.visualize

from tangermeme.plot import plot_logo
from adjustText import adjust_text

logger = logging.getLogger(__name__)

def plot_predictions(
    preds: np.ndarray,
    tasks: pd.DataFrame,
    output_start: int,
    output_end: int,
    elements_to_highlight_df: pd.DataFrame,
    task_names: list,
    transcript_exons: pd.DataFrame,
    gene_of_interest: str,
    filename: str
):
    """
    Plot model predictions and save to file.
    """
    logger.info(f"Plotting predictions to {filename}")
    fig = grelu.visualize.plot_tracks(
        preds,
        start_pos=output_start,
        end_pos=output_end,
        titles=task_names,
        figsize=(20, 6),
        highlight_intervals=elements_to_highlight_df,
        facecolor="blue",
        annotations={f"{gene_of_interest} exons": transcript_exons}
    )
    fig.savefig(filename)
    plt.close(fig)

def plot_slices(
    mutation_tasks: list,
    ism_results: dict,
    tasks: pd.DataFrame,
    slices_df: pd.DataFrame,
    heatmap_dir: str,
    logo_dir: str,
    gene_of_interest: str,
    elements_to_highlight: str,
    chrom: str
):
    """
    Plot ISM slices as heatmaps and sequence logos.
    """
    logger.info("Plotting slices")
    for results in mutation_tasks:
        for slice_num, row in slices_df.iterrows():
            start_pos = row['plot_start']
            end_pos = row['plot_end']
            logger.debug(f"Plotting slice {slice_num} from {start_pos} to {end_pos} for result {results}")
            if start_pos < 0 or end_pos >= ism_results[results].shape[1]:
                logger.error(f"Invalid start or end position for slice {slice_num}: start_pos={start_pos}, end_pos={end_pos}")
                continue
            for method, dir_, figsize in [("heatmap", heatmap_dir, (20, 1.5)), ("logo", logo_dir, (15.5, 1.5))]:
                grelu.visualize.plot_ISM(
                    ism_results[results],
                    start_pos=start_pos,
                    end_pos=end_pos,
                    method=method,
                    figsize=figsize,
                    center=0 if method == "heatmap" else None
                )
                plt.title(f"Track: {tasks['sample'].loc[results]} slice {slice_num}", fontsize=14)
                plt.xlabel(
                    f"Log2FC expression level of {gene_of_interest} {elements_to_highlight} expression level: \n"
                    f"Point mutations from {chrom} {row['coord_start']:,} to {row['coord_end']:,}"
                )
                save_path = os.path.join(dir_, f"{elements_to_highlight}_slice_{slice_num}_track_{results}.png")
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()

def plot_full_window(
    mutation_tasks: list,
    ism_results: dict,
    tasks: pd.DataFrame,
    heatmap_dir: str,
    mutation_intervals: pd.DataFrame,
    gene_of_interest: str,
    elements_to_highlight: str,
    chrom: str
):
    """
    Plot full ISM window as a heatmap.
    """
    logger.info("Plotting full window")
    for results in mutation_tasks:
        grelu.visualize.plot_ISM(
            ism_results[results],
            start_pos=0,
            end_pos=ism_results[results].shape[1] - 1,
            method="heatmap",
            figsize=(20, 1.5),
            center=0
        )
        plt.title(f"Track: {tasks['sample'].loc[results]}", fontsize=14)
        plt.xlabel(
            f"Log2FC expression level of {gene_of_interest} {elements_to_highlight} expression level: \n"
            f"Point mutations from {chrom} {mutation_intervals.start.iloc[0]:,} to {mutation_intervals.end.iloc[0]:,}"
        )
        save_path = os.path.join(heatmap_dir, f"{elements_to_highlight}_full_track_{results}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def _plot_gene_elements(
    ax,
    elements: pd.DataFrame,
    start: int = None,
    end: int = None,
    color_exon: str = "blue",
    color_intron: str = "black",
    exon_text_color: str | None = "white",
    intron_text_color: str | None = "black"
):
    """
    Plot gene elements on an axis.
    Exons are truncated at UTR boundaries (strand-aware, no gap).
    Supplying None for exon_text_color or intron_text_color disables the text labels.
    """
    elements_list = list(elements.itertuples())

    def _maybe_text(x, y, s, color, **kwargs):
        if color is None:
            return
        ax.text(x, y, s, **kwargs)

    # Draw introns first
    for element in elements_list:
        element_name = str(element.element).lower()
        if "intron" in element_name:
            plot_start = max(element.start, start) if start is not None else element.start
            plot_end = min(element.end, end) if end is not None else element.end
            if element.end < (start if start is not None else element.start) or element.start > (end if end is not None else element.end):
                continue
            ax.plot([plot_start, plot_end], [0, 0], color=color_intron, linewidth=2)
            _maybe_text((plot_start + plot_end) / 2, 0.01, element.element, intron_text_color, ha='center', va='bottom')

    # Draw exons, truncating at UTRs
    for exon in elements_list:
        exon_name = str(exon.element).lower()
        # Treat as exon if name contains exon AND NOT utr
        if "exon" in exon_name and "utr" not in exon_name:
            plot_start = max(exon.start, start) if start is not None else exon.start
            plot_end = min(exon.end, end) if end is not None else exon.end
            if exon.end < (start if start is not None else exon.start) or exon.start > (end if end is not None else exon.end):
                continue
            
            # Find overlapping UTRs
            overlapping_utrs = [
                utr for utr in elements_list
                if ("utr" in str(getattr(utr, "element_type", "")).lower() or "utr" in str(utr.element).lower())
                and utr.chrom == exon.chrom
                and utr.transcript_id == exon.transcript_id
                and not (utr.end < plot_start or utr.start > plot_end)
            ]
            
            # If no overlap, draw full exon
            if not overlapping_utrs:
                rect_height = 1.0
                rect_y = -rect_height / 2
                ax.add_patch(plt.Rectangle((plot_start, rect_y), plot_end - plot_start + 1, rect_height,
                                           color=color_exon, linewidth=2))
                _maybe_text((plot_start + plot_end) / 2, 0, exon.element, exon_text_color, ha='center', va='center', fontsize=8)
            else:
                # Split exon into non-UTR segments
                exon_segments = [(plot_start, plot_end)]
                for utr in overlapping_utrs:
                    new_segments = []
                    for seg_start, seg_end in exon_segments:
                        if utr.start > seg_start:
                            new_segments.append((seg_start, utr.start - 1))
                        if utr.end < seg_end:
                            new_segments.append((utr.end + 1, seg_end))
                    exon_segments = [seg for seg in new_segments if seg[0] <= seg[1]]
                
                for seg_start, seg_end in exon_segments:
                    if seg_start <= seg_end:
                        ax.add_patch(plt.Rectangle((seg_start, -0.5), seg_end - seg_start + 1, 1.0,
                                                   color=color_exon, linewidth=2))
                        _maybe_text((seg_start + seg_end) / 2, 0, exon.element, exon_text_color, ha='center', va='center', fontsize=8)

    # Draw UTRs last
    for utr in elements_list:
        utr_name = str(utr.element).lower()
        utr_type = str(getattr(utr, "element_type", "")).lower()
        if "utr" in utr_type or "utr" in utr_name:
            plot_start = max(utr.start, start) if start is not None else utr.start
            plot_end = min(utr.end, end) if end is not None else utr.end
            if utr.end < (start if start is not None else utr.start) or utr.start > (end if end is not None else utr.end):
                continue
            rect_height = 0.5
            rect_y = -rect_height / 2
            ax.add_patch(plt.Rectangle((plot_start, rect_y), plot_end - plot_start + 1, rect_height,
                                       color=color_exon, linewidth=1, edgecolor='black', zorder=10))
            _maybe_text((plot_start + plot_end) / 2, 0, 'utr', exon_text_color, ha='center', va='center', fontsize=8, zorder=11)

def plot_gene_map(
    elements_df: pd.DataFrame,
    ax=None,
    title: str = "",
    xlim: tuple = None,
    start: int = None,
    end: int = None,
    color_exon: str = "#0068fa",
    color_intron: str = "#969696",
    exon_text_color: str = "white",
    intron_text_color: str = "black",
    highlight_area: tuple = None,
    orientation: str = "RNA",
    save_path: str = None,
    figsize: tuple = (20, 1),
    show_xaxis_label: bool = True,
):
    """
    Plot a gene map on a provided axis or create a new one.
    Optionally highlight a region and/or save to file.
    orientation: "genomic" (default) or "RNA". If "RNA", '-' strand is flipped to 5'→3'.
    """
    own_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        own_fig = True

    if elements_df.empty:
        ax.text(0.5, 0.5, "No elements found for transcript", ha='center', va='center')
        ax.set_title(title)
        return

    # Determine strand and whether to flip x-axis
    strand = elements_df['strand'].iloc[0] if 'strand' in elements_df.columns else '+'
    flip_x = orientation == "RNA" and strand == '-'

    # Determine plotting region
    if start is None or end is None:
        if xlim is not None:
            start, end = xlim
        else:
            start = elements_df['start'].min()
            end = elements_df['end'].max()

    chrom = elements_df["chrom"].iloc[0] if "chrom" in elements_df.columns else ""
    def fmt(val): return f"{val:,}" if val is not None else "?"

    # Prepare axis label
    if highlight_area is not None:
        highlight_start, highlight_end = highlight_area
        left, right = sorted((highlight_start, highlight_end))
        x_axis_label = f'{chrom}:{fmt(start)}-{fmt(end)} | Highlight: {fmt(left)}-{fmt(right)}'
    else:
        left = right = None
        x_axis_label = f'{chrom}:{fmt(start)}-{fmt(end)}'

    # Plot elements
    _plot_gene_elements(
        ax, elements_df, start=start, end=end,
        color_exon=color_exon, color_intron=color_intron,
        exon_text_color=exon_text_color, intron_text_color=intron_text_color
    )

    # Highlight
    if highlight_area is not None:
        ax.axvspan(left, right, color='red', alpha=0.35, zorder=10.5)

    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(start, end)

    if flip_x:
        ax.set_xlim(ax.get_xlim()[1], ax.get_xlim()[0])

    ax.set_yticks([])
    ax.set_yticklabels([])
    if show_xaxis_label:
        ax.set_xlabel(x_axis_label)
    ax.set_xticks([])
    ax.set_title(title, fontsize=24)

    for spine in ax.spines.values():
        spine.set_visible(False)

    if save_path is not None and own_fig:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

def plot_gene_maps(
    elements_df: pd.DataFrame,
    save_dir: str,
    transcript_id: str = None,
    start_mut: int = None,
    end_mut: int = None,
    slices_df: pd.DataFrame = None
):
    """
    Plot gene maps for all or selected transcripts.
    """
    logger.info("Plotting gene maps")
    if transcript_id is not None:
        elements_df = elements_df[elements_df['transcript_id'] == transcript_id]
    for tid in elements_df['transcript_id'].unique():
        df_tid = elements_df[elements_df['transcript_id'] == tid]
        # Full context
        plot_gene_map(
            df_tid,
            os.path.join(save_dir, f"{tid}_full_length_track.png"),
            f"Gene map of {tid}"
        )
        # Mutation region
        if start_mut is not None and end_mut is not None:
            plot_gene_map(
                df_tid,
                os.path.join(save_dir, f"{tid}_ism_length_track.png"),
                f"Gene map of {tid}",
                xlim=(start_mut, end_mut),
                start=start_mut,
                end=end_mut,
                color_exon="teal",
                exon_text_color="black"
            )
        # Slices
        if slices_df is not None:
            for slice_num, row in slices_df.iterrows():
                plot_gene_map(
                    df_tid,
                    os.path.join(save_dir, f"{tid}_slice_{slice_num}.png"),
                    f"Gene map of {tid}",
                    xlim=(row['coord_start'], row['coord_end']),
                    start=row['coord_start'],
                    end=row['coord_end']
                )

def plot_logo_and_optional_gene_map(
    arr: np.ndarray,
    row: pd.Series,
    seq_annotations: pd.DataFrame,
    logo_plot_dir: str,
    elements_df: pd.DataFrame = None,
    indexing_df: pd.DataFrame = None,
    tensor_center: int = 262144,
    PLOT_WINDOW: int = 104
):
    """
    Plot and save a logo plot for a single row.
    If elements_df and indexing_df are provided, also plot a gene map as a subplot.
    """
    seq_name = row.unique_ID
    plt_start = int(row.start) - (PLOT_WINDOW // 2)
    plt_end = int(row.start) + (PLOT_WINDOW // 2)
    plt_start = max(0, plt_start)
    plt_end = min(arr.shape[1], plt_end)
    print(f"Plotting logo for {seq_name} from {plt_start} to {plt_end}")

    if elements_df is not None and indexing_df is not None:
        # --- Get gene map info ---
        transcript_name = seq_name.split('_')[0]
        gene_name = transcript_name.split('-')[0]
        seq_elements_df = elements_df[elements_df['transcript_name'] == transcript_name]
        intron_start = indexing_df.loc[indexing_df['unique_ID'] == seq_name, 'start'].values[0]
        intron_end = indexing_df.loc[indexing_df['unique_ID'] == seq_name, 'end'].values[0]
        intron_midpt = (intron_start + intron_end) // 2
        if row.strand == '+':
            highlight_start = intron_midpt + (int(row.start) - tensor_center)
            highlight_end = intron_midpt + (int(row.end) - tensor_center)
        elif row.strand == '-':
            highlight_start = intron_midpt - (int(row.end) - tensor_center)
            highlight_end = intron_midpt - (int(row.start) - tensor_center)
        highlight_bounds = (highlight_start, highlight_end)

        # --- Create figure with two subplots ---
        fig, axs = plt.subplots(2, 1, figsize=(20, 6), gridspec_kw={'height_ratios': [2, 1]})

        # Logo plot (top)
        plot_logo(arr, ax=axs[0], start=plt_start, end=plt_end, annotations=seq_annotations)
        axs[0].set_title(f"Motif: {getattr(row, 'pattern_label', getattr(row, 'motif_name', 'unknown'))} | Seq: {seq_name} | Score: {row.score:.2f}")
        axs[0].set_xlabel("Position")
        axs[0].set_ylabel("Attribution Score")

        # Gene map (bottom)
        plot_gene_map(
            elements_df=seq_elements_df,
            ax=axs[1],
            title=f"{gene_name}",
            highlight_area=highlight_bounds
        )

        plt.tight_layout()
        plt.savefig(f"{logo_plot_dir}/logo_and_map_{seq_name}.png")
        plt.close(fig)
        print(f"Logo+gene map plot saved for {seq_name} at: \n{logo_plot_dir}/logo_and_map_{seq_name}.png")
    else:
        # Only logo plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        plot_logo(arr, ax=ax, start=plt_start, end=plt_end, annotations=seq_annotations)
        ax.set_title(f"Motif: {getattr(row, 'pattern_label', getattr(row, 'motif_name', 'unknown'))} | Seq: {seq_name} | Score: {row.score:.2f}")
        ax.set_xlabel("Position")
        ax.set_ylabel("Attribution Score")
        plt.tight_layout()
        plt.savefig(f"{logo_plot_dir}/logo_plot_{seq_name}.png")
        plt.close(fig)
        print(f"Logo plot saved for {seq_name} at: \n{logo_plot_dir}/logo_plot_{seq_name}.png")
   
def plot_cosi_boxplot(
    dataframes: dict,
    save_name: str = "cosi_boxplot.png",
    value_label: str = "CoSI",
    plot_dir: str = ".",
    unique_IDs: list = None # list of intron names to highlight, if any
):
    """
    Create a boxplot of CoSI values for each DataFrame.
    """
    cosi_data = [df['CoSI'] for df in dataframes.values()]
    df_names = list(dataframes.keys())
    timepoints = [name.split('_')[1] for name in df_names]

    flattened_data = np.concatenate(cosi_data)
    labels = np.concatenate([[tp] * len(df) for tp, df in zip(timepoints, cosi_data)])

    palette = sns.color_palette("coolwarm", len(df_names))
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(x=labels, y=flattened_data, palette=palette, ax=ax, showfliers=False)  # Disable default outliers


    # Add uniform outlier dots
    for i, (timepoint, data) in enumerate(zip(timepoints, cosi_data)):
        q1, q3 = data.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = data[(data < lower) | (data > upper)]
        # Jitter x positions for visibility
        x_jitter = i + np.random.uniform(-0.15, 0.15, size=len(outliers))
        ax.scatter(x_jitter, outliers, color='black', s=40, alpha=0.7, edgecolor='none', zorder=10)

    ax.set_title(f'{value_label}', fontsize=22)
    ax.set_xlabel('Timepoint', fontsize=20)
    ax.set_ylabel('CoSI Value', fontsize=20)
    plt.xticks(rotation=45, ha='right', fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, save_name))
    plt.close()
    logger.info(f"Saved boxplot to {os.path.join(plot_dir, save_name)}")

def plot_cosi_boxplot_from_df(
    df: pd.DataFrame,
    save_name: str = "cosi_boxplot.png",
    value_label: str = "CoSI",
    plot_dir: str = ".",
    unique_IDs: list = None # list of intron names to highlight, if any
):
    """
    Create a boxplot of CoSI values from an aggregate DataFrame of all timepoints,
    overlaying only outlier points with jitter and custom x labels, sorted by timepoint.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os
    import re

    # Find all CoSI columns (timepoints)
    cosi_cols = [col for col in df.columns if col.startswith("CoSI")]
    if not cosi_cols:
        raise ValueError("No columns starting with 'CoSI' found in DataFrame.")

    # Extract minute values and sort columns accordingly
    def extract_minutes(tp):
        m = re.search(r'_(\d+)_', tp)
        return int(m.group(1)) if m else float('inf')
    cosi_cols_sorted = sorted(cosi_cols, key=extract_minutes)

    # Melt to long format using sorted columns
    long_df = df.melt(
        id_vars=[col for col in df.columns if col not in cosi_cols],
        value_vars=cosi_cols_sorted,
        var_name="timepoint",
        value_name="CoSI"
    ).dropna(subset=["CoSI"])

    # Optionally highlight specific introns
    if unique_IDs is not None:
        long_df["highlight"] = long_df["unique_ID"].isin(unique_IDs)
    else:
        long_df["highlight"] = False

    # Get sorted unique timepoints and x labels
    unique_timepoints = [col for col in cosi_cols_sorted]
    x_labels = [str(extract_minutes(tp)) for tp in unique_timepoints]

    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(
        data=long_df,
        x="timepoint",
        y="CoSI",
        order=unique_timepoints,
        palette="coolwarm"
    )

    # Calculate and plot outliers only, with jitter
    for i, timepoint in enumerate(unique_timepoints):
        tp_data = long_df[long_df["timepoint"] == timepoint]
        q1 = tp_data["CoSI"].quantile(0.25)
        q3 = tp_data["CoSI"].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = tp_data[(tp_data["CoSI"] < lower) | (tp_data["CoSI"] > upper)]
        # Add jitter to x positions
        x_jitter = i + np.random.uniform(-0.15, 0.15, size=len(outliers))
        ax.scatter(
            x_jitter,
            outliers["CoSI"],
            color="black",
            alpha=0.6,
            zorder=10
        )
        # Overlay highlighted outliers (if any)
        if unique_IDs is not None:
            highlighted = outliers[outliers["highlight"]]
            if not highlighted.empty:
                x_jitter_hl = i + np.random.uniform(-0.15, 0.15, size=len(highlighted))
                ax.scatter(
                    x_jitter_hl,
                    highlighted["CoSI"],
                    color="red",
                    s=60,
                    edgecolor="white",
                    zorder=11
                )

    ax.set_title(f'{value_label}', fontsize=22)
    ax.set_xlabel('Timepoint', fontsize=20)
    ax.set_ylabel('CoSI Value', fontsize=20)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, save_name))
    plt.close()
    print(f"Saved boxplot to {os.path.join(plot_dir, save_name)}")
    
def get_transcript_and_highlight_bounds(elements_df, seq_name, row_start, row_end, tensor_center=262144):
    """
    Given elements_df, a sequence name, and row start/end, return:
    """
    transcript_name = seq_name.split('_')[0]
    seq_elements_df = elements_df[elements_df['transcript_name'] == transcript_name]
    print(f"Transcript: {transcript_name}, Elements: {seq_elements_df.shape[0]}")
    
    tscript_start = seq_elements_df['start'].min()
    tscript_end = seq_elements_df['end'].max()
    print(f"Transcript Start: {tscript_start}, Transcript End: {tscript_end}")
    tscript_midpt = (tscript_start + tscript_end) // 2
    print(f"Transcript Midpoint: {tscript_midpt}")
    highlight_start = tscript_midpt + (row_start - tensor_center)
    highlight_end = tscript_midpt + (row_end - tensor_center)
    highlight_bounds = (highlight_start, highlight_end)
    
    return transcript_name, seq_elements_df, highlight_bounds

def plot_motif_scatter(
    df,
    topN=5,
    dot_color='royalblue',
    title='Motif Representation: Control vs Primary',
    save_path=None,
    motif_type_col=None
):
    import matplotlib.pyplot as plt

    plt.style.use('seaborn-v0_8-white')
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'legend.fontsize': 13,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'figure.dpi': 150,
        'axes.linewidth': 1.2,
        'lines.markersize': 8,
        'lines.linewidth': 2,
    })

    fig, ax = plt.subplots(figsize=(8, 8))

    if motif_type_col and isinstance(dot_color, dict):
        for motif_type, color in dot_color.items():
            subdf = df[df[motif_type_col] == motif_type]
            ax.scatter(
                subdf['percent_match_ctrl'],
                subdf['percent_match_primary'],
                color=color, edgecolor='white', s=90, alpha=0.85, label=motif_type, zorder=2
            )
    else:
        ax.scatter(
            df['percent_match_ctrl'],
            df['percent_match_primary'],
            color=dot_color, edgecolor='white', s=90, alpha=0.85, label='Motifs', zorder=2
        )

    ax.plot([0, 100], [0, 100], 'k--', lw=1.5, label='y = x', zorder=1)
    #ax.plot([0, 22], [0, 22], 'k--', lw=1.5, label='y = x', zorder=1)

    deviation = (df['percent_match_primary'] - df['percent_match_ctrl']).abs()
    top = df.loc[deviation.nlargest(topN).index]

    texts = []

    import re
    id_pattern = re.compile(r'_(\d+)$')

    for _, row in top.iterrows():
        motif_name = str(row['motif_name'])
        m = id_pattern.search(motif_name)
        motif_label = f"motif {m.group(1)}" if m else "motif"
        texts.append(
            ax.text(
                row['percent_match_ctrl'],
                row['percent_match_primary'],
                motif_label,
                fontsize=12, fontweight='bold', color='black', alpha=0.95, zorder=3
            )
        )

    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle='-', color='gray', lw=1.2, alpha=0.7),
        force_explode=(-30, 10),
    )

    ax.set_title(title, pad=15)
    ax.set_xlabel('Percent of Control Introns with Motif', labelpad=10)
    ax.set_ylabel('Percent of Primary Introns with Motif', labelpad=10)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.grid(True, linestyle=':', linewidth=1, alpha=0.5)
    #ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved scatter plot to:\n {save_path}")
    else:
        plt.show()
  
def _plot_read_density_tracks(
    ax_gene,
    gs,
    n_tracks,
    timepoints,
    bw_files,
    chrom,
    region_start,
    region_end,
    y_scale,
    flip_x,
    highlight_spans=None,
    gene_label=None,
    offset=1
):
    """
    Helper to plot read density tracks below a gene map.
    highlight_spans: list of (left, right) tuples or None, one per track (or single tuple for all)
    gene_label: string for error messages
    Returns: last ax used (for xlabel etc)
    """
    colors = plt.cm.viridis(np.linspace(0, 1, n_tracks))
    last_ax = None
    for i, (bw_key, color) in enumerate(zip(timepoints, colors)):
        ax = ax_gene.figure.add_subplot(gs[i+offset, 0], sharex=ax_gene)
        file_path = bw_files[bw_key]
        import pyBigWig
        bw = pyBigWig.open(file_path)
        try:
            values = np.array(bw.values(str(chrom), region_start, region_end))
        except RuntimeError as e:
            print(f"Error fetching values for gene {gene_label if gene_label else chrom} in {bw_key}: {e}")
            bw.close()
            continue
        # Apply log scale if requested
        if y_scale == "log":
            values = np.log(values + 1e-9)
        elif y_scale == "log1p":
            values = np.log1p(values)
        # Flip values for '-' strand in RNA orientation
        if flip_x:
            values = values[::-1]
            x_range = range(region_end-1, region_start-1, -1)
        else:
            x_range = range(region_start, region_end)
            
        # Highlight region if provided
        if highlight_spans is not None:
            # highlight_spans can be a list of spans or a single tuple
            if isinstance(highlight_spans, list) and len(highlight_spans) > 0 and isinstance(highlight_spans[0], tuple):
                # If only one track, highlight all spans
                if n_tracks == 1:
                    for span in highlight_spans:
                        left, right = span
                        ax.axvspan(left, right, color='orange', alpha=0.4, zorder=0)
                else:
                    span = highlight_spans[i] if i < len(highlight_spans) else None
                    if span is not None:
                        left, right = span
                        ax.axvspan(left, right, color='orange', alpha=0.4, zorder=0)
            else:
                left, right = highlight_spans
                ax.axvspan(left, right, color='orange', alpha=0.4, zorder=0)
        # Plot
        timepoint_label = f"{bw_key.split('_')[-1]} min"
        ax.plot(x_range, values, color=color)
        ax.fill_between(x_range, values, color=color, alpha=0.3)
        ax.set_ylabel(timepoint_label, fontsize=22)
        ax.grid(True)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        if i < n_tracks - 1:
            ax.set_xticklabels([])
        if flip_x:
            ax.set_xlim(region_end, region_start)
        bw.close()
        last_ax = ax
    return last_ax

def _select_elements_for_sequence(elements_df: pd.DataFrame, seq_name: str) -> pd.DataFrame:
    """
    Resolve elements for a given seq_name using a robust, ordered strategy:

    1) Exact match on transcript_name (after stripping _<digits>)
    2) Startswith match on transcript_name
    3) Exact match on gene_name
    4) Fallback: substring match on gene_name
    """

    if not isinstance(seq_name, str) or not seq_name:
        return pd.DataFrame()

    # Strip trailing _<digits>
    base = re.sub(r"_(\d+)$", "", seq_name)

    subset = pd.DataFrame()

    # Strategy A: exact transcript match
    if "transcript_name" in elements_df.columns:
        subset = elements_df[elements_df["transcript_name"] == base].copy()
        if not subset.empty:
            logging.info(f"[select_elements] Strategy A matched transcript_name '{base}' ({len(subset)} rows)")
            return subset

    # Strategy A2: startswith transcript (for safety)
    if "transcript_name" in elements_df.columns:
        subset = elements_df[elements_df["transcript_name"].str.startswith(base, na=False)].copy()
        if not subset.empty:
            logging.info(f"[select_elements] Strategy A2 matched transcript_name startswith '{base}' ({len(subset)} rows)")
            return subset

    # Strategy B: exact gene match
    if "gene_name" in elements_df.columns:
        subset = elements_df[elements_df["gene_name"] == base].copy()
        if not subset.empty:
            logging.info(f"[select_elements] Strategy B matched gene_name '{base}' ({len(subset)} rows)")
            return subset

    # Strategy C: fallback — gene contains base
    if "gene_name" in elements_df.columns:
        subset = elements_df[elements_df["gene_name"].str.contains(base, na=False)].copy()
        if not subset.empty:
            logging.info(f"[select_elements] Strategy C matched gene_name contains '{base}' ({len(subset)} rows)")
            return subset

    logging.warning(f"[select_elements] No elements found for '{seq_name}'")
    return pd.DataFrame()

def plot_gene_map_and_read_densities(
    elements_df: pd.DataFrame,
    bw_files: dict,
    gene: str,
    tscript_name: str,
    timepoints: list = None,
    highlight_area: tuple = None,
    save_path: str = None,
    orientation: str = "RNA",
    figsize: tuple = (20, 2.5),
    y_scale: str = "linear",
    track_highlight: str = None,  # e.g. "intron_3"
    start_region_name: str = None,  # e.g. "exon_2" or "intron_3"
    end_region_name: str = None,    # e.g. "exon_3" or "intron_4"
    axes=None                        # for plotting on existing axes
):
    """
    Plot a gene map (top) and a series of read density histograms (bottom) for a gene or region.
    If start_region_name and/or end_region_name are provided, plot only that region.
    If axes is provided, use those axes for plotting (axes[0]=gene map, axes[1:]=tracks).
    """
    import matplotlib.pyplot as plt

    # Subset to transcript
    elements_df = elements_df[elements_df['transcript_name'] == tscript_name]
    strand = elements_df['strand'].iloc[0] if 'strand' in elements_df.columns else '+'
    flip_x = orientation == "RNA" and strand == '-'
    print(f"Using strand '{strand}' for orientation '{orientation}'")

    chrom = elements_df['chrom'].iloc[0]

    # Determine plotting region
    if start_region_name is not None:
        start_row = elements_df[elements_df['element'] == start_region_name]
        if start_row.empty:
            raise ValueError(f"Start region '{start_region_name}' not found in elements_df for transcript {tscript_name}")
        plot_start = int(start_row['end'].iloc[0]) if strand == '-' else int(start_row['start'].iloc[0])
    else:
        plot_start = elements_df['start'].min()

    if end_region_name is not None:
        end_row = elements_df[elements_df['element'] == end_region_name]
        if end_row.empty:
            raise ValueError(f"End region '{end_region_name}' not found in elements_df for transcript {tscript_name}")
        plot_end = int(end_row['start'].iloc[0]) if strand == '-' else int(end_row['end'].iloc[0])
    else:
        plot_end = elements_df['end'].max()

    plot_start, plot_end = min(plot_start, plot_end), max(plot_start, plot_end)
    print(f"Plotting region: {chrom}:{plot_start:,}-{plot_end:,}")

    if timepoints is None:
        timepoints = sorted(bw_files.keys())
    n_tracks = len(timepoints)

    height_ratios = [0.5] + [1]*n_tracks

    # --- Figure and axes setup ---
    if axes is None:
        fig = plt.figure(figsize=(figsize[0], figsize[1] + 2 * n_tracks))
        gs = fig.add_gridspec(n_tracks + 1, 1, height_ratios=height_ratios)
        axes = [fig.add_subplot(gs[0, 0])]
        axes += [fig.add_subplot(gs[i+1, 0], sharex=axes[0]) for i in range(n_tracks)]
    else:
        if len(axes) != n_tracks + 1:
            raise ValueError(f"Number of axes ({len(axes)}) does not match number of tracks + 1 ({n_tracks + 1})")
        fig = axes[0].figure

    # Subset elements_df to only those overlapping the region
    elements_df_sub = elements_df[(elements_df['end'] >= plot_start) & (elements_df['start'] <= plot_end)].copy()
    plot_gene_map(
        elements_df=elements_df_sub,
        ax=axes[0],
        title=f"{gene}",
        xlim=(plot_start, plot_end),
        highlight_area=highlight_area,
        orientation=orientation,
        start=plot_start,
        end=plot_end
    )
    axes[0].set_xticklabels([])

    # Prepare highlight spans for each track (intron highlight)
    if track_highlight is not None:
        if isinstance(track_highlight, list):
            highlight_spans = []
            for elem in track_highlight:
                elem_row = elements_df.loc[elements_df['element'] == elem]
                if elem_row.empty:
                    continue
                elem_start = elem_row['start'].values[0]
                elem_end = elem_row['end'].values[0]
                if flip_x:
                    left, right = max(elem_start, elem_end), min(elem_start, elem_end)
                else:
                    left, right = min(elem_start, elem_end), max(elem_start, elem_end)
                highlight_spans.append((left, right))
        else:
            intron_start = elements_df.loc[elements_df['element'] == track_highlight, 'start'].values[0]
            intron_end = elements_df.loc[elements_df['element'] == track_highlight, 'end'].values[0]
            if flip_x:
                highlight_left, highlight_right = max(intron_start, intron_end), min(intron_start, intron_end)
            else:
                highlight_left, highlight_right = min(intron_start, intron_end), max(intron_start, intron_end)
            highlight_spans = (highlight_left, highlight_right)
    else:
        highlight_spans = None

    # Plot read density tracks
    last_ax = None
    for i, (bw_key, ax_rd) in enumerate(zip(timepoints, axes[1:])):
        file_path = bw_files[bw_key]
        import pyBigWig
        bw = pyBigWig.open(file_path)
        try:
            values = np.array(bw.values(str(chrom), plot_start, plot_end))
        except RuntimeError as e:
            print(f"Error fetching values for gene {gene if gene else chrom} in {bw_key}: {e}")
            bw.close()
            continue
        if y_scale == "log":
            values = np.log(values + 1e-9)
        elif y_scale == "log1p":
            values = np.log1p(values)
        if flip_x:
            values = values[::-1]
            x_range = range(plot_end-1, plot_start-1, -1)
        else:
            x_range = range(plot_start, plot_end)
        # Highlight region if provided
        if highlight_spans is not None:
            if isinstance(highlight_spans, list) and len(highlight_spans) > 0 and isinstance(highlight_spans[0], tuple):
                if n_tracks == 1:
                    for span in highlight_spans:
                        left, right = span
                        ax_rd.axvspan(left, right, color='orange', alpha=0.4, zorder=0)
                else:
                    span = highlight_spans[i] if i < len(highlight_spans) else None
                    if span is not None:
                        left, right = span
                        ax_rd.axvspan(left, right, color='orange', alpha=0.4, zorder=0)
            else:
                left, right = highlight_spans
                ax_rd.axvspan(left, right, color='orange', alpha=0.4, zorder=0)
        timepoint_label = f"{bw_key.split('_')[-1]} min"
        ax_rd.plot(x_range, values, color=plt.cm.viridis(i / n_tracks))
        ax_rd.fill_between(x_range, values, color=plt.cm.viridis(i / n_tracks), alpha=0.3)
        ax_rd.set_ylabel(timepoint_label, fontsize=22)
        ax_rd.grid(True)
        ax_rd.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        if i < n_tracks - 1:
            ax_rd.set_xticklabels([])
        if flip_x:
            ax_rd.set_xlim(plot_end, plot_start)
        bw.close()
        last_ax = ax_rd

    if last_ax is not None:
        last_ax.set_xlabel("Position")

    # Always save if save_path is provided
    if save_path:
        print(f"Saving plot to {save_path}")
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        return

    # Only show if axes is None (i.e., standalone plot)
    if axes is None:
        fig.suptitle(f"Gene Map and Signal Intensity for {gene} at Each Time Point in {y_scale} scale")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()
    else:
        return fig, axes

def plot_logo_gene_map_and_read_densities(
    arr: np.ndarray,
    row: pd.Series,
    seq_annotations: pd.DataFrame,
    elements_df: pd.DataFrame,
    indexing_df: pd.DataFrame,
    logo_plot_dir: str,
    bw_files: dict | None = None,
    tensor_center: int = 262144,
    PLOT_WINDOW: int = 104,
    timepoints: list = None,
    orientation: str = "RNA",
    y_scale: str = "linear",
    figsize: tuple = (20, 8),
    start_region_name: str = None,
    end_region_name: str = None,
    attr_respect_to: str = "intron_only",
):
    """
    Plot a logo plot (top), gene map (middle), and optionally read density histograms (bottom) for a single row.
    If bw_files is None or empty, only the logo and gene map are plotted.
    """
    
    logger.info(
        "[PLOT DEBUG] intron=%s example_index=%s row_start=%s row_end=%s arr_shape=%s slice_len=%s",
        row.get("unique_ID"), row.get("example_index"),
        row["start"], row["end"],
        arr.shape,
        int(row["end"] - row["start"])
    )

    seq_name = row.unique_ID
    plt_start = int(row.start) - (PLOT_WINDOW // 2)
    plt_end = int(row.start) + (PLOT_WINDOW // 2)
    plt_start = max(0, plt_start)
    #plt_end = min(arr.shape[1], plt_end)
    #plt_end = min(arr.shape[0], plt_end)

    # 1. Select Elements
    seq_elements_df = _select_elements_for_sequence(elements_df, seq_name)
    if seq_elements_df.empty:
        # Fallback to display even if empty, just to avoid crash, but log error
        logger.error(f"No elements found for {seq_name}. Check naming consistency.")
        # Create dummy df to allow plotting to proceed (will show 'No elements found')
        seq_elements_df = pd.DataFrame(columns=['chrom','start','end','strand','transcript_name'])
        transcript_name = seq_name
        gene_name = seq_name
    else:
        transcript_name = seq_elements_df['transcript_name'].iloc[0]
        gene_name = seq_elements_df['gene_name'].iloc[0] if 'gene_name' in seq_elements_df.columns else transcript_name

    # 2. Get Highlight Bounds (Motif/Seqlet)
    # Ensure seq_name exists in indexing_df
    if seq_name in indexing_df['unique_ID'].values:
        row_idx = indexing_df.loc[indexing_df['unique_ID'] == seq_name].iloc[0]
        
        # Determine the "center" of the genomic map relative to the model input
        # Since input sequences are CENTERED on the RBP peak (element_start/element_end),
        # we MUST use the element's center as the reference anchor, 
        # NOT the transcript's center (even if plotting the whole transcript).
        
        if 'element_start' in row_idx and pd.notna(row_idx['element_start']):
            # Use the preserved element bounds from the CSV
            center_ref_point = (int(row_idx['element_start']) + int(row_idx['element_end'])) // 2
        else:
            # Fallback (e.g., if attr_respect_to was intron_only, 'start' IS the element)
            center_ref_point = (int(row_idx['start']) + int(row_idx['end'])) // 2
             
    else:
        logger.error(f"{seq_name} not in indexing_df. Cannot calc highlight.")
        center_ref_point = 0 
        
    if row.strand == '+':
        highlight_start = center_ref_point + (int(row.start) - tensor_center)
        highlight_end = center_ref_point + (int(row.end) - tensor_center)
    elif row.strand == '-':
        highlight_start = center_ref_point - (int(row.end) - tensor_center)
        highlight_end = center_ref_point - (int(row.start) - tensor_center)
    else:
        highlight_start, highlight_end = center_ref_point, center_ref_point
    highlight_bounds = (highlight_start, highlight_end)

    # 3. Get Element Highlight Bounds (RBP site)
    element_highlight_bounds = None
    if seq_name in indexing_df['unique_ID'].values:
        row_idx = indexing_df.loc[indexing_df['unique_ID'] == seq_name].iloc[0]
        
        if 'element_start' in row_idx and pd.notna(row_idx['element_start']):
             el_s = int(row_idx['element_start'])
             el_e = int(row_idx['element_end'])
             element_highlight_bounds = (min(el_s, el_e), max(el_s, el_e))
        else:
             el_s = int(row_idx['start'])
             el_e = int(row_idx['end'])
             element_highlight_bounds = (min(el_s, el_e), max(el_s, el_e))

    # 4. Determine Plot Region (Genome Coords)
    if not seq_elements_df.empty:
        chrom = seq_elements_df['chrom'].iloc[0]
        plot_start = seq_elements_df['start'].min()
        plot_end = seq_elements_df['end'].max()
    else:
        # Fallback bounds based on highlight if no elements found
        chrom = "Unknown"
        plot_start = highlight_start - 1000
        plot_end = highlight_end + 1000

    # Tracks setup
    if bw_files:
        if timepoints is None: timepoints = sorted(bw_files.keys())
        n_tracks = len(timepoints)
    else:
        timepoints = []
        n_tracks = 0

    # Layout
    if n_tracks == 0:
        fig = plt.figure(figsize=(figsize[0], max(figsize[1], 6.0)), constrained_layout=True)
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1])
        offset = 2
    else:
        height_ratios = [2, 0.5] + ([1] * n_tracks)
        fig = plt.figure(figsize=(figsize[0], figsize[1] + 2 * n_tracks), constrained_layout=True)
        gs = fig.add_gridspec(len(height_ratios), 1, height_ratios=height_ratios)
        offset = len(height_ratios) - n_tracks

    # Top: Logo
    ax_logo = fig.add_subplot(gs[0, 0])
    print(f"attribution array shape: {arr.shape}, plotting from {plt_start} to {plt_end}")
    
    arr_np = np.asarray(arr)
    if arr_np.ndim != 2:
        raise ValueError(f"Expected 2D attribution array, got {arr_np.shape}")
    if arr_np.shape[0] == 4:
        arr_logo = arr_np
    elif arr_np.shape[1] == 4:
        arr_logo = arr_np.T
    else:
        raise ValueError(f"Attributions must have a channel axis of 4, got {arr_np.shape}")
    seq_len = arr_logo.shape[1]

    logger.info(
        "[PLOT DEBUG] slicing arr with start=%d end=%d (len=%d)",
        int(row["start"]),
        int(row["end"]),
        int(row["end"] - row["start"])
    )



    plt_start = max(0, int(row.start) - (PLOT_WINDOW // 2))
    plt_end = min(seq_len, int(row.start) + (PLOT_WINDOW // 2))
    if plt_end <= plt_start:
        raise ValueError(f"Empty logo window for {seq_name}: start={plt_start}, end={plt_end}")
    
    window = arr_logo[:, plt_start:plt_end]
    logger.info(
        "[PLOT DEBUG] sliced window shape=%s window_sum=%.6e",
        window.shape,
        float(np.sum(window))
    )

    print(f"attribution array shape for logo: {arr_logo.shape}, plotting from {plt_start} to {plt_end}")
    
    plot_logo(arr_logo, ax=ax_logo, start=plt_start, end=plt_end, annotations=seq_annotations)
    ax_logo.set_title(f"Motif: {getattr(row, 'pattern_label', getattr(row, 'motif_name', 'unknown'))} | Seq: {seq_name}")
    ax_logo.set_ylabel("Attribution Score")
    if n_tracks == 0: ax_logo.set_xlabel("")
    else: ax_logo.set_xlabel("Position")

    # Middle: Gene Map
    ax_gene = fig.add_subplot(gs[1, 0])
    plot_gene_map(
        elements_df=seq_elements_df,
        ax=ax_gene,
        title=f"{gene_name}",
        xlim=(plot_start, plot_end),
        highlight_area=highlight_bounds, # This is the RED highlight for the motif
        orientation=orientation,
        start=plot_start,
        end=plot_end
    )
    
    # Apply RBP highlight (Orange) to Gene Map
    if element_highlight_bounds is not None:
        el_left, el_right = element_highlight_bounds
        ax_gene.axvspan(el_left, el_right, alpha=0.5, color="orange", zorder=100, clip_on=False, linewidth=0, label="RBP Site")

    # Bottom: Tracks (if any)
    if n_tracks > 0 and not seq_elements_df.empty:
        strand = seq_elements_df['strand'].iloc[0]
        flip_x = orientation == "RNA" and strand == '-'
        
        # Calculate highlight spans for tracks 
        if element_highlight_bounds:
            highlight_left, highlight_right = element_highlight_bounds
            if flip_x:
                highlight_spans = (min(highlight_left, highlight_right), max(highlight_left, highlight_right))
            else:
                highlight_spans = (min(highlight_left, highlight_right), max(highlight_left, highlight_right))
        else:
            highlight_spans = None
        
        last_ax = _plot_read_density_tracks(
            ax_gene=ax_gene,
            gs=gs,
            n_tracks=n_tracks,
            timepoints=timepoints,
            bw_files=bw_files,
            chrom=chrom,
            region_start=plot_start,
            region_end=plot_end,
            y_scale=y_scale,
            flip_x=flip_x,
            highlight_spans=highlight_spans,
            gene_label=transcript_name,
            offset=offset
        )
        if last_ax is not None:
            last_ax.set_xlabel("Genomic Position")

    save_path = f"{logo_plot_dir}/logo_map_densities_{seq_name}.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")