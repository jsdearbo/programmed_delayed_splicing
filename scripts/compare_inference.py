import os, sys
import numpy as np
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BASE_DIR = "/netfiles/dmlabshare1/rstagg/programmed_delayed_splicing"
OUT_DIR  = os.path.join(BASE_DIR, "rstest_outputs", "inference_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

FT_PREDS = os.path.join(BASE_DIR, "rstest_outputs", "ft_ighm_preds.npy")
PT_PREDS = os.path.join(BASE_DIR, "rstest_outputs", "pt_ighm_preds.npy")

INPUT_START = 113157856
BIN_SIZE    = 32
N_BINS      = 6144

label_start = INPUT_START + (524288 - N_BINS * BIN_SIZE) // 2
bin_centers  = label_start + np.arange(N_BINS) * BIN_SIZE + BIN_SIZE // 2
bin_centers_mb = bin_centers / 1e6

IGHM_START_MB = 113418558 / 1e6
IGHM_END_MB   = 113422730 / 1e6

# Load predictions
ft = np.load(FT_PREDS)
pt = np.load(PT_PREDS)

best_pt, best_r = [], []
for fi in range(ft.shape[1]):
    cors  = [pearsonr(ft[0, fi], pt[0, j])[0] for j in range(pt.shape[1])]
    bi    = int(np.argmax(cors))
    best_pt.append(bi)
    best_r.append(cors[bi])

tracks = [
    (ft[0, 0],          "#5BAEE8", f"Fine-tuned  •  Task 0: SRR6978978 (LLPC)"),
    (pt[0, best_pt[0]], "#E8603C", f"Pre-trained  •  PT task {best_pt[0]}  (r={best_r[0]:.3f})"),
    (ft[0, 1],          "#5BB55B", f"Fine-tuned  •  Task 1: SRR6978987 (LLPC)"),
    (pt[0, best_pt[1]], "#F5A623", f"Pre-trained  •  PT task {best_pt[1]}  (r={best_r[1]:.3f})"),
]

# Plot
fig, axes = plt.subplots(
    len(tracks), 1,
    figsize=(14, 3.2 * len(tracks)),
    sharex=True,
)
fig.suptitle(
    f"Borzoi inference on IgHM locus (chr12:{INPUT_START:,}–{INPUT_START+524288:,}, mm10)\n"
    "Fine-tuned vs best-matching pre-trained track",
    fontsize=12, fontweight="normal", y=0.99
)

x = bin_centers_mb

for ax, (signal, color, label) in zip(axes, tracks):
    ax.fill_between(x, signal, color=color, alpha=0.80, lw=0)
    ax.plot(x, signal, color=color, lw=0.6)

    ax.text(0.01, 0.96, label,
            transform=ax.transAxes, va="top", ha="left",
            fontsize=9, color="#222222")

    ax.set_ylabel("Pred. signal", fontsize=8)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)

axes[-1].set_xlabel(f"Genomic position (Mb) [chr12]", fontsize=9)
axes[-1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))

fig.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(OUT_DIR, "track_comparison_mpl.png")
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out}")