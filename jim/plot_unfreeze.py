"""Comparison figure for the unfreezing experiment (N = 0, 2, 4, 6, 12 last DINOv3 blocks fine-tuned).

Reads results/<task>_unf<N>/{summary.json,history.csv}; N = 0 is Andreas's frozen baseline, whose numbers are
copied from the outputs of dinov3_dpt_seg_depth.ipynb (commit 89b1762).
Usage: python jim/plot_unfreeze.py [RESULTS_DIR]   (default: ../results next to jim/)
       -> RESULTS_DIR/unfreeze_comparison.png and RESULTS_DIR/lr_and_seeds.png
"""
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results"
NS = [0, 2, 4, 6, 12]
XLABELS = ["0\nfrozen", "2", "4", "6", "12\nall"]

BASELINE = {
    "seg": {"val": {"mIoU": 0.8937}, "test": {"mIoU": 0.8382, "IoU_Impact Crater": 0.5818, "IoU_Rock": 0.7633,
                                                "IoU_Mountain": 0.8723}},
    "depth": {"val": {"AbsRel": 0.0547}, "test": {"AbsRel": 0.1376, "d1": 0.7598}},
}
BASELINE_CURVE = {
    "mIoU": [0.8613, 0.8636, 0.8769, 0.8757, 0.8884, 0.8884, 0.8883, 0.8931, 0.8933, 0.8937],
    "AbsRel": [0.0909, 0.0779, 0.0593, 0.0790, 0.0548, 0.0547, 0.0615, 0.0552, 0.0559, 0.0554],
}

# Validated with the dataviz skill's validate_palette.js (light surface)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]   # ordinal: N = 0 ... 12


def load(task, n):
    if n == 0:
        return BASELINE[task], pd.DataFrame({"epoch": range(1, 11), **{k: v for k, v in BASELINE_CURVE.items()
                                                                        if (k == "mIoU") == (task == "seg")}})
    d = RESULTS / f"{task}_unf{n}"
    return json.loads((d / "summary.json").read_text())["results"], pd.read_csv(d / "history.csv")


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)


def line(ax, ys, color, label, fmt="{:.3f}", marker="o"):
    x = range(len(ys))
    ax.plot(x, ys, color=color, linewidth=2, marker=marker, markersize=6.5,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    ax.annotate(f"{label}  " + fmt.format(ys[-1]), (len(ys) - 1, ys[-1]), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    ax.set_xticks(list(x), XLABELS)
    ax.set_xlim(-0.3, len(ys) + 0.6)


runs = {t: [load(t, n) for n in NS] for t in ("seg", "depth")}
fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), facecolor=SURFACE)

# --- segmentation ---
ax = axes[0, 0]
line(ax, [r["val"]["mIoU"] for r, _ in runs["seg"]], BLUE, "val")
line(ax, [r["test"]["mIoU"] for r, _ in runs["seg"]], ORANGE, "test", marker="s")
style(ax, "Segmentation: mIoU (higher is better)", "mIoU")
ax.legend(frameon=False, fontsize=8.5, loc="upper left")

ax = axes[0, 1]
for key, color, lab, m in [("IoU_Impact Crater", BLUE, "crater", "o"), ("IoU_Rock", ORANGE, "rock", "s"),
                           ("IoU_Mountain", AQUA, "mountain", "^")]:
    line(ax, [r["test"][key] for r, _ in runs["seg"]], color, lab, marker=m)
style(ax, "Segmentation: test IoU of the rare classes", "IoU")
ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.0, 0.47))

ax = axes[0, 2]
for (r, h), n, c in zip(runs["seg"], NS, RAMP):
    ax.plot(h["epoch"], h["mIoU"], color=c, linewidth=2, label=f"N={n}")
style(ax, "Segmentation: val mIoU per epoch", "val mIoU")
ax.set_xlabel("epoch", color=INK2, fontsize=9)
ax.legend(frameon=False, fontsize=8.5, loc="lower right", title="unfrozen blocks", title_fontsize=8.5)

# --- depth ---
ax = axes[1, 0]
# Shaded bar: min-max of val AbsRel over epochs 6-10, i.e. how much the score moves without changing N
spread = [h["AbsRel"].iloc[5:] for _, h in runs["depth"]]
ax.vlines(range(len(NS)), [s.min() for s in spread], [s.max() for s in spread], color="#cde2fb", linewidth=10,
          label="spread over epochs 6-10")
line(ax, [r["val"]["AbsRel"] for r, _ in runs["depth"]], BLUE, "best val", fmt="{:.4f}")
style(ax, "Depth: val AbsRel (lower is better)", "AbsRel")
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
ax.set_xlabel("unfrozen last N blocks", color=INK2, fontsize=9)

ax = axes[1, 1]
line(ax, [r["test"]["AbsRel"] for r, _ in runs["depth"]], ORANGE, "test", fmt="{:.4f}", marker="s")
style(ax, "Depth: test AbsRel (lower is better)", "AbsRel")
ax.set_xlabel("unfrozen last N blocks", color=INK2, fontsize=9)

ax = axes[1, 2]
for (r, h), n, c in zip(runs["depth"], NS, RAMP):
    ax.plot(h["epoch"], h["AbsRel"], color=c, linewidth=2, label=f"N={n}")
style(ax, "Depth: val AbsRel per epoch (noisy)", "val AbsRel")
ax.set_ylim(0.04, 0.10)
ax.text(3.0, 0.0985, "N=4 epoch 1 = 0.150 (off scale)", fontsize=8, color=INK2, va="top")
ax.set_xlabel("epoch", color=INK2, fontsize=9)
ax.legend(frameon=False, fontsize=8.5, loc="upper right", title="unfrozen blocks", title_fontsize=8.5)

for ax in axes[0, :2]:
    ax.set_xlabel("unfrozen last N blocks", color=INK2, fontsize=9)

fig.suptitle("DINOv3 ViT-B/16 + DPT on LuSNAR: fine-tuning the last N backbone blocks "
             "(10 epochs, backbone LR 1e-5, one seed)", x=0.01, ha="left", fontsize=12.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.5, w_pad=2.5)
out = RESULTS / "unfreeze_comparison.png"
fig.savefig(out, dpi=150, facecolor=SURFACE)
print("saved", out)


# ===== Figure 2: backbone LR (N=12) and depth seeds =====
LRS = ["3e-6", "1e-5", "3e-5"]
lr_runs = {t: [json.loads((RESULTS / (f"{t}_unf12" if lr == "1e-5" else f"{t}_unf12_lr{lr}") / "summary.json")
                          .read_text())["results"] for lr in LRS] for t in ("seg", "depth")}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=SURFACE)

for ax, task, metric, title, fmt in [
        (axes[0], "seg", "mIoU", "Segmentation (N=12): mIoU vs backbone LR (higher is better)", "{:.3f}"),
        (axes[1], "depth", "AbsRel", "Depth (N=12): AbsRel vs backbone LR (lower is better)", "{:.4f}")]:
    for split, color, m in (("val", BLUE, "o"), ("test", ORANGE, "s")):
        ys = [r[split][metric] for r in lr_runs[task]]
        ax.plot(range(3), ys, color=color, linewidth=2, marker=m, markersize=6.5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=split)
        ax.annotate(f"{split}  " + fmt.format(ys[-1]), (2, ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK2)
    ax.set_xticks(range(3), LRS); ax.set_xlim(-0.3, 2.9)
    style(ax, title, metric)
    ax.set_xlabel("backbone learning rate (1e-5 = batch 1)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="center left")
axes[1].set_yscale("log")
axes[1].yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.05, 0.06, 0.08, 0.10, 0.12, 0.14]))
axes[1].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
axes[1].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
axes[1].set_ylabel("AbsRel (log scale)", color=INK2, fontsize=9)

ax = axes[2]
for i, n in enumerate([0, 6, 12]):
    vals = [BASELINE["depth"]["test"]["AbsRel"] if n == 0 and s == 0 else
            json.loads((RESULTS / (f"depth_unf{n}" if s == 0 else f"depth_unf{n}_s{s}") / "summary.json")
                       .read_text())["results"]["test"]["AbsRel"] for s in (0, 1, 2)]
    ax.scatter([i] * 3, vals, s=42, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=3,
               label="one seed" if i == 0 else None)
    mean = sum(vals) / 3
    ax.hlines(mean, i - 0.22, i + 0.22, color=INK, linewidth=2, zorder=4, label="mean of 3 seeds" if i == 0 else None)
    ax.annotate(f"{mean:.4f}", (i + 0.22, mean), xytext=(6, 0), textcoords="offset points", va="center",
                fontsize=8.5, color=INK2)
ax.set_xticks(range(3), ["0\nfrozen", "6", "12\nall"]); ax.set_xlim(-0.5, 2.8)
style(ax, "Depth: test AbsRel over 3 seeds (lower is better)", "test AbsRel")
ax.set_xlabel("unfrozen last N blocks (backbone LR 1e-5)", color=INK2, fontsize=9)
ax.legend(frameon=False, fontsize=8.5, loc="upper right")

fig.suptitle("Batch 2: backbone learning rate at N=12, and seed variation for depth", x=0.01, ha="left",
             fontsize=12.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.5)
out = RESULTS / "lr_and_seeds.png"
fig.savefig(out, dpi=150, facecolor=SURFACE)
print("saved", out)
