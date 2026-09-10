"""Dose-response plots: one figure per breakdown, comparing the unweighted
and rater-weighted curves side by side, one line per arm.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ARM_COLORS = {"standard": "#5b8def", "meme": "#e0605a", "control": "#7a8599"}


def plot_dose_response(
    unweighted_curve: pd.DataFrame,
    weighted_curve: pd.DataFrame,
    doses: list[int],
    out_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for ax, curve_df, subtitle in (
        (axes[0], unweighted_curve, "Unweighted"),
        (axes[1], weighted_curve, "Rater-weighted"),
    ):
        if curve_df is not None and not curve_df.empty:
            for arm, group in curve_df.groupby("arm"):
                group = group.sort_values("dose")
                ax.errorbar(
                    group["dose"],
                    group["log_strength_mean"],
                    yerr=group["log_strength_std"].fillna(0),
                    marker="o",
                    capsize=3,
                    label=arm,
                    color=ARM_COLORS.get(arm),
                )
        ax.axhline(0.0, linestyle="--", color="gray", linewidth=1, label="baseline (no LoRA)")
        ax.set_xscale("log")
        if doses:
            ax.set_xticks(doses)
            ax.set_xticklabels([str(d) for d in doses])
        ax.set_xlabel("dose (# training images)")
        ax.set_title(subtitle)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Kirk-ness (log-strength vs. baseline)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rater_calibration(rater_perf_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    if not rater_perf_df.empty:
        ax.scatter(rater_perf_df["n_calibration"], rater_perf_df["accuracy"], alpha=0.6)
    ax.set_xlabel("# calibration pairs answered")
    ax.set_ylabel("calibration accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Rater calibration performance")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
