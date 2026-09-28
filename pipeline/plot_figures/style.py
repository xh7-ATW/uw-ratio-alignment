"""Shared Matplotlib style for paper figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def configure_paper_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 150,
        }
    )


def add_legend(ax: plt.Axes, **kwargs) -> None:
    legend = ax.legend(
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#d0d0d0",
        borderpad=0.3,
        handlelength=2.4,
        handletextpad=0.5,
        labelspacing=0.25,
        **kwargs,
    )
    legend.set_zorder(10)


def style_axis(
    ax: plt.Axes,
    *,
    xlabel: str = "Optimizer step",
    ylabel: str,
    threshold: float | None = None,
    tick_fontsize: int = 28,
    label_fontsize: int = 34,
) -> None:
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(True, alpha=0.28)
    if threshold is not None:
        ax.axhline(threshold, color="gray", linestyle="--", linewidth=2.2)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(path)
