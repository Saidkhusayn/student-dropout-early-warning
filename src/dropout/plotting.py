"""Shared plot style so every figure in the project looks the same."""

from __future__ import annotations

import matplotlib.pyplot as plt

from dropout import config

# One fixed color per outcome, used in every figure (color follows the entity).
OUTCOME_COLORS = {
    "Graduate": "#2a78d6",  # blue
    "Dropout": "#eb6834",   # orange
    "Enrolled": "#1baf7a",  # aqua
}
OUTCOME_ORDER = ["Dropout", "Enrolled", "Graduate"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def set_style() -> None:
    """Apply the project's matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "text.color": INK,
        "legend.frameon": False,
    })


def save(fig, name: str) -> None:
    """Save a figure to results/figures/<name>.png."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / f"{name}.png", bbox_inches="tight")


# One fixed color and marker per model (validated as a set; labels are drawn on the lines).
MODEL_STYLE = {
    "logreg": {"color": "#6250d6", "marker": "o", "label": "Logistic regression"},
    "random_forest": {"color": "#eda100", "marker": "s", "label": "Random forest"},
    "gboost": {"color": "#e87ba4", "marker": "^", "label": "Gradient boosting"},
}
