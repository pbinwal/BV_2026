"""Plot pitch-entropy distances against correlation from a derived CSV."""

import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PITCH_ENTROPY_CSV = os.path.join(
    REPO_ROOT,
    "data",
    "pitch_entropy_distance",
    "pitch_entropy_correlations.csv",
)
SUPP_SAVE_DIR = os.path.join(REPO_ROOT, "figures", "Supplementary", "Supp_2")

CATEGORY_COLORS = {
    "non-significant": "#bdbdbd",
    "positive": "#e66557",
    "negative": "#6d90f0",
}


def plot_scatter(combined_df, context_type):
    """Plot z-scored pitch-entropy distance against correlation rho."""
    plot_df = combined_df.dropna(
        subset=["z_scored_acoustic_dist", "rho", "sig_status_aft_all_corr"]
    ).copy()
    if plot_df.empty:
        print(f"No plottable {context_type} pitch-entropy rows found.")
        return

    fig, ax = plt.subplots(figsize=(2.2, 1.9))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    if context_type == "next":
        for _, row in plot_df.iterrows():
            mirror = plot_df[
                (plot_df["syl1"] == row["syl2"])
                & (plot_df["syl2"] == row["syl1"])
                & (plot_df["bird_num"] == row["bird_num"])
            ]
            if not mirror.empty:
                mirror_row = mirror.iloc[0]
                ax.plot(
                    [row["z_scored_acoustic_dist"], mirror_row["z_scored_acoustic_dist"]],
                    [row["rho"], mirror_row["rho"]],
                    color="darkgrey",
                    linewidth=0.5,
                    zorder=1,
                )

    for category, color in CATEGORY_COLORS.items():
        subset = plot_df[plot_df["sig_status_aft_all_corr"] == category]
        if not subset.empty:
            ax.scatter(
                subset["z_scored_acoustic_dist"],
                subset["rho"],
                color=color,
                s=20,
                alpha=0.6,
                edgecolor="black",
                linewidth=0.5,
                zorder=2,
            )

    ax.axhline(0, color="grey", linestyle="--", linewidth=1, zorder=1)
    ax.set_xlabel("Acoustic distance", fontsize=10)
    ax.set_ylabel("Correlation", fontsize=10)
    ax.tick_params(axis="both", which="major", length=6, width=1, labelsize=10)
    ax.tick_params(axis="both", which="minor", length=3, width=0.5)
    ax.xaxis.set_major_locator(FixedLocator([-2.5, 0, 2.5]))
    ax.set_xlim(left=min(ax.get_xlim()[0], -2.7), right=max(ax.get_xlim()[1], 2.7))
    ax.xaxis.set_minor_locator(FixedLocator([-1.25, 1.25]))
    ax.yaxis.set_major_locator(FixedLocator([-0.8, 0, 0.8]))
    ax.set_ylim(bottom=min(ax.get_ylim()[0], -0.9), top=max(ax.get_ylim()[1], 0.9))
    ax.yaxis.set_minor_locator(FixedLocator([-0.4, 0.4]))
    ax.set_title("Adjacent" if context_type == "adjacent" else "Next", fontsize=10)
    fig.tight_layout()

    figure_name = (
        "supp_2_c_pitch_entropy.png"
        if context_type == "adjacent"
        else "supp_2_d_pitch_entropy.png"
    )
    save_path = os.path.join(SUPP_SAVE_DIR, figure_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".png", ".svg"), dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()
    plt.close(fig)


def main():
    if not os.path.exists(PITCH_ENTROPY_CSV):
        raise FileNotFoundError(
            f"Missing derived pitch-entropy table: {PITCH_ENTROPY_CSV}. "
            "Run scripts/supp_2_de_pitch_entropy_old.py first."
        )

    pitch_df = pd.read_csv(PITCH_ENTROPY_CSV)
    os.makedirs(SUPP_SAVE_DIR, exist_ok=True)
    plot_scatter(pitch_df[pitch_df["context"] == "adjacent"], "adjacent")
    plot_scatter(pitch_df[pitch_df["context"] == "next"], "next")


if __name__ == "__main__":
    main()