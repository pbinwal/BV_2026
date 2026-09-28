"""Plot delta duration against correlation from the derived CSV."""

import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DURATION_CSV = os.path.join(
    REPO_ROOT, "data", "delta_duration", "duration_correlations.csv"
)
SUPP_SAVE_DIR = os.path.join(REPO_ROOT, "figures", "Supplementary", "Supp_2")

CATEGORY_COLORS = {
    "non-significant": "#bdbdbd",
    "positive": "#e66557",
    "negative": "#6d90f0",
}


def plot_duration_vs_correlation(combined_df, pair_type):
    """Plot one context's delta durations against correlation coefficients."""
    plot_df = combined_df.dropna(
        subset=["delta_duration", "rho", "sig_status_aft_all_corr"]
    ).copy()
    if plot_df.empty:
        print(f"No plottable {pair_type} duration/correlation rows found.")
        return

    fig, ax = plt.subplots(figsize=(2.2, 1.9), dpi=300)

    if pair_type == "next":
        plot_df["pair_key"] = plot_df["syl1"] + "_" + plot_df["syl2"]
        for _, row in plot_df.iterrows():
            mirror_key = f"{row['syl2']}_{row['syl1']}"
            mirror = plot_df[
                (plot_df["pair_key"] == mirror_key)
                & (plot_df["bird_num"] == row["bird_num"])
            ]
            if not mirror.empty:
                mirror_row = mirror.iloc[0]
                ax.plot(
                    [row["delta_duration"], mirror_row["delta_duration"]],
                    [row["rho"], mirror_row["rho"]],
                    color="darkgrey",
                    linewidth=0.5,
                    zorder=1,
                )

    for category, color in CATEGORY_COLORS.items():
        subset = plot_df[plot_df["sig_status_aft_all_corr"] == category]
        if not subset.empty:
            ax.scatter(
                subset["delta_duration"],
                subset["rho"],
                color=color,
                label=category,
                s=20,
                alpha=0.6,
                edgecolor="black",
                linewidth=0.5,
                zorder=2,
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.axhline(0, color="grey", linestyle="--", linewidth=1, zorder=1)
    ax.set_xlabel(r"$\Delta$ duration (ms)", fontsize=10)
    ax.set_ylabel("Correlation", fontsize=10)
    ax.tick_params(axis="both", which="major", length=6, width=1, labelsize=10)
    ax.tick_params(axis="both", which="minor", length=3, width=0.5)
    x_max = plot_df["delta_duration"].max()
    x_tick_step = 50
    x_axis_max = max(x_tick_step, (int(x_max // x_tick_step) + 1) * x_tick_step)
    ax.set_xlim(right=x_axis_max)
    ax.xaxis.set_major_locator(MultipleLocator(x_tick_step))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_ylim(bottom=-0.9, top=0.9)
    ax.set_yticks([-0.8, 0, 0.8])
    ax.set_yticks([-0.4, 0.4], minor=True)
    ax.set_title("Adjacent" if pair_type == "adjacent" else "Next", fontsize=10)
    fig.tight_layout()

    figure_name = (
        "supp_2_f_dur.png" if pair_type == "adjacent"
        else "supp_2_g_dur.png"
    )
    save_path = os.path.join(SUPP_SAVE_DIR, figure_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".png", ".svg"), bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()
    plt.close(fig)


def main():
    if not os.path.exists(DURATION_CSV):
        raise FileNotFoundError(
            f"Missing derived duration table: {DURATION_CSV}. "
            "Run scripts/supp_2_hi_dur_old.py first."
        )

    duration_df = pd.read_csv(DURATION_CSV)
    adjacent_df = duration_df[duration_df["context"] == "adjacent"]
    next_without_self_df = duration_df[
        duration_df["context"] == "next_without_self"
    ]

    os.makedirs(SUPP_SAVE_DIR, exist_ok=True)
    plot_duration_vs_correlation(adjacent_df, "adjacent")
    plot_duration_vs_correlation(next_without_self_df, "next")


if __name__ == "__main__":
    main()