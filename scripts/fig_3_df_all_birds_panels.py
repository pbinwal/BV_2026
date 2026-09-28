"""fig_3_all_birds_panels.py

Plots all-birds summary panels for occurrence and absolute-position analyses.
For each panel, every bird/syllable combination is drawn as a grey line, the
overall median across all is shown as a red dashed line, and the example
bird/syllable highlighted in the companion script is circled in blue.

  fig_3_d_occ.png  — median repeat number vs occurrence order
                             (blue circle: Bird 6 / syllable H)
    fig_3_f_pos.png  — median repeat number vs absolute position, visualized in
                       20-syllable bins after excluding positions above the
                       absolute position 200
                             (blue circle: Bird 1 / syllable U)

Saved as PNG (300 dpi) to figures/Figure 3/.

Prerequisites:
  fig_3_all_birds_rpt_by_pos.py must be run first to generate:
    output/Positions csvs/occurrence_analysis_median.csv
    output/Positions csvs/position_analysis_median.csv
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import MultipleLocator

from get_repeat_data import REPO_ROOT

# --- Paths ---
yaml_path = os.path.join(REPO_ROOT, "yamls", "bird_1.yaml")
with open(yaml_path, "r") as f:
    config = yaml.safe_load(f)

data_dir = os.path.join(REPO_ROOT, config["position_csvs_path"])
save_dir = os.path.join(REPO_ROOT, "figures", "Figure 3")
os.makedirs(save_dir, exist_ok=True)

occ_fp = os.path.join(data_dir, "occurrence_analysis_median.csv")
pos_fp = os.path.join(data_dir, "position_analysis_median.csv")
for fp in [occ_fp, pos_fp]:
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Required input not found: {fp}\nRun fig_3_all_birds_rpt_by_pos.py first.")

occ_df = pd.read_csv(occ_fp)
pos_df = pd.read_csv(pos_fp)

# ══════════════════════════════════════════════════════════════════════════════
# Occurrence: all birds
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(3.8, 3.2), dpi=300)

for (bird, syl), group in occ_df.groupby(["bird_num", "syllable"]):
    ax.plot(group["occurrence"], group["median_repeat_number"],
            marker="o", linewidth=1, color="#888888", markersize=3)

overall_occ = occ_df.groupby("occurrence")["median_repeat_number"].median().reset_index()
ax.plot(overall_occ["occurrence"], overall_occ["median_repeat_number"],
        marker="o", linestyle="--", color="red", markersize=3, linewidth=2, zorder=5)

highlight = occ_df[(occ_df["bird_num"] == 6) & (occ_df["syllable"] == "H")]
ax.scatter(highlight["occurrence"], highlight["median_repeat_number"],
           s=30, facecolors="none", edgecolors="blue", linewidths=2, zorder=6)

max_occ = int(occ_df["occurrence"].max())
ax.set_xticks(list(range(1, max_occ + 1, 2)))
ax.set_xticks(list(range(1, max_occ + 1)), minor=True)
ax.yaxis.set_major_locator(MultipleLocator(4))
ax.yaxis.set_minor_locator(MultipleLocator(2))
ax.set_ylim(0, 32)
ax.set_xlabel("Occurrence number", fontsize=11)
ax.set_ylabel("Median repeat number", fontsize=11)
ax.tick_params(axis="both", labelsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()

out = os.path.join(save_dir, "fig_3_d_occ.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.savefig(out.replace('.png', '.svg'), bbox_inches="tight")
plt.show()
plt.close(fig)
print(f"Saved: {out}")

# ══════════════════════════════════════════════════════════════════════════════
# Position: all birds
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(3.8, 3.2), dpi=300)

position_bin_size = 20
position_plot_max = 200
position_percentile = (pos_df["abs_pos"] <= position_plot_max).mean() * 100
print(
    f"Position plot: excluding abs_pos > {position_plot_max} "
    "(visualization only)."
)
print(
    f"Position plot cutoff {position_plot_max} includes "
    f"{position_percentile:.2f}% of raw position observations."
)
print(f"Position plot bin size: {position_bin_size} absolute-position units.")
plot_pos_df = pos_df[pos_df["abs_pos"] <= position_plot_max].copy()
plot_pos_df["position_bin"] = (
    (plot_pos_df["abs_pos"] - 1) // position_bin_size
) * position_bin_size + 1
plot_pos_df = (
    plot_pos_df
    .groupby(["bird_num", "syllable", "position_bin"], as_index=False)["median_repeat_number"]
    .median()
)

for (bird, syl), group in plot_pos_df.groupby(["bird_num", "syllable"]):
    ax.plot(group["position_bin"], group["median_repeat_number"],
            marker="o", linewidth=1, color="#888888", markersize=3)

overall_pos = plot_pos_df.groupby("position_bin")["median_repeat_number"].median().reset_index()
ax.plot(overall_pos["position_bin"], overall_pos["median_repeat_number"],
        marker="o", linestyle="--", color="red", markersize=3, linewidth=2, zorder=5)

highlight = plot_pos_df[(plot_pos_df["bird_num"] == 1) & (plot_pos_df["syllable"] == "U")]
ax.scatter(highlight["position_bin"], highlight["median_repeat_number"],
           s=30, facecolors="none", edgecolors="blue", linewidths=2, zorder=6)

ax.set_xticks([0, 45, 90, 135, 180])
ax.set_xticks([22.5, 67.5, 112.5, 157.5], minor=True)
ax.yaxis.set_major_locator(MultipleLocator(4))
ax.yaxis.set_minor_locator(MultipleLocator(2))
ax.set_ylim(0, 32)
ax.set_xlabel("Position", fontsize=11)
ax.set_ylabel("Median repeat number", fontsize=11)
ax.tick_params(axis="both", labelsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()

out = os.path.join(save_dir, "fig_3_f_pos.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.savefig(out.replace('.png', '.svg'), bbox_inches="tight")
plt.show()
plt.close(fig)
print(f"Saved: {out}")
