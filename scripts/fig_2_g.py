"""fig_2_g.py

Combines per-bird correlation-by-phrase-distance CSVs across all birds, applies
two independent Benjamini-Hochberg (BH) corrections, and produces scatter plots
of |correlation| vs. phrase distance (Figure 2G).

Prerequisites:
  Run the per-bird correlation-by-phrase-distance analysis first so that the
  input CSVs below exist.

Input CSVs (read from scripts/output/Correlations by distance z corrected/):
  bird_*_dist_by_phrase_vs_corrs_with_syl.csv
      Observed Spearman correlations per syllable pair and phrase distance.
      Expected columns: Bird_ID, Syllable_Pair, Distance, Correlation,
                        Sample_size, p_value
  bird_*_dist_by_phrase_vs_corrs_with_syl_ztest.csv
      Z-test results comparing observed vs synthetic correlations.
      Expected columns: same as above plus z, ztest_p

BH corrections applied:
  1. On all original p_values → p_value_adjusted_BH
  2. On z-test p-values → ztest_p_adjusted_BH
  A pair is marked 'significant' only when both BH-corrected p-values < 0.05.

Thresholds:
  - Minimum sample size: 100 (rows below this are dropped before any analysis)
  - Maximum phrase distance: 20

Outputs (saved to figures/Figure 2/):
  fig_2_g_scatter.png  — |Correlation| vs distance (0-indexed)
  fig_2_g.png          — same with distances translated to 1-indexed phrases,
                         rectangular legend (Figure 2G panel)

Also saves a combined all-birds CSV:
  scripts/output/Correlations by distance z corrected/all_birds_corr_by_dist.csv
"""

import glob
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from matplotlib.lines import Line2D
from statsmodels.stats.multitest import multipletests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

z_corr_dir = os.path.join(REPO_ROOT, "output", "Correlations by distance z corrected")
fig2_dir   = os.path.join(REPO_ROOT, "figures", "Figure 2")

obs_files   = sorted(glob.glob(os.path.join(z_corr_dir, "bird_*_dist_by_phrase_vs_corrs_with_syl.csv")))
ztest_files = sorted(glob.glob(os.path.join(z_corr_dir, "bird_*_dist_by_phrase_vs_corrs_with_syl_ztest.csv")))

print(f"Found {len(obs_files)} observed correlation files")
print(f"Found {len(ztest_files)} z-test correlation files")

if not obs_files or not ztest_files:
    print("Error: could not find correlation files in:\n  " + z_corr_dir)
    exit()

SAMPLE_THRESHOLD   = 100
DISTANCE_THRESHOLD = 20

# --- Aggregate observed data ---
all_obs_dfs = []
for f in obs_files:
    df = pd.read_csv(f)
    if "Sample_size" in df.columns:
        df = df[df["Sample_size"] >= SAMPLE_THRESHOLD].copy()
    df = df[df["Distance"] <= DISTANCE_THRESHOLD]
    all_obs_dfs.append(df)

all_obs_df = pd.concat(all_obs_dfs, ignore_index=True)
print(f"\nCombined observed data: {len(all_obs_df)} rows")

# BH correction on all original p-values
all_obs_df["p_value_adjusted_BH"] = multipletests(all_obs_df["p_value"], method="fdr_bh")[1]

# --- Aggregate z-test data ---
all_ztest_dfs = []
for f in ztest_files:
    df = pd.read_csv(f)
    if "Sample_size" in df.columns:
        df = df[df["Sample_size"] >= SAMPLE_THRESHOLD].copy()
    df = df[df["Distance"] <= DISTANCE_THRESHOLD]
    all_ztest_dfs.append(df)

all_ztest_df = pd.concat(all_ztest_dfs, ignore_index=True)
print(f"Combined z-test data: {len(all_ztest_df)} rows")

# BH correction on z-test p-values
all_ztest_df["ztest_p_adjusted_BH"] = multipletests(all_ztest_df["ztest_p"], method="fdr_bh")[1]

# Merge BH-corrected original p-values into z-test dataframe
all_ztest_df = all_ztest_df.merge(
    all_obs_df[["Bird_ID", "Syllable_Pair", "Distance", "p_value_adjusted_BH"]],
    on=["Bird_ID", "Syllable_Pair", "Distance"],
    how="left",
)

# Significance: significant only if both BH-corrected p-values < 0.05
all_ztest_df["Significance"] = "other"
all_ztest_df.loc[all_ztest_df["p_value_adjusted_BH"] >= 0.05, "Significance"] = "non-significant"
all_ztest_df.loc[
    (all_ztest_df["p_value_adjusted_BH"] < 0.05)
    & (all_ztest_df["ztest_p_adjusted_BH"].notna())
    & (all_ztest_df["ztest_p_adjusted_BH"] < 0.05),
    "Significance",
] = "significant"

# Merge z-test significance back into observed dataframe
all_obs_df = all_obs_df.merge(
    all_ztest_df[["Bird_ID", "Syllable_Pair", "Distance", "z", "ztest_p", "ztest_p_adjusted_BH", "Significance"]],
    on=["Bird_ID", "Syllable_Pair", "Distance"],
    how="left",
)

# Fill NaN Significance for rows that weren't z-tested
all_obs_df.loc[all_obs_df["Significance"].isna() & (all_obs_df["p_value_adjusted_BH"] >= 0.05), "Significance"] = "non-significant"
all_obs_df.loc[all_obs_df["Significance"].isna() & (all_obs_df["p_value_adjusted_BH"] < 0.05), "Significance"] = "other"

# Save combined CSV
os.makedirs(z_corr_dir, exist_ok=True)
combined_csv = os.path.join(z_corr_dir, "all_birds_corr_by_dist.csv")
all_obs_df.to_csv(combined_csv, index=False)
print(f"\nSaved combined CSV: {combined_csv}")

# Print summary
print(f"\nTotal rows: {len(all_obs_df)}")
print(f"Significant (both corrections < 0.05): {(all_obs_df['Significance'] == 'significant').sum()}")
print(f"Non-significant: {(all_obs_df['Significance'] == 'non-significant').sum()}")
print("\nSignificance by bird:")
for bird in sorted(all_obs_df["Bird_ID"].unique()):
    bird_df = all_obs_df[all_obs_df["Bird_ID"] == bird]
    sig   = (bird_df["Significance"] == "significant").sum()
    total = len(bird_df)
    print(f"  {bird}: {sig} / {total} significant")

all_obs_df["Abs_Correlation"] = all_obs_df["Correlation"].abs()
sig_df    = all_obs_df[all_obs_df["Significance"] == "significant"]
nonsig_df = all_obs_df[all_obs_df["Significance"] != "significant"]

os.makedirs(fig2_dir, exist_ok=True)

# ── Figure 2G: translated distances (0→1, 1→2, …) — Fig 2G panel ───────────
translated_df = all_obs_df.copy()
translated_df["Distance_translated"] = translated_df["Distance"] + 1
sig_df_t    = translated_df[translated_df["Significance"] == "significant"]
nonsig_df_t = translated_df[translated_df["Significance"] != "significant"]

fig2, ax2 = plt.subplots(figsize=(3, 3), dpi=300)

ax2.scatter(nonsig_df_t["Distance_translated"], nonsig_df_t["Abs_Correlation"],
            color="#888888", marker="x", alpha=0.9, s=15,
            label="Non-significant", linewidths=0.5)
ax2.scatter(sig_df_t["Distance_translated"], sig_df_t["Abs_Correlation"],
            color="#009E73", marker="o", alpha=0.9, s=15,
            label="Significant", edgecolors="#009E73", linewidths=0.5)

ax2.set_xlabel("Distance in phrases", fontsize=11, color="black")
ax2.set_ylabel("|Correlation|", fontsize=11, color="black")
ax2.tick_params(axis="both", which="major", labelsize=11, colors="black", length=8, width=0.8)
ax2.tick_params(axis="both", which="minor", length=4, width=0.5, colors="black")
ax2.set_xticks([0, 5, 10, 15, 20])
ax2.set_xticklabels(["0", "5", "10", "15", "20"])
ax2.set_xlim(0, 20)
ax2.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax2.yaxis.set_major_locator(ticker.MaxNLocator(5))
ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_linewidth(0.8)
ax2.spines["bottom"].set_linewidth(0.8)

legend_elements = [
    Line2D([0], [0], marker="o", color="w", label="Significant",
           markerfacecolor="#009E73", markeredgecolor="#009E73", markersize=4, linewidth=0),
    Line2D([0], [0], marker="x", color="w", label="Non-significant",
           markerfacecolor="#888888", markeredgecolor="#888888", markersize=4, linewidth=0),
]
legend = ax2.legend(legend_elements, ["significant correlations", "non-significant"],
                    loc="upper right", fontsize=6, frameon=True)
legend.get_frame().set_facecolor("white")
legend.get_frame().set_edgecolor("black")
legend.get_frame().set_linewidth(0.7)
legend.get_frame().set_boxstyle("Square", pad=0.2)

plt.tight_layout()
fig2g_path = os.path.join(fig2_dir, "fig_2_g.png")
plt.savefig(fig2g_path, dpi=300, bbox_inches="tight", transparent=False)
plt.savefig(fig2g_path.replace('.png', '.svg'), bbox_inches="tight", transparent=False)
print(f"Saved: {fig2g_path}")
plt.show()
