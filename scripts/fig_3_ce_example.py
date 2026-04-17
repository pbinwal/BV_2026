"""fig_3_all_birds_rpt_by_pos.py

For all birds, computes repeat-number distributions as a function of (a) within-song
occurrence order and (b) relative position within the song, then runs linear regressions
for both and saves the results.

A 5% threshold is applied throughout:
- Occurrence: only occurrence numbers that account for >= 5% of all instances of that
  syllable in that bird are retained.
- Relative position: the continuous relative position is binned into quartiles; quartile
  bins that account for < 5% of instances are dropped.

Outputs (saved to the directory specified by position_csvs_path in bird_1.yaml):
  occurrence_analysis_median.csv   — median repeat number per occurrence
  position_analysis_median.csv     — median repeat number per quartile

  occurrence_regression.csv        — linear regression results (repeat ~ occurrence)
  position_regression.csv          — linear regression results (repeat ~ rel_pos)

Both regression CSVs include FDR-corrected and Bonferroni-corrected p-values.

Example panels (saved to figures/Figure 3/):
  fig_3_c_occ.png  — Bird 6, syllable H, occurrences 1–4
  fig_3_e_pos.png  — Bird 1, syllable U, position quartiles 1–4
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from statsmodels.stats.multitest import multipletests

from get_compressed_syntax import load_and_process_data
from get_repeat_data import REPO_ROOT, syllables_mapping

# --- Load output directory from YAML ---
yaml_path = os.path.join(REPO_ROOT, "yamls", "bird_1.yaml")
with open(yaml_path, "r") as f:
    config = yaml.safe_load(f)

output_dir = os.path.join(REPO_ROOT, config["position_csvs_path"])
os.makedirs(output_dir, exist_ok=True)

save_dir = os.path.join(REPO_ROOT, "figures", "Figure 3")
os.makedirs(save_dir, exist_ok=True)

PCT_THRESHOLD = 5.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_syllables(song):
    """Return list of (syllable, repeat_number) from a compressed song string."""
    return re.findall(r"([a-zA-Z])(\d+)", song)


def filter_by_threshold(df, group_col, pct_threshold=PCT_THRESHOLD):
    """Drop rows whose group_col value accounts for < pct_threshold % of the total."""
    counts = df[group_col].value_counts()
    freqs  = counts / counts.sum() * 100
    valid  = freqs[freqs >= pct_threshold].index
    return df[df[group_col].isin(valid)].copy()


def filter_quartile_threshold(df, pct_threshold=PCT_THRESHOLD):
    counts = df["quartile"].value_counts()
    freqs  = counts / counts.sum() * 100
    valid  = freqs[freqs >= pct_threshold].index
    return df[df["quartile"].isin(valid)].copy()


# ── Main loop: collect raw data across all birds ──────────────────────────────

summary_occurrence = []
summary_position   = []

for bird_num, sylls in syllables_mapping.items():
    print(f"\n===== Bird {bird_num} =====")
    df = load_and_process_data(bird_num)
    if "song_id" not in df.columns:
        df["song_id"] = df.index.astype(str)

    for syll in sylls:

        # ── Q1: occurrence order ──────────────────────────────────────────────
        occ_data = []
        for song_id, song in zip(df["song_id"], df["compressed_song"]):
            occ_counter = 0
            for s, n in extract_syllables(song):
                if s == syll:
                    occ_counter += 1
                    occ_data.append((occ_counter, int(n), song_id))

        if occ_data:
            df_occ = pd.DataFrame(occ_data, columns=["occurrence", "repeat_number", "song_id"])
            df_occ = filter_by_threshold(df_occ, "occurrence")
            if not df_occ.empty:
                df_occ["bird_num"]     = int(bird_num)
                df_occ["syllable"]     = syll.upper()
                df_occ["bird_syllable"] = f"{bird_num}_{syll.upper()}"
                summary_occurrence.append(df_occ)

        # ── Q2: relative position ─────────────────────────────────────────────
        pos_data = []
        for song_id, song in zip(df["song_id"], df["compressed_song"]):
            matches  = extract_syllables(song)
            song_len = sum(int(n) for _, n in matches)
            cur_pos  = 1
            for s, n in matches:
                if s == syll:
                    rel_pos = cur_pos / song_len
                    pos_data.append((rel_pos, int(n), song_id))
                cur_pos += int(n)

        if pos_data:
            df_pos = pd.DataFrame(pos_data, columns=["rel_pos", "repeat_number", "song_id"])
            if not df_pos.empty:
                df_pos["bird_num"]     = int(bird_num)
                df_pos["syllable"]     = syll.upper()
                df_pos["bird_syllable"] = f"{bird_num}_{syll.upper()}"
                summary_position.append(df_pos)


# ── Save occurrence data ──────────────────────────────────────────────────────

full_occ_df = pd.concat(summary_occurrence, ignore_index=True)
full_occ_df = full_occ_df[["bird_num", "syllable", "bird_syllable", "occurrence", "repeat_number", "song_id"]]

median_occ_df = (
    full_occ_df
    .groupby(["bird_num", "syllable", "bird_syllable", "occurrence"], as_index=False)["repeat_number"]
    .median()
    .rename(columns={"repeat_number": "median_repeat_number"})
)
median_occ_df.to_csv(os.path.join(output_dir, "occurrence_analysis_median.csv"), index=False)
print("Saved: occurrence_analysis_median.csv")


# ── Save position data ────────────────────────────────────────────────────────

full_pos_df = pd.concat(summary_position, ignore_index=True)
full_pos_df = full_pos_df[["bird_num", "syllable", "bird_syllable", "rel_pos", "repeat_number", "song_id"]]

# Bin into quartiles then apply 5% threshold per bird/syllable
full_pos_df["quartile"] = full_pos_df.groupby(["bird_num", "syllable", "bird_syllable"])["rel_pos"].transform(
    lambda x: pd.qcut(x, 4, labels=False, duplicates="drop") + 1
)

filtered_pos = []
for _, group in full_pos_df.groupby(["bird_num", "syllable", "bird_syllable"]):
    filtered = filter_quartile_threshold(group)
    if not filtered.empty:
        filtered_pos.append(filtered)

full_pos_df = pd.concat(filtered_pos, ignore_index=True) if filtered_pos else full_pos_df.iloc[0:0]

median_pos_df = (
    full_pos_df
    .groupby(["bird_num", "syllable", "bird_syllable", "quartile"], as_index=False)["repeat_number"]
    .median()
    .rename(columns={"repeat_number": "median_repeat_number"})
)
median_pos_df.to_csv(os.path.join(output_dir, "position_analysis_median.csv"), index=False)
print("Saved: position_analysis_median.csv")


# ── Linear regression: repeat ~ occurrence ───────────────────────────────────

print("\n\n===== Linear Regressions: Occurrence =====")
occ_results = []
for bird_syll, group in full_occ_df.groupby("bird_syllable"):
    x = group["occurrence"].values
    y = group["repeat_number"].values
    if len(x) < 3:
        continue
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    rpt_std  = np.std(y, ddof=1)
    rpt_mean = np.mean(y)
    print(f"  {bird_syll}: slope={slope:.3f}, R²={r_value**2:.3f}, p={p_value:.4f}")
    occ_results.append({
        "bird_syllable": bird_syll,
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_value**2,
        "p_value_uncorrected": p_value,
        "std_err": std_err,
        "n": len(x),
        "repeat_mean": rpt_mean,
        "repeat_std": rpt_std,
    })

occ_reg_df = pd.DataFrame(occ_results)
_, p_fdr,  _, _ = multipletests(occ_reg_df["p_value_uncorrected"], method="fdr_bh")
_, p_bonf, _, _ = multipletests(occ_reg_df["p_value_uncorrected"], method="bonferroni")
occ_reg_df["p_value_fdr"]         = p_fdr
occ_reg_df["p_value_bonferroni"]  = p_bonf
occ_reg_df["normalised_slope"]    = occ_reg_df["slope"] / occ_reg_df["repeat_std"]
occ_reg_df.to_csv(os.path.join(output_dir, "occurrence_regression.csv"), index=False)
print(f"Saved: occurrence_regression.csv")

sig_occ = occ_reg_df[occ_reg_df["p_value_fdr"] < 0.05]
if not sig_occ.empty:
    s = sig_occ["slope"].values
    sn = sig_occ["normalised_slope"].values
    print(f"  FDR-significant slopes: range=({s.min():.3f}, {s.max():.3f}), mean={s.mean():.3f}, SE={s.std(ddof=1)/len(s)**0.5:.3f}")
    print(f"  Standardised: range=({sn.min():.3f}, {sn.max():.3f}), mean={sn.mean():.3f}, SE={sn.std(ddof=1)/len(sn)**0.5:.3f}")


# ── Linear regression: repeat ~ rel_pos ──────────────────────────────────────

print("\n\n===== Linear Regressions: Position =====")
pos_results = []
for bird_syll, group in full_pos_df.groupby("bird_syllable"):
    x = group["rel_pos"].values
    y = group["repeat_number"].values
    if len(x) < 3:
        continue
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    rpt_std  = np.std(y, ddof=1)
    rpt_mean = np.mean(y)
    print(f"  {bird_syll}: slope={slope:.3f}, R²={r_value**2:.3f}, p={p_value:.4f}")
    pos_results.append({
        "bird_syllable": bird_syll,
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_value**2,
        "p_value_uncorrected": p_value,
        "std_err": std_err,
        "n": len(x),
        "repeat_mean": rpt_mean,
        "repeat_std": rpt_std,
    })

pos_reg_df = pd.DataFrame(pos_results)
_, p_fdr,  _, _ = multipletests(pos_reg_df["p_value_uncorrected"], method="fdr_bh")
_, p_bonf, _, _ = multipletests(pos_reg_df["p_value_uncorrected"], method="bonferroni")
pos_reg_df["p_value_fdr"]        = p_fdr
pos_reg_df["p_value_bonferroni"] = p_bonf
pos_reg_df["normalised_slope"]   = pos_reg_df["slope"] / pos_reg_df["repeat_std"]
pos_reg_df.to_csv(os.path.join(output_dir, "position_regression.csv"), index=False)
print(f"Saved: position_regression.csv")

sig_pos = pos_reg_df[pos_reg_df["p_value_fdr"] < 0.05]
if not sig_pos.empty:
    s = sig_pos["slope"].values
    sn = sig_pos["normalised_slope"].values
    print(f"  FDR-significant slopes: range=({s.min():.3f}, {s.max():.3f}), mean={s.mean():.3f}, SE={s.std(ddof=1)/len(s)**0.5:.3f}")
    print(f"  Standardised: range=({sn.min():.3f}, {sn.max():.3f}), mean={sn.mean():.3f}, SE={sn.std(ddof=1)/len(sn)**0.5:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Example panels (generated from in-memory data)
# ══════════════════════════════════════════════════════════════════════════════

OCC_COLOR = "#fb6a4aff"
POS_COLOR = "#e32f27ff"

# ── Occurrence example: Bird 6, syllable H ───────────────────────────────────

bird_num_occ       = 6
syllable_occ       = "H"
selected_occ_order = [1, 2, 3, 4]

df_occ_ex = full_occ_df[
    (full_occ_df["bird_num"].astype(str) == str(bird_num_occ)) &
    (full_occ_df["syllable"].str.upper() == syllable_occ)
]

if df_occ_ex.empty:
    print(f"No occurrence data for bird {bird_num_occ} syllable {syllable_occ}.")
else:
    max_rn = int(df_occ_ex["repeat_number"].max())
    nrows, ncols = 4, 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.9, 2.6), sharex=True, sharey=True, dpi=300)
    axes = axes.flatten()

    for j, occ in enumerate(selected_occ_order):
        ax = axes[j]
        data = df_occ_ex[df_occ_ex["occurrence"] == occ]
        if not data.empty:
            bins = range(0, max_rn + 2)
            counts, _ = np.histogram(data["repeat_number"], bins=bins)
            rel_freq = counts / counts.sum() if counts.sum() > 0 else counts
            ax.bar(bins[:-1], rel_freq, width=1.0, color=OCC_COLOR, edgecolor="k")

        ax.set_xticks(np.arange(0, 13, 6))
        ax.set_xticks(np.array([3, 9]), minor=True)
        ax.set_xlim(0, 13)
        ax.set_yticks(np.arange(0, 0.41, 0.4))
        ax.set_yticks(np.arange(0, 0.61, 0.2), minor=True)
        ax.set_ylim(0.0, 0.6)
        ax.text(0.97, 0.95, f"{occ}", transform=ax.transAxes,
                fontsize=11, ha="right", va="top")
        if j == nrows * ncols - 1:
            ax.set_xlabel("Repeat Number", fontsize=11)
            ax.tick_params(axis="x", which="major", labelbottom=True, labelsize=11)
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", which="major", labelbottom=False)
        ax.tick_params(axis="y", which="major", labelleft=True, labelsize=11)

    for k in range(len(selected_occ_order), nrows * ncols):
        fig.delaxes(axes[k])

    fig.supylabel("Relative Frequency", fontsize=11, x=0.01)
    plt.subplots_adjust(hspace=0, left=0.38)
    fig.suptitle(f"Bird {bird_num_occ}: {syllable_occ}\nOccurrence", y=1.03, x=0.64, fontsize=11, ha="center")
    out = os.path.join(save_dir, "fig_3_c_occ.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Position example: Bird 1, syllable U ─────────────────────────────────────

bird_num_pos = 1
syllable_pos = "U"

df_pos_ex = full_pos_df[
    (full_pos_df["bird_num"].astype(str) == str(bird_num_pos)) &
    (full_pos_df["syllable"].str.upper() == syllable_pos)
]

if df_pos_ex.empty:
    print(f"No position data for bird {bird_num_pos} syllable {syllable_pos}.")
else:
    quartiles = sorted(df_pos_ex["quartile"].dropna().unique())
    max_rn    = int(df_pos_ex["repeat_number"].max())
    nrows, ncols = 4, 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.9, 2.6), sharex=True, sharey=True, dpi=300)
    axes = axes.flatten()

    for j, pos in enumerate(quartiles):
        ax = axes[j]
        data = df_pos_ex[df_pos_ex["quartile"] == pos]
        if not data.empty:
            bins = range(0, max_rn + 2)
            counts, _ = np.histogram(data["repeat_number"], bins=bins)
            rel_freq = counts / counts.sum() if counts.sum() > 0 else counts
            ax.bar(bins[:-1], rel_freq, width=1.0, color=POS_COLOR, edgecolor="k")

        ax.set_xticks(np.arange(0, max_rn + 5, 8))
        ax.set_xticks(np.arange(0, max_rn + 3, 4), minor=True)
        ax.set_xlim(0, max_rn + 2.0)
        ax.set_yticks(np.arange(0, 0.41, 0.4))
        ax.set_yticks(np.arange(0, 0.61, 0.2), minor=True)
        ax.set_ylim(0.0, 0.6)
        ax.text(0.97, 0.95, f"Q{int(pos)}", transform=ax.transAxes,
                fontsize=11, ha="right", va="top")
        if j == nrows * ncols - 1:
            ax.set_xlabel("Repeat Number", fontsize=11)
            ax.tick_params(axis="x", which="major", labelbottom=True, labelsize=11)
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", which="major", labelbottom=False)
        ax.tick_params(axis="y", which="major", labelleft=True, labelsize=11)

    for k in range(len(quartiles), nrows * ncols):
        fig.delaxes(axes[k])

    fig.supylabel("Relative Frequency", fontsize=11, x=0.01)
    plt.subplots_adjust(hspace=0, left=0.38)
    fig.suptitle(f"Bird {bird_num_pos}: {syllable_pos}\nPosition", y=1.03, x=0.64, fontsize=11, ha="center")
    out = os.path.join(save_dir, "fig_3_e_pos.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
