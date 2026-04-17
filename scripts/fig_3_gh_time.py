"""fig_3_time.py

For all birds, collects repeat-number data binned by time of day (2-hour windows),
applies a 5% threshold, saves raw and median CSVs, runs a linear regression
(repeat ~ hour_block), and produces two Figure 3 panels:

  Example panel (single bird/syllable, stacked histograms):
    fig_3_time_example.png  — Bird 2, syllable C, hour blocks 6, 10, 14, 18

  All-birds summary panel (median repeat number vs time of day):
    fig_3_time_all_birds.png — each bird/syllable as a grey line, overall median
                               as a red dashed line, Bird 2/C circled in blue

Output CSVs (saved to output/Time of day csvs/):
  repeat_by_time_median.csv  — median repeat number per bird/syllable/hour block
  time_regression.csv        — linear regression results with FDR + Bonferroni

Output figures (saved to figures/Figure 3/):
  fig_3_time_example.png
  fig_3_time_all_birds.png

Prerequisites:
  None — this script generates all required CSVs.
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import MultipleLocator
from scipy import stats
from statsmodels.stats.multitest import multipletests

from get_compressed_syntax import load_and_process_data
from get_repeat_data import REPO_ROOT, syllables_mapping

# --- Paths ---
yaml_path = os.path.join(REPO_ROOT, "yamls", "bird_1.yaml")
with open(yaml_path, "r") as f:
    config = yaml.safe_load(f)

output_dir = os.path.join(REPO_ROOT, config["time_csvs_path"])
save_dir   = os.path.join(REPO_ROOT, "figures", "Figure 3")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

TIME_WINDOW   = 2      # hours
PCT_THRESHOLD = 5.0

# Bird-specific regex patterns for extracting time from wav_file
# bird 2 (pk02gr02): .wav file
# bird 6 (wh04br04): _partN.wav file
# all others: .cbin file
TIME_REGEX = {
    "2": r"_(\d{6})\.\d+\.wav$",
    "6": r"_(\d{8,})_part\d+\.wav$",
}
TIME_REGEX_DEFAULT = r"_(\d{6})\.-?\d+"


def filter_by_threshold(df, group_col, pct_threshold=PCT_THRESHOLD):
    counts = df[group_col].value_counts()
    freqs  = counts / counts.sum() * 100
    valid  = freqs[freqs >= pct_threshold].index
    return df[df[group_col].isin(valid)].copy()


# ══════════════════════════════════════════════════════════════════════════════
# 1. Data collection
# ══════════════════════════════════════════════════════════════════════════════

summary_time = []

for bird_num, sylls in syllables_mapping.items():
    print(f"\n===== Bird {bird_num} =====")
    df = load_and_process_data(bird_num)
    if "song_id" not in df.columns:
        df["song_id"] = df.index.astype(str)

    pattern = TIME_REGEX.get(bird_num, TIME_REGEX_DEFAULT)
    df["time_str"] = df["wav_file"].str.extract(pattern)[0]
    df = df.dropna(subset=["time_str"])
    if df.empty:
        print(f"  No valid time_str for bird {bird_num}, skipping.")
        continue

    df["hour"]       = df["time_str"].str[:2].astype(int)
    df["hour_block"] = (df["hour"] // TIME_WINDOW) * TIME_WINDOW

    for syll in sylls:
        time_data = []
        for song_id, hour_block, song in zip(df["song_id"], df["hour_block"], df["compressed_song"]):
            for s, n in re.findall(r"([a-zA-Z])(\d+)", song):
                if s == syll:
                    time_data.append((hour_block, int(n), song_id))
        if time_data:
            df_t = pd.DataFrame(time_data, columns=["hour_block", "repeat_number", "song_id"])
            df_t = filter_by_threshold(df_t, "hour_block")
            if not df_t.empty:
                df_t["bird_num"]      = int(bird_num)
                df_t["syllable"]      = syll.upper()
                df_t["bird_syllable"] = f"{bird_num}_{syll.upper()}"
                summary_time.append(df_t)

if not summary_time:
    raise RuntimeError("No time-of-day data collected. Check wav_file format and regex patterns.")

full_time_df = pd.concat(summary_time, ignore_index=True)
full_time_df = full_time_df[["bird_num", "syllable", "bird_syllable", "hour_block", "repeat_number", "song_id"]]

median_time_df = (
    full_time_df
    .groupby(["bird_num", "syllable", "bird_syllable", "hour_block"], as_index=False)["repeat_number"]
    .median()
    .rename(columns={"repeat_number": "median_repeat_number"})
)
median_time_df.to_csv(os.path.join(output_dir, "repeat_by_time_median.csv"), index=False)
print(f"Saved: {os.path.join(output_dir, 'repeat_by_time_median.csv')}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Linear regression: repeat_number ~ hour_block
# ══════════════════════════════════════════════════════════════════════════════

time_results = []

for bird_syll, group in full_time_df.groupby("bird_syllable"):
    x = group["hour_block"].values
    y = group["repeat_number"].values
    if len(x) < 3:
        continue
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    rpt_std  = np.std(y, ddof=1)
    rpt_mean = np.mean(y)
    time_results.append({
        "bird_syllable":       bird_syll,
        "slope":               slope,
        "intercept":           intercept,
        "r_squared":           r_value ** 2,
        "p_value_uncorrected": p_value,
        "std_err":             std_err,
        "n":                   len(x),
        "repeat_mean":         rpt_mean,
        "repeat_std":          rpt_std,
    })

reg_df = pd.DataFrame(time_results)
if not reg_df.empty:
    _, p_fdr,  _, _ = multipletests(reg_df["p_value_uncorrected"], method="fdr_bh")
    _, p_bonf, _, _ = multipletests(reg_df["p_value_uncorrected"], method="bonferroni")
    reg_df["p_value_fdr"]         = p_fdr
    reg_df["p_value_bonferroni"]  = p_bonf
    reg_df["normalised_slope"]    = reg_df["slope"] / reg_df["repeat_std"]
    reg_fp = os.path.join(output_dir, "time_regression.csv")
    reg_df.to_csv(reg_fp, index=False)
    print(f"Saved: {reg_fp}")

    sig_time = reg_df[reg_df["p_value_fdr"] < 0.05]
    if not sig_time.empty:
        s = sig_time["slope"].values
        sn = sig_time["normalised_slope"].values
        print(f"  FDR-significant slopes: range=({s.min():.3f}, {s.max():.3f}), mean={s.mean():.3f}, SE={s.std(ddof=1)/len(s)**0.5:.3f}")
        print(f"  Standardised: range=({sn.min():.3f}, {sn.max():.3f}), mean={sn.mean():.3f}, SE={sn.std(ddof=1)/len(sn)**0.5:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Example panel: Bird 2, syllable C, hour blocks 6, 10, 14, 18
# ══════════════════════════════════════════════════════════════════════════════

TIME_COLOR      = "#b21218ff"
bird_num_time   = 2
syllable_time   = "C"
selected_hours  = [6, 10, 14, 18]

df_time_ex = full_time_df[
    (full_time_df["bird_num"] == bird_num_time) &
    (full_time_df["syllable"] == syllable_time)
]

if df_time_ex.empty:
    print(f"No time data for bird {bird_num_time} syllable {syllable_time}.")
else:
    max_rn = int(df_time_ex["repeat_number"].max())
    nrows, ncols = 4, 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.9, 2.6), sharex=True, sharey=True, dpi=300)
    axes = axes.flatten()

    for j, hour in enumerate(selected_hours):
        ax   = axes[j]
        data = df_time_ex[df_time_ex["hour_block"] == hour]
        if not data.empty:
            bins     = range(0, max_rn + 2)
            counts, _ = np.histogram(data["repeat_number"], bins=bins)
            rel_freq  = counts / counts.sum() if counts.sum() > 0 else counts
            ax.bar(bins[:-1], rel_freq, width=1.0, color=TIME_COLOR, edgecolor="k")

        ax.set_xticks(np.arange(0, max_rn + 5, 4))
        ax.set_xticks(np.arange(0, max_rn + 3, 2), minor=True)
        ax.set_xlim(0, max_rn + 1.5)
        ax.set_yticks(np.arange(0, 0.41, 0.4))
        ax.set_yticks(np.arange(0, 0.61, 0.2), minor=True)
        ax.set_ylim(0.0, 0.6)
        ax.text(0.97, 0.95, f"{hour}–{hour + TIME_WINDOW} h",
                transform=ax.transAxes, fontsize=11, ha="right", va="top")
        if j == nrows * ncols - 1:
            ax.set_xlabel("Repeat Number", fontsize=11)
            ax.tick_params(axis="x", which="major", labelbottom=True, labelsize=11)
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", which="major", labelbottom=False)
        ax.tick_params(axis="y", which="major", labelleft=True, labelsize=11)

    for k in range(len(selected_hours), nrows * ncols):
        fig.delaxes(axes[k])

    fig.supylabel("Relative Frequency", fontsize=11, x=0.01)
    plt.subplots_adjust(hspace=0, left=0.38)
    fig.suptitle(f"Bird {bird_num_time}: {syllable_time}\nTime of Day",
                 y=1.03, x=0.64, fontsize=11, ha="center")
    out = os.path.join(save_dir, "fig_3_time_example.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace('.png', '.svg'), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. All-birds panel: median repeat number vs time of day
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(3.8, 3.2), dpi=300)

for (bird, syl), group in median_time_df.groupby(["bird_num", "syllable"]):
    ax.plot(group["hour_block"] + 1, group["median_repeat_number"],
            marker="o", linewidth=1, color="#888888", markersize=3)

overall_time = median_time_df.groupby("hour_block")["median_repeat_number"].median().reset_index()
ax.plot(overall_time["hour_block"] + 1, overall_time["median_repeat_number"],
        marker="o", linestyle="--", color="red", markersize=3, linewidth=2, zorder=5)

highlight = median_time_df[(median_time_df["bird_num"] == 2) & (median_time_df["syllable"] == "C")]
ax.scatter(highlight["hour_block"] + 1, highlight["median_repeat_number"],
           s=30, facecolors="none", edgecolors="blue", linewidths=2, zorder=6)

min_hour = int(median_time_df["hour_block"].min())
max_hour = int(median_time_df["hour_block"].max()) + 2
x_major  = np.arange(min_hour, max_hour + 1, 4)
x_minor  = np.arange(min_hour + 2, max_hour + 1, 4)
ax.set_xticks(x_major)
ax.set_xticklabels([str(t) for t in x_major], fontsize=11)
ax.set_xticks(x_minor, minor=True)
ax.yaxis.set_major_locator(MultipleLocator(4))
ax.yaxis.set_minor_locator(MultipleLocator(2))
ax.set_ylim(0, 32)
ax.set_xlabel("Time of Day", fontsize=11)
ax.set_ylabel("Median repeat number", fontsize=11)
ax.tick_params(axis="both", labelsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()

out = os.path.join(save_dir, "fig_3_time_all_birds.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.savefig(out.replace('.png', '.svg'), bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
