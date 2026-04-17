"""fig_2_d.py

RUN FOR BIRD 2: c to reproduce Fig 2 D

For a single bird and syllable of interest, visualises the significant
correlations between that syllable's repeat count and every other syllable's
repeat count as a function of relative phrase position (Figure 2D).

Usage
-----
Run the script and enter the bird number and syllable when prompted.

Prerequisites
-------------
Run fig_2_synth_control.py for all birds and then fig_2_g.py once so that
  output/Correlations by distance z corrected/all_birds_corr_by_dist.csv
exists.

Outputs (saved to figures/Figure 2/)
-------------------------------------
  bird_{n}_{syl}_eg.png         — bar chart clipped to ±2 positions
  bird_{n}_{syl}_eg_by_distance.png — same with absolute-distance x-axis
  bird_{n}_all_syls.png          — all-syllable subplot grid for bird
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from get_repeat_data import REPO_ROOT, syllables_mapping

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(
    REPO_ROOT, "output", "Correlations by distance z corrected",
    "all_birds_corr_by_dist.csv",
)
MAX_DIST = 10
SAVE_DIR = os.path.join(REPO_ROOT, "figures", "Figure 2")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

# ── User input ────────────────────────────────────────────────────────────────
available = ", ".join(syllables_mapping.keys())
bird_num = input(f"Available birds: {available}\nEnter bird number: ").strip()
if bird_num not in syllables_mapping:
    print(f"Invalid bird number. Available: {available}")
    raise SystemExit

syl = input(f"Syllables for bird {bird_num}: {syllables_mapping[bird_num]}\nEnter syllable: ").strip()

bird_id   = f"bird_{bird_num}"
bird_label = f"Bird {bird_num}"

# ── Helper: build position summary for one syllable ───────────────────────────
def build_summary(df_in, syl_i):
    """Return a DataFrame with columns [Position, Other_Syl, Correlation]."""
    syl_df = df_in[
        (df_in["Bird_ID"] == bird_id) &
        (df_in["Syllable_Pair"].str.contains(syl_i, regex=False)) &
        (df_in["Significance"] == "significant") &
        (df_in["Distance"] <= MAX_DIST)
    ].reset_index(drop=True)

    if syl_df.empty:
        return pd.DataFrame(columns=["Position", "Other_Syl", "Correlation"])

    rows = []
    for _, row in syl_df.iterrows():
        left, right = [s.strip() for s in row["Syllable_Pair"].split("->")]
        dist = int(row["Distance"])
        corr = row["Correlation"]
        if left == syl_i and right == syl_i:
            rows.append({"Position": -(dist + 1), "Other_Syl": syl_i.upper(), "Correlation": corr})
            rows.append({"Position": +(dist + 1), "Other_Syl": syl_i.upper(), "Correlation": corr})
        elif right == syl_i:
            rows.append({"Position": -(dist + 1), "Other_Syl": left.upper(), "Correlation": corr})
        else:
            rows.append({"Position": +(dist + 1), "Other_Syl": right.upper(), "Correlation": corr})

    return (
        pd.DataFrame(rows)
        .sort_values("Position")
        .reset_index(drop=True)
        .query(f"Position.between(-{MAX_DIST}, {MAX_DIST})")
        .reset_index(drop=True)
    )


# ── Helper: draw bars on an axis ──────────────────────────────────────────────
def draw_bars(ax, summary_df, positions=None, abs_x=False):
    bar_width_single = 0.4
    plist = sorted(summary_df["Position"].unique()) if positions is None else positions
    for pos in plist:
        pos_rows = summary_df[summary_df["Position"] == pos].reset_index(drop=True)
        if pos_rows.empty:
            continue
        n = len(pos_rows)
        offsets = np.linspace(-(n - 1) * bar_width_single / 2,
                               (n - 1) * bar_width_single / 2, n)
        bw = bar_width_single if n == 1 else bar_width_single * 0.9
        x_base = abs(pos) * np.sign(pos) if abs_x else pos
        for i, (_, row) in enumerate(pos_rows.iterrows()):
            corr  = row["Correlation"]
            other = row["Other_Syl"]
            color = "lightgrey"
            x = x_base + offsets[i]
            ax.bar(x, corr, width=bw, color=color, edgecolor="black",
                   linewidth=0.5, zorder=2)
            va   = "bottom" if corr >= 0 else "top"
            ypos = corr + (0.005 if corr >= 0 else -0.005)
            ax.text(x, ypos, other, ha="center", va=va, fontsize=8, color="black")


# ── Single-syllable summary ───────────────────────────────────────────────────
summary_df = build_summary(df, syl)

# Print text summary
print(f"\n=== Summary for syllable '{syl.upper()}' (positions -{MAX_DIST} to +{MAX_DIST}) ===")
for pos in range(-MAX_DIST, MAX_DIST + 1):
    pos_rows = summary_df[summary_df["Position"] == pos]
    if pos_rows.empty:
        continue
    label = f"{'+'if pos>0 else ''}{pos}"
    entries = [f"{r['Other_Syl']} (r={r['Correlation']:.3f})" for _, r in pos_rows.iterrows()]
    print(f"  {label:>3}  |  {len(pos_rows)} correlation(s)  |  {',  '.join(entries)}")

positions_present = sorted(summary_df["Position"].unique())

# ── Plot: by-distance bar chart (abs x-axis labels, full extent) ──────────────
max_pos = max(max(abs(p) for p in positions_present) if positions_present else 0, 10)
x_limit = max_pos + 1  # one unit of padding beyond last data point

fig2, ax2 = plt.subplots(figsize=(2.5, 1.5))
draw_bars(ax2, summary_df, positions=positions_present, abs_x=True)

ax2.set_ylim(-0.8, 0.8)
ax2.set_yticks([-0.8, 0, 0.8])
ax2.set_yticklabels(["-0.8", "0", "+0.8"], fontsize=8)
ax2.set_yticks([-0.4, 0.4], minor=True)
ax2.tick_params(axis="y", labelsize=10, which="both")
tick_positions = list(range(-max_pos, max_pos + 1, 5))
if 0 not in tick_positions:
    tick_positions = sorted(set(tick_positions + [0]))
ax2.set_xticks(tick_positions)
ax2.set_xticklabels([str(abs(p)) for p in tick_positions], fontsize=8)
ax2.set_xticks(list(range(-max_pos, max_pos + 1)), minor=True)
ax2.tick_params(axis="x", labelsize=10, which="both")
ax2.axhline(0, color="black", linewidth=0.8)
ax2.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
ax2.set_xlabel(f"Distance from {syl.upper()}", fontsize=10)
ax2.set_ylabel("Correlation", fontsize=10)
ax2.set_title(f"{bird_label}: {syl.upper()}", fontsize=10)
ax2.set_xlim(-x_limit, x_limit)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
fig_name2 = f"bird_{bird_num}_{syl}_eg_by_distance"
fig2.savefig(os.path.join(SAVE_DIR, f"{fig_name2}.png"), dpi=300, bbox_inches="tight")
fig2.savefig(os.path.join(SAVE_DIR, f"{fig_name2}.svg"), bbox_inches="tight")
print(f"Saved: {fig_name2}.png")
plt.show()
