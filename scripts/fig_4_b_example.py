"""fig_5_shap_plot.py

Reads the SHAP importance CSV for a single bird/syllable (produced by
rf_by_phrase.py) and plots a grouped vertical bar chart.

Usage
-----
Run the script; it will prompt for bird number and syllable label.

Outputs (saved to figures/Figure 4/)
-------------------------------------
  fig_5_bird_{n}_{syl}_grouped_shap_vertical.png
"""

import os
import re
import sys
from collections import defaultdict

import collections
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

from get_repeat_data import REPO_ROOT, syllables_mapping

# ── Colour / hatch scheme (matches rf_shap_summary.py) ───────────────────────
reds      = plt.get_cmap("Reds", 7)
base_blue = mcolors.to_rgb("#2171b5")

def fade_to_white(color, factor):
    return tuple((1 - factor) * c + factor * 1.0 for c in color)

fade_factors = {4: 0.90, 3: 0.60, 2: 0.30, 1: 0.0}
feature_color_map = {}
feature_hatch_map = {}

for num in [1, 2, 3, 4]:
    color = fade_to_white(base_blue, fade_factors[num])
    for prefix in (f"prev_syl_{num}", f"rpt_num_prev_{num}",
                   f"next_syl_{num}",  f"next_rpt_num_{num}"):
        feature_color_map[prefix] = color
        feature_hatch_map[prefix] = "///" if "rpt_num" in prefix else None

for fallback in ("next_syl", "next_rpt_num"):
    feature_color_map[fallback] = fade_to_white(base_blue, fade_factors[1])
    feature_hatch_map[fallback] = "///" if "rpt_num" in fallback else None

other_features = ["song_length", "target_occurrence_num", "target_relative_pos", "recording_hour"]
for i, g in enumerate(other_features):
    feature_color_map[g] = reds((i + 2) / 7)
    feature_hatch_map[g] = None

custom_feature_order = [
    "prev_syl_4", "rpt_num_prev_4",
    "prev_syl_3", "rpt_num_prev_3",
    "prev_syl_2", "rpt_num_prev_2",
    "prev_syl_1", "rpt_num_prev_1",
    "next_syl", "next_syl_1", "next_rpt_num", "next_rpt_num_1",
    "next_syl_2", "next_rpt_num_2",
    "next_syl_3", "next_rpt_num_3",
    "next_syl_4", "next_rpt_num_4",
    "song_length", "target_occurrence_num", "target_relative_pos", "recording_hour",
]

feature_display_labels = {
    "prev_syl_4":            "-4",
    "rpt_num_prev_4":        "-4",
    "prev_syl_3":            "-3",
    "rpt_num_prev_3":        "-3",
    "prev_syl_2":            "-2",
    "rpt_num_prev_2":        "-2",
    "prev_syl_1":            "-1",
    "rpt_num_prev_1":        "-1",
    "next_syl":              "+1",
    "next_syl_1":            "+1",
    "next_rpt_num":          "+1",
    "next_rpt_num_1":        "+1",
    "next_syl_2":            "+2",
    "next_rpt_num_2":        "+2",
    "next_syl_3":            "+3",
    "next_rpt_num_3":        "+3",
    "next_syl_4":            "+4",
    "next_rpt_num_4":        "+4",
    "song_length":           "song\nlen.",
    "target_occurrence_num": "occ.\nnum.",
    "target_relative_pos":   "rel.\npos.",
    "recording_hour":        "t.o.d",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def group_individual_features(features_list):
    grouped = defaultdict(list)
    pattern = r"^(target_syl|prev_syl_\d+|next_syl(?:_\d+)?)_[a-zA-Z]$"
    for f in features_list:
        if re.match(pattern, f):
            grouped["_".join(f.split("_")[:-1])].append(f)
        else:
            grouped[f].append(f)
    return grouped

# ── User input ────────────────────────────────────────────────────────────────
bird_num = input("Enter bird number (1-6): ").strip()
if bird_num not in syllables_mapping:
    raise ValueError(f"Bird {bird_num!r} not found in syllables_mapping.")

available = syllables_mapping[bird_num]
print(f"Available syllables for Bird {bird_num}: {available}")
syl = input("Enter syllable label: ").strip()
if syl not in available:
    raise ValueError(f"Syllable {syl!r} not in syllables for Bird {bird_num}.")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(REPO_ROOT, "output", "RF results", "per_phrase_rf_models")
SAVE_DIR = os.path.join(REPO_ROOT, "figures", "Figure 4")
os.makedirs(SAVE_DIR, exist_ok=True)

syl_dir  = os.path.join(BASE_DIR, f"bird_{bird_num}", syl)
shap_csv = os.path.join(syl_dir, f"bird_{bird_num}_{syl}_shap_importance.csv")
xlsx     = os.path.join(syl_dir, f"bird_{bird_num}_{syl}_rf_results.xlsx")

if not os.path.isfile(shap_csv):
    sys.exit(
        f"ERROR: No Random Forest model output found for Bird {bird_num}, syllable '{syl}'.\n"
        f"Please run rf_by_phrase.py for Bird {bird_num} first, then re-run this script."
    )

# ── Load and normalise ────────────────────────────────────────────────────────
df = pd.read_csv(shap_csv)

target_std = None
if os.path.isfile(xlsx):
    try:
        rf_df = pd.read_excel(xlsx)
        if "Target_RptNum_Std" in rf_df.columns:
            target_std = float(rf_df["Target_RptNum_Std"].iloc[0])
    except Exception as e:
        print(f"[WARNING] Could not read {xlsx}: {e}")

if target_std:
    df["SHAP_Importance"] /= target_std
    print(f"Normalised SHAP by target std = {target_std:.4f}")
else:
    print("[WARNING] Target std not found; SHAP values are unnormalised.")

# ── Group one-hot features ────────────────────────────────────────────────────
gf = group_individual_features(df["Feature"].tolist())
grouped_shap = {}
for gname, feats in gf.items():
    grouped_shap[gname] = df[df["Feature"].isin(feats)]["SHAP_Importance"].sum()

# Only show ±2 context features (drop ±3 and ±4) plus the 4 other features
features_to_keep = {
    "prev_syl_2", "rpt_num_prev_2",
    "prev_syl_1", "rpt_num_prev_1",
    "next_syl",   "next_syl_1",   "next_rpt_num",   "next_rpt_num_1",
    "next_syl_2", "next_rpt_num_2",
    "song_length", "target_occurrence_num", "target_relative_pos", "recording_hour",
}

# ── Order features ────────────────────────────────────────────────────────────
ordered_features = (
    [g for g in custom_feature_order if g in grouped_shap and g in features_to_keep] +
    [g for g in sorted(grouped_shap)  if g not in custom_feature_order and g in features_to_keep]
)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 3))

for j, group in enumerate(ordered_features):
    val   = grouped_shap[group]
    color = feature_color_map.get(group, "#cccccc")
    hatch = feature_hatch_map.get(group, None)
    ax.bar(j, val, width=0.6, color=color,
           hatch=hatch or "", edgecolor="white" if hatch else "black", linewidth=0.5, zorder=1)
    if hatch:
        ax.bar(j, val, width=0.6, color="none",
               hatch="", edgecolor="black", linewidth=1.5, zorder=2)

# Dashed line between -1 and +1
m1 = [i for i, g in enumerate(ordered_features) if feature_display_labels.get(g) == "-1"]
p1 = [i for i, g in enumerate(ordered_features) if feature_display_labels.get(g) == "+1"]
if m1 and p1:
    ax.axvline((max(m1) + min(p1)) / 2, color="gray", linestyle="--", linewidth=1, zorder=0)

label_to_idx = collections.defaultdict(list)
for idx, g in enumerate(ordered_features):
    label_to_idx[feature_display_labels.get(g, g)].append(idx)
ax.set_xticks([sum(v) / len(v) for v in label_to_idx.values()])
ax.set_xticklabels(list(label_to_idx.keys()), rotation=0, ha="center", fontsize=10)
ax.tick_params(axis="y", labelsize=10)
ax.set_ylabel("Normalised SHAP Importance", fontsize=11)
ax.set_title(f"Bird {bird_num}: Phrase {syl.upper()}", fontsize=12)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

import matplotlib.patches as mpatches
legend_identity = mpatches.Patch(facecolor="white", edgecolor="black", linewidth=1.0,
                                  label="Phrase identity")
legend_repeat   = mpatches.Patch(facecolor="white", edgecolor="black", linewidth=1.5,
                                  hatch="///", label="Repeat number")
ax.legend(handles=[legend_identity, legend_repeat], frameon=False,
          fontsize=10, loc="upper left")

fig.tight_layout()
stem = f"fig_5_bird_{bird_num}_{syl}_grouped_shap_vertical"
out = os.path.join(SAVE_DIR, f"{stem}.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(SAVE_DIR, f"{stem}.svg"), bbox_inches="tight")
print(f"Saved: {out}")
plt.show()

# ── Supplementary version: context ±4 ────────────────────────────────────────
features_to_keep_supp = {
    "prev_syl_4", "rpt_num_prev_4",
    "prev_syl_3", "rpt_num_prev_3",
    "prev_syl_2", "rpt_num_prev_2",
    "prev_syl_1", "rpt_num_prev_1",
    "next_syl",   "next_syl_1",   "next_rpt_num",   "next_rpt_num_1",
    "next_syl_2", "next_rpt_num_2",
    "next_syl_3", "next_rpt_num_3",
    "next_syl_4", "next_rpt_num_4",
    "song_length", "target_occurrence_num", "target_relative_pos", "recording_hour",
}

ordered_features_supp = (
    [g for g in custom_feature_order if g in grouped_shap and g in features_to_keep_supp] +
    [g for g in sorted(grouped_shap)  if g not in custom_feature_order and g in features_to_keep_supp]
)

fig_s, ax_s = plt.subplots(figsize=(8, 3))

for j, group in enumerate(ordered_features_supp):
    val   = grouped_shap[group]
    color = feature_color_map.get(group, "#cccccc")
    hatch = feature_hatch_map.get(group, None)
    ax_s.bar(j, val, width=0.6, color=color,
             hatch=hatch or "", edgecolor="white" if hatch else "black", linewidth=0.5, zorder=1)
    if hatch:
        ax_s.bar(j, val, width=0.6, color="none",
                 hatch="", edgecolor="black", linewidth=1.5, zorder=2)

m1_s = [i for i, g in enumerate(ordered_features_supp) if feature_display_labels.get(g) == "-1"]
p1_s = [i for i, g in enumerate(ordered_features_supp) if feature_display_labels.get(g) == "+1"]
if m1_s and p1_s:
    ax_s.axvline((max(m1_s) + min(p1_s)) / 2, color="gray", linestyle="--", linewidth=1, zorder=0)

label_to_idx_s = collections.defaultdict(list)
for idx, g in enumerate(ordered_features_supp):
    label_to_idx_s[feature_display_labels.get(g, g)].append(idx)
ax_s.set_xticks([sum(v) / len(v) for v in label_to_idx_s.values()])
ax_s.set_xticklabels(list(label_to_idx_s.keys()), rotation=0, ha="center", fontsize=10)
ax_s.tick_params(axis="y", labelsize=10)
ax_s.set_ylabel("Normalised SHAP Importance", fontsize=11)
ax_s.set_title(f"Bird {bird_num}: Phrase {syl.upper()}", fontsize=12)
ax_s.spines["top"].set_visible(False)
ax_s.spines["right"].set_visible(False)
ax_s.axhline(0, color="gray", linewidth=0.5, linestyle="--")

ax_s.legend(handles=[legend_identity, legend_repeat], frameon=False,
            fontsize=10, loc="upper left")

fig_s.tight_layout()
supp3a_dir = os.path.join(REPO_ROOT, "figures", "Supplementary", "Supp_3")
os.makedirs(supp3a_dir, exist_ok=True)
supp3a_stem = f"supp_3_a_bird_{bird_num}_{syl}"
supp3a_out = os.path.join(supp3a_dir, f"{supp3a_stem}.png")
fig_s.savefig(supp3a_out, dpi=300, bbox_inches="tight")
fig_s.savefig(os.path.join(supp3a_dir, f"{supp3a_stem}.svg"), bbox_inches="tight")
print(f"Saved: {supp3a_out}")
plt.show()
