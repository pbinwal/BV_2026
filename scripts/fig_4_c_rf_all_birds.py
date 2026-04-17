"""rf_shap_summary.py

Reads per-syllable SHAP importance CSVs for all birds (produced by
rf_by_phrase.py) and generates the all-birds SHAP summary figure (Figure 4).

Prerequisites
-------------
Run rf_by_phrase.py for all 6 birds first so that
  output/RF results/per_phrase_rf_models/bird_{n}/{syl}/bird_{n}_{syl}_shap_importance.csv
  output/RF results/per_phrase_rf_models/bird_{n}/{syl}/bird_{n}_{syl}_rf_results.xlsx
exist for every bird/syllable.

Thresholds
----------
  DELTA_THRESHOLD  : max allowed (CV_R2 - OOB) before excluding a model
  MIN_SAMPLE_SIZE  : minimum sample size required for inclusion

Outputs (saved to figures/Figure 4/)
-------------------------------------
  fig_5_rf_summary.png  — all-birds SHAP summary scatter+bar figure
"""

import collections
import os
import sys

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from get_repeat_data import REPO_ROOT, syllables_mapping

# ── Config ────────────────────────────────────────────────────────────────────
DELTA_THRESHOLD  = 0.1
MIN_SAMPLE_SIZE  = 1000
BASE_DIR  = os.path.join(REPO_ROOT, "output", "RF results", "per_phrase_rf_models")
SAVE_DIR  = os.path.join(REPO_ROOT, "figures", "Figure 4")
os.makedirs(SAVE_DIR, exist_ok=True)

birds = list(syllables_mapping.keys())   # ["1", "2", ..., "6"]

# ── Colour / hatch scheme ─────────────────────────────────────────────────────
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
feature_color_map["next_rpt_num_1"] = fade_to_white(base_blue, fade_factors[1])
feature_hatch_map["next_rpt_num_1"] = "///"

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

features_to_keep = {
    "prev_syl_2", "rpt_num_prev_2",
    "prev_syl_1", "rpt_num_prev_1",
    "next_syl",   "next_syl_1",   "next_rpt_num",   "next_rpt_num_1",
    "next_syl_2", "next_rpt_num_2",
    "song_length", "target_occurrence_num", "target_relative_pos", "recording_hour",
}

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

bird_markers = {"1": "o", "2": "v", "3": "^", "4": "D", "5": "s", "6": "P"}

# ── Helpers ───────────────────────────────────────────────────────────────────
import re
from collections import defaultdict

def group_individual_features(features_list):
    grouped = defaultdict(list)
    pattern = r"^(target_syl|prev_syl_\d+|next_syl(?:_\d+)?)_[a-zA-Z]$"
    for f in features_list:
        if re.match(pattern, f):
            grouped["_".join(f.split("_")[:-1])].append(f)
        else:
            grouped[f].append(f)
    return grouped

def read_r2_score(syl_dir, bird_num, syl):
    xlsx = os.path.join(syl_dir, f"bird_{bird_num}_{syl}_rf_results.xlsx")
    if os.path.isfile(xlsx):
        try:
            df = pd.read_excel(xlsx)
            r2         = float(df["CV_R2_Mean"].iloc[0])  if "CV_R2_Mean"       in df.columns else None
            sample     = int(df["Sample_Size"].iloc[0])   if "Sample_Size"      in df.columns else None
            oob        = float(df["OOB_Score"].iloc[0])   if "OOB_Score"        in df.columns else None
            target_std = float(df["Target_RptNum_Std"].iloc[0]) if "Target_RptNum_Std" in df.columns else None
            return r2, sample, oob, target_std
        except Exception as e:
            print(f"[WARNING] Could not read {xlsx}: {e}")
    return None, None, None, None

# ── Collect data for all birds ────────────────────────────────────────────────
stacked_bird_sylls       = []
stacked_r2               = []
stacked_sample           = []
stacked_oob              = []
stacked_grouped_shap     = []
stacked_all_features     = []
valid_bird_indices       = []

for ax_idx, bird_num in enumerate(birds):
    bird_dir = os.path.join(BASE_DIR, f"bird_{bird_num}")
    all_sylls = syllables_mapping[bird_num]
    sylls, r2_scores, sample_sizes, oob_scores, target_stds = [], {}, {}, {}, {}
    grouped_shap_per_syl = {}
    all_grouped_features = set()

    for syl in all_sylls:
        syl_dir = os.path.join(bird_dir, syl)
        r2, sample, oob, tstd = read_r2_score(syl_dir, bird_num, syl)
        if (
            r2 is not None and
            sample is not None and sample >= MIN_SAMPLE_SIZE and
            oob is not None and (r2 - oob) <= DELTA_THRESHOLD
        ):
            shap_csv = os.path.join(syl_dir, f"bird_{bird_num}_{syl}_shap_importance.csv")
            if os.path.isfile(shap_csv):
                df = pd.read_csv(shap_csv)
                if tstd:
                    df["SHAP_Importance"] /= tstd
                gf = group_individual_features(df["Feature"].tolist())
                gs = {}
                for gname, feats in gf.items():
                    gs[gname] = df[df["Feature"].isin(feats)]["SHAP_Importance"].sum()
                    all_grouped_features.add(gname)
                grouped_shap_per_syl[syl] = gs
                sylls.append(syl)
                r2_scores[syl]    = r2
                sample_sizes[syl] = sample
                oob_scores[syl]   = oob
                target_stds[syl]  = tstd

    if sylls:
        valid_bird_indices.append(ax_idx)

    stacked_bird_sylls.append(sylls)
    stacked_r2.append(r2_scores)
    stacked_sample.append(sample_sizes)
    stacked_oob.append(oob_scores)
    stacked_grouped_shap.append(grouped_shap_per_syl)
    stacked_all_features.append(sorted(all_grouped_features))
if not valid_bird_indices:
    sys.exit(
        "ERROR: No Random Forest model output found for any bird.\n"
        "Please run rf_by_phrase.py for each bird (1-6) first, then re-run this script."
    )

missing = [birds[i] for i in range(len(birds)) if i not in valid_bird_indices]
if missing:
    print(f"[WARNING] No valid RF output found for Bird(s): {', '.join(missing)}. "
          "Run rf_by_phrase.py for those birds to include them.")

# ── Print model summary statistics ───────────────────────────────────────────
all_r2_vals = []
print("\n[MODEL SUMMARY] Models passing thresholds (delta_R2<={}, n>={}):".format(
    DELTA_THRESHOLD, MIN_SAMPLE_SIZE))
total_models = 0
for ax_idx in valid_bird_indices:
    bird_num = birds[ax_idx]
    sylls = stacked_bird_sylls[ax_idx]
    r2_vals = [stacked_r2[ax_idx][s] for s in sylls]
    all_r2_vals.extend(r2_vals)
    total_models += len(sylls)
    r2_str = ", ".join(f"{s}={v:.3f}" for s, v in zip(sylls, r2_vals))
    print(f"  Bird {bird_num}: {len(sylls)} model(s) - {r2_str}")

if all_r2_vals:
    print(f"\n  Total models in figure : {total_models}")
    print(f"  R2 range               : {min(all_r2_vals):.3f} - {max(all_r2_vals):.3f}")
    print(f"  Mean R2                : {np.mean(all_r2_vals):.3f}")
print()

# ── Compute global feature set (±2 + other) ───────────────────────────────────
all_features_global = set()
for ax_idx in valid_bird_indices:
    for g in stacked_all_features[ax_idx]:
        if g in features_to_keep:
            all_features_global.add(g)

feature_groups_global = (
    [g for g in custom_feature_order if g in all_features_global and g in features_to_keep] +
    [g for g in sorted(all_features_global) if g not in custom_feature_order and g in features_to_keep]
)

mean_shap = {}
for group in feature_groups_global:
    vals = [
        stacked_grouped_shap[ax_idx].get(syl, {}).get(group, 0)
        for ax_idx in valid_bird_indices
        for syl in stacked_bird_sylls[ax_idx]
    ]
    mean_shap[group] = np.mean(vals) if vals else 0

# ── Figure: summary bar + scatter ────────────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(8, 3))

for j, group in enumerate(feature_groups_global):
    color = feature_color_map.get(group, "#cccccc")
    hatch = feature_hatch_map.get(group, None)
    ax4.bar(j, mean_shap[group], width=0.6, color=color,
            hatch=hatch or "", edgecolor="white" if hatch else "black", linewidth=0.5, zorder=1)
    if hatch:
        ax4.bar(j, mean_shap[group], width=0.6, color="none",
                hatch="", edgecolor="black", linewidth=1.5, zorder=2)

# Dashed line between -1 and +1
m1 = [i for i, g in enumerate(feature_groups_global) if feature_display_labels.get(g) == "-1"]
p1 = [i for i, g in enumerate(feature_groups_global) if feature_display_labels.get(g) == "+1"]
if m1 and p1:
    ax4.axvline((max(m1) + min(p1)) / 2, color="gray", linestyle="--", linewidth=1, zorder=0)

for ax_idx in valid_bird_indices:
    bird_num = birds[ax_idx]
    marker = bird_markers.get(bird_num, "o")
    for syl in stacked_bird_sylls[ax_idx]:
        xs = list(range(len(feature_groups_global)))
        ys = [stacked_grouped_shap[ax_idx].get(syl, {}).get(g, 0) for g in feature_groups_global]
        ax4.scatter(xs, ys, marker=marker, s=40, color="black",
                    alpha=0.7, edgecolors="black", linewidths=0.5, zorder=3)

label_to_idx = collections.defaultdict(list)
for idx, g in enumerate(feature_groups_global):
    label_to_idx[feature_display_labels.get(g, g)].append(idx)

ax4.set_xticks([np.mean(v) for v in label_to_idx.values()])
ax4.set_xticklabels(list(label_to_idx.keys()), rotation=0, ha="center", fontsize=10)
ax4.tick_params(axis="y", labelsize=10)
ax4.set_ylabel("Normalised SHAP Importance", fontsize=11)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)
ax4.axhline(0, color="gray", linewidth=0.5, linestyle="--")

bird_legend = [
    plt.scatter([], [], marker=bird_markers.get(b, "o"), s=40,
                edgecolors="black", linewidths=0.5, label=f"Bird {b}", color="black")
    for b in birds if any(birds[i] == b for i in valid_bird_indices)
]
ax4.legend(handles=bird_legend, frameon=False, fontsize=11, loc="upper right")

fig4.tight_layout()
fig4.savefig(os.path.join(SAVE_DIR, "fig_5_rf_summary.png"), dpi=300, bbox_inches="tight")
fig4.savefig(os.path.join(SAVE_DIR, "fig_5_rf_summary.svg"), bbox_inches="tight")
print(f"Saved: {os.path.join(SAVE_DIR, 'fig_5_rf_summary.png')}")
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

all_features_global_supp = set()
for ax_idx in valid_bird_indices:
    for g in stacked_all_features[ax_idx]:
        if g in features_to_keep_supp:
            all_features_global_supp.add(g)

feature_groups_supp = (
    [g for g in custom_feature_order if g in all_features_global_supp and g in features_to_keep_supp] +
    [g for g in sorted(all_features_global_supp) if g not in custom_feature_order and g in features_to_keep_supp]
)

mean_shap_supp = {}
for group in feature_groups_supp:
    vals = [
        stacked_grouped_shap[ax_idx].get(syl, {}).get(group, 0)
        for ax_idx in valid_bird_indices
        for syl in stacked_bird_sylls[ax_idx]
    ]
    mean_shap_supp[group] = np.mean(vals) if vals else 0

fig_s, ax_s = plt.subplots(figsize=(8, 3))

for j, group in enumerate(feature_groups_supp):
    color = feature_color_map.get(group, "#cccccc")
    hatch = feature_hatch_map.get(group, None)
    ax_s.bar(j, mean_shap_supp[group], width=0.6, color=color,
             hatch=hatch or "", edgecolor="white" if hatch else "black", linewidth=0.5, zorder=1)
    if hatch:
        ax_s.bar(j, mean_shap_supp[group], width=0.6, color="none",
                 hatch="", edgecolor="black", linewidth=1.5, zorder=2)

m1_s = [i for i, g in enumerate(feature_groups_supp) if feature_display_labels.get(g) == "-1"]
p1_s = [i for i, g in enumerate(feature_groups_supp) if feature_display_labels.get(g) == "+1"]
if m1_s and p1_s:
    ax_s.axvline((max(m1_s) + min(p1_s)) / 2, color="gray", linestyle="--", linewidth=1, zorder=0)

for ax_idx in valid_bird_indices:
    bird_num = birds[ax_idx]
    marker = bird_markers.get(bird_num, "o")
    for syl in stacked_bird_sylls[ax_idx]:
        xs = list(range(len(feature_groups_supp)))
        ys = [stacked_grouped_shap[ax_idx].get(syl, {}).get(g, 0) for g in feature_groups_supp]
        ax_s.scatter(xs, ys, marker=marker, s=40, color="black",
                     alpha=0.7, edgecolors="black", linewidths=0.5, zorder=3)

label_to_idx_s = collections.defaultdict(list)
for idx, g in enumerate(feature_groups_supp):
    label_to_idx_s[feature_display_labels.get(g, g)].append(idx)

ax_s.set_xticks([np.mean(v) for v in label_to_idx_s.values()])
ax_s.set_xticklabels(list(label_to_idx_s.keys()), rotation=0, ha="center", fontsize=10)
ax_s.tick_params(axis="y", labelsize=10)
ax_s.set_ylabel("Normalised SHAP Importance", fontsize=11)
ax_s.spines["top"].set_visible(False)
ax_s.spines["right"].set_visible(False)
ax_s.axhline(0, color="gray", linewidth=0.5, linestyle="--")

bird_legend_s = [
    plt.scatter([], [], marker=bird_markers.get(b, "o"), s=40,
                edgecolors="black", linewidths=0.5, label=f"Bird {b}", color="black")
    for b in birds if any(birds[i] == b for i in valid_bird_indices)
]
ax_s.legend(handles=bird_legend_s, frameon=False, fontsize=11, loc="upper right")

fig_s.tight_layout()
supp2_dir = os.path.join(REPO_ROOT, "figures", "Supplementary", "Supp_3")
os.makedirs(supp2_dir, exist_ok=True)
supp2_path = os.path.join(supp2_dir, "supp_3.png")
fig_s.savefig(supp2_path, dpi=300, bbox_inches="tight")
fig_s.savefig(supp2_path.replace('.png', '.svg'), bbox_inches="tight")
print(f"Saved: {supp2_path}")
plt.show()
