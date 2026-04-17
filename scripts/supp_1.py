"""
Supplementary Figure 1: Three z-test distribution panels for bird 2
  Panel 1: C->B adjacent
  Panel 2: C->E next
  Panel 3: B->C next
Arranged side by side, A4 width x 4 inches height, single shared legend at bottom centre.

Output: figures/Supplementary/Supp_1/supp_1.png  (+ .svg)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import yaml
from scipy.stats import norm
import matplotlib.lines as mlines

# ── Config ────────────────────────────────────────────────────────────────────
BIRD_NUM  = "2"
N_SYNTH   = 100
FONT_SIZE = 11
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR  = os.path.join(REPO_ROOT, "figures", "Supplementary", "Supp_1")

# Colours (Okabe-Ito / colourblind-friendly)
CB_COLORS = {
    "non-significant": "#BDBBBB",
    "positive":        "#f7af91ff",
    "negative":        "#93b4e9ff",
}
KDE_COLOR      = '#555555'
MEDIAN_COLOR   = 'black'
OBSERVED_COLOR = '#D55E00'

# Panels: (syl1, syl2, analysis_type)  where analysis_type = 'adj' or 'next'
PANELS = [
    ('c', 'b', 'adj'),
    ('c', 'e', 'next'),
    ('b', 'c', 'next'),
]

# ── Load YAML config for bird 2 ───────────────────────────────────────────────
yaml_path = os.path.join(REPO_ROOT, "yamls", f"bird_{BIRD_NUM}.yaml")
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

adjacent_dir_synth = os.path.join(REPO_ROOT, config['adjacent_dir_synth'])
adjacent_dir       = os.path.join(REPO_ROOT, config['adjacent_dir'])
next_dir_synth     = os.path.join(REPO_ROOT, config['next_dir_synth'])
next_dir           = os.path.join(REPO_ROOT, config['next_dir'])

real_adj_path  = os.path.join(adjacent_dir, f"bird_{BIRD_NUM}_adjacent_corr_ALL.csv")
real_next_path = os.path.join(next_dir,     f"bird_{BIRD_NUM}_next_corr_ALL.csv")

df_real_adj  = pd.read_csv(real_adj_path)
df_real_next = pd.read_csv(real_next_path)


# ── Helper: collect synthetic + real correlation for one pair ─────────────────
def get_data(syl1, syl2, analysis_type):
    if analysis_type == 'adj':
        synth_folder    = adjacent_dir_synth
        filename_suffix = "_adjacent_corr_ALL_synth.csv"
        corr_col        = 'rho'
        df_real         = df_real_adj
    else:
        synth_folder    = next_dir_synth
        filename_suffix = "_next_corr_ALL_synth.csv"
        corr_col        = 'correlation'
        df_real         = df_real_next

    # Synthetic correlations
    synth_vals = []
    for i in range(1, N_SYNTH + 1):
        csv_path = os.path.join(synth_folder, f"bird_{BIRD_NUM}",
                                f"bird_{BIRD_NUM}_{i}{filename_suffix}")
        if os.path.exists(csv_path):
            df_s = pd.read_csv(csv_path)
            df_s_pair = df_s[(df_s['syl1'] == syl1) &
                             (df_s['syl2'] == syl2) &
                             (df_s['status'] != 'insufficient_data')]
            if not df_s_pair.empty:
                synth_vals.extend(df_s_pair[corr_col].tolist())

    # Observed correlation
    df_pair = df_real[(df_real['syl1'] == syl1) &
                      (df_real['syl2'] == syl2) &
                      (df_real['status'] != 'insufficient_data')]
    real_rho = df_pair[corr_col].values[0] if not df_pair.empty else None

    return synth_vals, real_rho, corr_col


# ── Compute z-test p-value ────────────────────────────────────────────────────
def compute_p(synth_vals, real_rho):
    synth_mean = np.mean(synth_vals)
    synth_std  = np.std(synth_vals, ddof=1)
    if synth_std == 0:
        return np.nan
    z = (real_rho - synth_mean) / synth_std
    return 1 - norm.cdf(z) if real_rho > 0 else norm.cdf(z)


def format_p(p):
    if np.isnan(p):
        return "p=nan"
    if p >= 0.05:
        return f"p={p:.3f}"
    if p < 0.001:
        return "p<0.001"
    if p < 0.01:
        return "p<0.01"
    if p < 0.05:
        return "p<0.05"
    return f"p={p:.3f}"


# ── Build figure ──────────────────────────────────────────────────────────────
A4_WIDTH   = 8.27  # inches
FIG_HEIGHT = 3.5

fig, axes = plt.subplots(1, 3, figsize=(A4_WIDTH, FIG_HEIGHT))

median_handle   = None
observed_handle = None

for ax, (syl1, syl2, atype) in zip(axes, PANELS):
    synth_vals, real_rho, _ = get_data(syl1, syl2, atype)
    if not synth_vals or real_rho is None:
        ax.set_visible(False)
        continue

    p_value        = compute_p(synth_vals, real_rho)
    synth_median   = np.median(synth_vals)
    analysis_label = "Adjacent" if atype == 'adj' else "Next"

    # Decide histogram colour based on z-test result
    if np.isnan(p_value) or p_value >= 0.05:
        hist_color = CB_COLORS["non-significant"]
    elif real_rho > 0:
        hist_color = CB_COLORS["positive"]
    else:
        hist_color = CB_COLORS["negative"]

    # Histogram + KDE
    sns.histplot(synth_vals, binwidth=0.01, kde=True, color=hist_color, ax=ax,
                 line_kws={'color': KDE_COLOR, 'linewidth': 1.5}, edgecolor=None)

    # Observed line colour
    if hist_color == CB_COLORS["positive"]:
        obs_line_color = "#d55e00"   # Okabe-Ito vermillion
    elif hist_color == CB_COLORS["negative"]:
        obs_line_color = "#0072b2"   # Okabe-Ito blue
    else:
        obs_line_color = "#888888"

    median_line   = ax.axvline(synth_median, color=MEDIAN_COLOR,   linestyle='--', linewidth=2, label='Random resample')
    observed_line = ax.axvline(real_rho,     color=obs_line_color, linestyle='-',  linewidth=2, label='Observed value')

    if median_handle is None:
        median_handle   = median_line
        observed_handle = observed_line

    # P-value annotation
    ax.text(0.97, 0.92, format_p(p_value), transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right', fontsize=FONT_SIZE,
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    # Labels
    ax.set_title(f"{syl1.upper()}→{syl2.upper()} ({analysis_label})", fontsize=FONT_SIZE)
    ax.set_xlabel("Correlation", fontsize=FONT_SIZE)
    ax.set_ylabel("Frequency",   fontsize=FONT_SIZE)
    ax.tick_params(labelsize=FONT_SIZE)

    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Axis limits and ticks
    ax.set_xlim(-0.2, 0.6)
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_major_locator(ticker.FixedLocator([0, 4, 8, 12, 16, 20, 24]))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(2))
    ax.set_ylim(0, 24)

# ── Shared legend at bottom centre ───────────────────────────────────────────
if median_handle and observed_handle:
    legend_obs = mlines.Line2D([], [], color='black', linestyle='-', linewidth=2, label='Observed value')
    fig.legend(
        handles=[median_handle, legend_obs],
        labels=['Random resample', 'Observed value'],
        loc='lower center',
        ncol=2,
        fontsize=FONT_SIZE,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

plt.tight_layout(rect=[0, 0.08, 1, 1])

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(SAVE_DIR, exist_ok=True)
png_path = os.path.join(SAVE_DIR, "supp_1.png")
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(png_path.replace('.png', '.svg'), bbox_inches='tight')
print(f"Saved: {png_path}")

plt.show()
