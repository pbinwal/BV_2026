# BV_2026 — Code Repository

This repository contains the analyses and figure-generation scripts accompanying the manuscript.

***NOTE: Figures might look aesthetically different from those in the manuscript because of changes that were later made in Inkspace

For code or data-related inquiries please contact xxx

---

## Repository Structure

```
data/           — Per-bird song data CSVs (bird_1_df.csv … bird_6_df.csv)
                  data/acoustic_distances/bird_{n}/ contains two distance matrices per bird:
                    bird_{n}_context_sensitive.csv   — non-symmetric; used when analysing acoustic
                                                       distances between correlated phrases that
                                                       were adjacent, to account for subtle
                                                       differences in acoustics (Wohlgemuth, 2010).
                    bird_{n}_context_agnostic.csv    — symmetric; used for next phrase-pair analyses.
yamls/          — Per-bird configuration files (paths, directories)
scripts/        — All analyses and plotting scripts
output/         — Computed outputs (transition matrices, correlations, etc.)
figures/        — Saved figure files
```

---

## Reproducing the Figures

Scripts should be run from the `scripts/` directory. 

---

### Figure 1

**Panel A** — no script

**Panel B** — `fig_1_b.py` — per-syllable repeat-number distributions. Prompts for bird number.

> **Prerequisites: Run `plot_trans_diag.py` for all birds to get transition probabilites which are used by CDEFG

**Panels C, D, E, F** — `fig_1_cdef_plot_rpts_by_context.py` — run for required bird and phrase. 
For Cand E: script used to get only transition probabilities.

**Panel G** — `fig_1_g.py` 

---

### Figure 2

**Panel A** — no script 

**Panel B** — `fig_2_b_plot_adjacent_corr.py` — Bird 1, syllable pair a→u.

> **Prerequisites: Run `z_test_corr.py` for all birds Adjacent and Next with savemode = y to get required csvs before proceeding to panel C

**Panel C** — `fig_2_c.py` — all birds, run once. 

**Panel D** — `fig_2_d.py` — Bird 2, syllable C. Requires `fig_2_synth_control.py` to have been run for all birds and `fig_2_g.py` to have been run once first (to generate `output/Correlations by distance z corrected/all_birds_corr_by_dist.csv`).

**Panel E** — `plot_trans_diag.py` — Bird 2.

**Panel F** — `fig_2_f_corr_diag.py` — requires the CSVs produced by `fig_2_c.py`.

**Panel G** — `fig_2_g.py` — all birds, run once. Requires `fig_2_synth_control.py` to have been run for all birds first.

---

### Figure 3


> **Prerequisites: Run `fig_2_c.py` first to get csv that `fig_3_b_elasticity.py` uses.
**Panels A, B** — `fig_3_a_coloured_song.py`, `fig_3_b_elasticity.py`

**Panels C, E** — `fig_3_ce_example.py` — also produces Panels D and F (see below).

**Panels D, F** — `fig_3_df_all_birds_panels.py` — all birds, run once. Requires `fig_3_ce_example.py` to have been run first to generate the median CSVs in `output/Positions csvs/`.

**Panel G, H** — `fig_3_gh_time.py` — G: Bird 2, syllable C, time-of-day example, H: all birds, median repeat number vs time of day.

*`fig_3_ce_example.py`* collects repeat-by-occurrence and repeat-by-relative-position data across all birds, saves median analysis CSVs and regression results to `output/Positions csvs/`, and generates the example panels (C, E) directly from memory.

---

### Figure 4

> **Prerequisites: Before running any Figure 4 script, run `rf_by_phrase.py` once for each bird (1–6). This trains the Random Forest models and saves the SHAP importance CSVs that the plotting scripts read.

** Panel A** —  no script

**Panel B** — `fig_4_b_example.py` — single bird/syllable SHAP importance bar chart. Prompts for bird number and syllable label.
inset using `fig_2_d.py`

**Panel C** — `fig_4_c_rf_all_birds.py` — all-birds normalised SHAP importance summary (bar + scatter). Run after `rf_by_phrase.py` has been run for all 6 birds.

---
### Supplementary Figures

**Supplementary Figure 1** — 
> **Prerequisites: Run `fig_2_b_plot_adjacent_corr.py` (You can input 'n' if you don't want to save a figure.) and `plot_corr_next.py` and `z_test_corr.py`for all birds before running supp_1_.py. 

`supp_1.py` — z-test distribution panels for Bird 2 (C→B adjacent, C→E next, B→C next). Saved to `figures/Supplementary/Supp_1/supp_1.png`.

> **Prerequisites: Run `fig_2_c.py` first to get csv that `fig_3_b_elasticity.py` uses.
**Supplementary Figure 2 (Panels A, B)** — `fig_3_b_elasticity.py` — same script as Figure 3B; supplementary panels saved alongside the main figure output. 


**Supplementary Figure 2 (Panel C)** — no script
**Supplementary Figure 2 (Panel DE)** — supp_2_de.py

**Supplementary Figure 3** — `fig_4_c_rf_all_birds.py` — same script as Figure 4C; saves the extended ±4 context SHAP summary to `figures/Supplementary/Supp_3/supp_3.png`.

---

## Script Descriptions

**`get_repeat_data.py`** — loads per-bird song data from CSV files.

**`get_compressed_syntax.py`** — compresses song sequences (e.g. `aaabb` → `a3b2`).

**`fig_2_synth_control.py`** — per-bird: computes observed Spearman correlations between syllable-pair repeat counts as a function of phrase distance, then z-tests each against a synthetic null distribution built from 100 permuted datasets.

**`fig_1_cdef_plot_rpts_by_context.py`** — plots repeat number distributions by syllable context (Fig 1 C–F).

**`fig_1_g.py`** — plots |correlation| vs phrase distance scatter (Fig 1G).

**`fig_2_b_plot_adjacent_corr.py`** — adjacent syllable-pair correlation scatterplots (Fig 2B).

**`fig_2_c.py`** — z-test and Benjamini-Hochberg correction pipeline; produces correlation pie charts and histogram (Fig 2C).

**`plot_trans_diag.py`** — greyscale transition heatmap and circular transition diagram (Fig 2E).

**`fig_2_f_corr_diag.py`** — correlation diagram with arrows scaled by |ρ| (Fig 2F).

**`fig_2_g.py`** — pools all birds, applies BH corrections, and plots |correlation| vs phrase distance (Fig 2G).

**`fig_3_a_coloured_song.py`** — plots colour-coded song sequences for a single bird, stacking up to 50 songs and colouring each syllable type consistently (Fig 3A).

**`elasticity.py`** — importable module; `build_repeat_df` and `calculate_repeat_number_elasticity` compute per-syllable repeat-number elasticity via OLS regression of `rpt_num ~ song_len`.

**`fig_3_b_elasticity.py`** — calls `elasticity.py` for all birds in memory; produces an all-birds elasticity histogram and adjacent/next elasticity pair scatters (Fig 3B).

**`fig_3_ce_example.py`** — collects repeat-by-occurrence and repeat-by-relative-position data across all birds; saves median CSVs and linear regression results to `output/Positions csvs/`; generates example stacked histogram panels for Bird 6/H (occurrence, Fig 3C) and Bird 1/U (position quartile, Fig 3E).

**`fig_3_a_coloured_song.py`** — plots colour-coded song sequences for a single bird, stacking up to 50 songs and colouring each syllable type consistently (Fig 3A).

**`elasticity.py`** — importable module; `build_repeat_df` and `calculate_repeat_number_elasticity` compute per-syllable repeat-number elasticity via OLS regression of `rpt_num ~ song_len`.

**`fig_3_b_elasticity.py`** — calls `elasticity.py` for all birds in memory; produces an all-birds elasticity histogram and adjacent/next elasticity pair scatters (Fig 3B).

**`fig_3_df_all_birds_panels.py`** — all-birds median repeat number vs occurrence order and vs position quartile, with overall median and example highlight (Fig 3D, 3F).

**`fig_3_gh_time.py`** — collects repeat-by-time-of-day data across all birds; saves median CSV and regression results to `output/Time of day csvs/`; generates the time-of-day example panel (Fig 3G) and all-birds summary panel (Fig 3H).

**`rf_by_phrase.py`** — per-bird: trains a Random Forest model predicting repeat number for each syllable type using contextual features (±4 neighbours, song length, occurrence order, relative position, time of day); saves model, CV/OOB scores, and SHAP importance CSV to `output/RF results/per syllable/bird_{n}/{syl}/`.

**`fig_4_b_example.py`** — reads the SHAP importance CSV for a single bird/syllable and plots a grouped vertical bar chart coloured by feature class (Fig 4B). Prompts interactively for bird number and syllable.

**`fig_4_c_rf_all_birds.py`** — reads all per-syllable SHAP CSVs across all birds, filters by model quality thresholds (ΔR²≤0.1, n≥1000), normalises by repeat-number SD, and plots the all-birds summary bar+scatter figure (Fig 4C).

**`z_test_corr.py`** — z-tests adjacent and next correlations against synthetic data.

**`plot_corr_next.py`** — next syllable correlation scatterplots.

**`plot_rpt_distr.py`** — repeat number distribution plots.

---

## Dependencies

See `requirements.txt` for the full list. Key packages:

- Python ≥ 3.9
- pandas, numpy, scipy, matplotlib, seaborn, networkx, statsmodels, pyyaml

