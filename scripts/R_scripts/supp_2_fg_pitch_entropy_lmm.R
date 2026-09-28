# supp_2_f_pitch_entropy_lmm.R
#
# Tests whether syllable pairs in different correlation categories
# (non-significant, positive, negative) differ in z-scored acoustic distance,
# using Linear Mixed Models with bird as a random effect.
#
# Models (one per context):
#   z_scored_acoustic_dist ~ sig_status_aft_all_corr + (1 | bird_num)
#
# Post-hoc pairwise Tukey contrasts via emmeans.
# DHARMa residual diagnostics plotted for both models.

library(tidyverse)
library(lme4)
library(lmerTest)
library(emmeans)
library(DHARMa)

# ── Paths (repo root hard-coded to avoid RStudio active-document detection issues) ──
repo_root <- "C:/Users/priyabinwal/Documents/Priya/Priya PhD Git repos/BV_2026_Binwal_Veit_2026"

pitch_entropy_csv <- file.path(
  repo_root,
  "data",
  "pitch_entropy_distance",
  "pitch_entropy_correlations.csv"
)

if (!file.exists(pitch_entropy_csv)) {
  stop(
    "Missing input CSV file:\n",
    pitch_entropy_csv,
    "\nRun scripts/supp_2_de_pitch_entropy.py first."
  )
}

# ── Load data ─────────────────────────────────────────────────────────────────
pitch_entropy_df <- read_csv(pitch_entropy_csv, show_col_types = FALSE)
adj_df <- pitch_entropy_df |>
  filter(context == "adjacent") |>
  mutate(context = "Adjacent")
next_df <- pitch_entropy_df |>
  filter(context == "next") |>
  mutate(context = "Next")

adj_df$bird_num  <- factor(adj_df$bird_num)
next_df$bird_num <- factor(next_df$bird_num)

adj_df$sig_status_aft_all_corr  <- factor(adj_df$sig_status_aft_all_corr,
                                           levels = c("non-significant", "positive", "negative"))
next_df$sig_status_aft_all_corr <- factor(next_df$sig_status_aft_all_corr,
                                           levels = c("non-significant", "positive", "negative"))

cat("Rows loaded — Adjacent:", nrow(adj_df), "  Next:", nrow(next_df), "\n\n")

# ── LMMs ──────────────────────────────────────────────────────────────────────
lmm_adj  <- lmer(z_scored_acoustic_dist ~ sig_status_aft_all_corr + (1 | bird_num), data = adj_df)
lmm_next <- lmer(z_scored_acoustic_dist ~ sig_status_aft_all_corr + (1 | bird_num), data = next_df)

cat("=== Adjacent LMM ===\n");  print(summary(lmm_adj))
cat("\n=== Next LMM ===\n");     print(summary(lmm_next))

# ── DHARMa diagnostics ────────────────────────────────────────────────────────
cat("\nDHARMa diagnostics — Adjacent:\n")
sim_adj  <- simulateResiduals(lmm_adj)
plot(sim_adj,  main = "DHARMa: Adjacent context")

cat("\nDHARMa diagnostics — Next:\n")
sim_next <- simulateResiduals(lmm_next)
plot(sim_next, main = "DHARMa: Next context")

# ── Post-hoc pairwise contrasts (Tukey) ──────────────────────────────────────
cat("\n=== Post-hoc contrasts — Adjacent ===\n")
emm_adj  <- emmeans(lmm_adj,  ~ sig_status_aft_all_corr)
print(pairs(emm_adj,  adjust = "tukey"))

cat("\n=== Post-hoc contrasts — Next ===\n")
emm_next <- emmeans(lmm_next, ~ sig_status_aft_all_corr)
print(pairs(emm_next, adjust = "tukey"))
