# supp_2_de_acoustic_lmm.R
#
# Tests whether syllable pairs in different correlation categories
# (non-significant, positive, negative) differ in z-scored acoustic distance,
# using Linear Mixed Models with bird as a random effect.
#
# Models (one per context):
#   z_scored_acoustic_dist ~ sig_status_aft_all_corr + (1 | bird_id)
#
# Post-hoc pairwise Tukey contrasts via emmeans.
# DHARMa residual diagnostics plotted for both models.

library(tidyverse)
library(lme4)
library(lmerTest)
library(emmeans)
library(DHARMa)

# ── Paths (relative to repo root) ────────────────────────────────────────────
repo_root <- tryCatch(
  dirname(rstudioapi::getActiveDocumentContext()$path) |> dirname(),
  error = function(e) getwd()
)

adj_csv  <- file.path(repo_root, "output", "Acoustic distances z corrected",
                      "combined_acoustic_corr_adj_df.csv")
next_csv <- file.path(repo_root, "output", "Acoustic distances z corrected",
                      "combined_acoustic_corr_next_df.csv")

# ── Load data ─────────────────────────────────────────────────────────────────
adj_df  <- read_csv(adj_csv,  show_col_types = FALSE) |> mutate(context = "Adjacent")
next_df <- read_csv(next_csv, show_col_types = FALSE) |> mutate(context = "Next")

adj_df$sig_status_aft_all_corr  <- factor(adj_df$sig_status_aft_all_corr,
                                           levels = c("non-significant", "positive", "negative"))
next_df$sig_status_aft_all_corr <- factor(next_df$sig_status_aft_all_corr,
                                           levels = c("non-significant", "positive", "negative"))

cat("Rows loaded — Adjacent:", nrow(adj_df), "  Next:", nrow(next_df), "\n\n")

# ── LMMs ──────────────────────────────────────────────────────────────────────
lmm_adj  <- lmer(z_scored_acoustic_dist ~ sig_status_aft_all_corr + (1 | bird_id), data = adj_df)
lmm_next <- lmer(z_scored_acoustic_dist ~ sig_status_aft_all_corr + (1 | bird_id), data = next_df)

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
