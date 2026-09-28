# supp_2_de_duration_lmm.R
#
# Tests whether adjacent and next correlation categories differ in delta
# duration, where delta duration is the absolute difference between the mean
# durations of the two syllables in an ordered pair.
#
# The Python script supp_2_hi_dur_old.py creates the derived input CSV:
#   data/delta_duration/duration_correlations.csv
#
# Model:
#   delta_duration ~ sig_status_aft_all_corr + (1 | bird_id)
#
# The script prints model summaries, displays DHARMa residual diagnostics, and
# prints Tukey-adjusted pairwise comparisons. It does not save any files.

# Load packages used for data manipulation, mixed models, post-hoc tests, and
# simulation-based residual diagnostics.
library(tidyverse)
library(lme4)
library(lmerTest)
library(emmeans)
library(DHARMa)

# Repo root hard-coded to avoid RStudio active-document detection issues.
repo_root <- "C:/Users/priyabinwal/Documents/Priya/Priya PhD Git repos/BV_2026_Binwal_Veit_2026"

# Build the path to the derived duration/correlation table.
duration_csv <- file.path(
  repo_root,
  "data",
  "delta_duration",
  "duration_correlations.csv"
)

# Stop here with an actionable message if the Python script has not generated
# the input tables yet. This prevents a missing-file error from cascading into
# misleading "object not found" errors for the data frames and models below.
if (!file.exists(duration_csv)) {
  stop(
    paste0(
      "Missing duration input table:\n",
      duration_csv,
      "\nRun scripts/supp_2_hi_dur_old.py first, then rerun this R script."
    )
  )
}

# Select adjacent pairs and next pairs without self-context events.
duration_df <- read_csv(duration_csv, show_col_types = FALSE)
adj_df <- duration_df |> filter(context == "adjacent") |> mutate(context = "Adjacent")
next_df <- duration_df |>
  filter(context == "next_without_self") |>
  mutate(context = "Next")

# Use the same category order as the acoustic LMM so model coefficients and
# post-hoc comparisons use non-significant as the reference category.
adj_df$sig_status_aft_all_corr <- factor(
  adj_df$sig_status_aft_all_corr,
  levels = c("non-significant", "positive", "negative")
)
next_df$sig_status_aft_all_corr <- factor(
  next_df$sig_status_aft_all_corr,
  levels = c("non-significant", "positive", "negative")
)

# Treat bird as a grouping variable for the random intercept.
adj_df$bird_id <- factor(adj_df$bird_num)
next_df$bird_id <- factor(next_df$bird_num)

# Confirm how many rows are available for each context before fitting models.
cat("Rows loaded — Adjacent:", nrow(adj_df), " Next:", nrow(next_df), "\n\n")

# Fit one mixed model for adjacent pairs and one for next pairs. The fixed
# effect tests whether delta duration differs among correlation categories;
# the random intercept accounts for baseline differences between birds.
lmm_adj <- lmer(
  delta_duration ~ sig_status_aft_all_corr + (1 | bird_id),
  data = adj_df
)
lmm_next <- lmer(
  delta_duration ~ sig_status_aft_all_corr + (1 | bird_id),
  data = next_df
)

# Print the fitted model summaries for both contexts.
cat("=== Adjacent delta-duration LMM ===\n")
print(summary(lmm_adj))
cat("\n=== Next delta-duration LMM ===\n")
print(summary(lmm_next))

# Simulate residuals and display DHARMa diagnostics for the adjacent model.
cat("\nDHARMa diagnostics — Adjacent delta duration:\n")
sim_adj <- simulateResiduals(lmm_adj)
plot(sim_adj, main = "DHARMa: Adjacent delta duration")

# Simulate residuals and display DHARMa diagnostics for the next model.
cat("\nDHARMa diagnostics — Next delta duration:\n")
sim_next <- simulateResiduals(lmm_next)
plot(sim_next, main = "DHARMa: Next delta duration")

# Estimate category means from the adjacent model and print all pairwise
# differences with Tukey multiplicity adjustment.
cat("\n=== Post-hoc contrasts — Adjacent delta duration ===\n")
emm_adj <- emmeans(lmm_adj, ~ sig_status_aft_all_corr)
print(pairs(emm_adj, adjust = "tukey"))

# Estimate category means from the next model and print all pairwise
# differences with Tukey multiplicity adjustment.
cat("\n=== Post-hoc contrasts — Next delta duration ===\n")
emm_next <- emmeans(lmm_next, ~ sig_status_aft_all_corr)
print(pairs(emm_next, adjust = "tukey"))




#################### duration visualise (plots)
# --- Violin plots of delta duration by correlation category ---

duration_plot_df <- bind_rows(adj_df, next_df) |>
  mutate(
    context = factor(context, levels = c("Adjacent", "Next")),
    sig_status_aft_all_corr = factor(
      sig_status_aft_all_corr,
      levels = c("non-significant", "positive", "negative")
    )
  )

duration_violin_plot <- ggplot(
  duration_plot_df,
  aes(
    x = sig_status_aft_all_corr,
    y = delta_duration,
    fill = sig_status_aft_all_corr
  )
) +
  geom_violin(
    trim = FALSE,
    alpha = 0.65,
    color = "black"
  ) +
  geom_boxplot(
    width = 0.16,
    outlier.shape = NA,
    fill = "white",
    color = "black"
  ) +
  geom_jitter(
    width = 0.08,
    alpha = 0.35,
    size = 1,
    color = "black"
  ) +
  facet_wrap(~ context, ncol = 1) +
  scale_fill_manual(
    values = c(
      "non-significant" = "#BDBDBD",
      "positive" = "#E66557",
      "negative" = "#6D90F0"
    )
  ) +
  labs(
    x = "Correlation category",
    y = "Delta duration"
  ) +
  theme_classic() +
  theme(
    legend.position = "none",
    strip.background = element_blank(),
    strip.text = element_text(size = 12),
    axis.text = element_text(size = 11),
    axis.title = element_text(size = 12)
  )

print(duration_violin_plot)
