# Most recent script: 17.04.2026 after z tests and 2 independent BH corrections
library(lme4)
library(lmerTest)
library(ggplot2)
library(dplyr)

# --- CONFIGURATION ---
model_type <- 1  # Default to linear model

# Read the combined CSV
csv_path <- "C:/Users/priyabinwal/Documents/Priya/Priya PhD Git repos/BV_2026/output/Correlations by distance z corrected/all_birds_corr_by_dist.csv"
df <- read.csv(csv_path)

print("Data loaded. Summary:")
print(head(df))
print(paste("Total rows:", nrow(df)))
print(paste("Unique Significance values:", paste(unique(df$Significance), collapse=", ")))

# Filter for significant correlations only
df_sig <- df[df$Significance == "significant", ]

print(paste("\nFiltered to significant rows:", nrow(df_sig)))
print(head(df_sig))

# Add absolute correlation column to both dataframes (must be before modeling/plotting)
df$Abs_Correlation <- abs(df$Correlation)
df_sig$Abs_Correlation <- abs(df_sig$Correlation)

print("\n" %+% "="*80)
print("LINEAR MIXED MODEL RESULTS")
print("="*80)
print("Formula: Abs_Correlation ~ Distance + (1 | Bird_ID)")
print("Data: Significant correlations only")
print("="*80)

print(summary(lmm_model))

##################
bird_pair_counts <- aggregate(Syllable_Pair ~ Bird_ID, data = df_sig, FUN = function(x) length(unique(x)))
colnames(bird_pair_counts) <- c("Bird_ID", "Unique_Syl_Pairs")

cat("\n==============================================\n")
cat("UNIQUE SYLLABLE PAIRS PER BIRD (significant only):\n")
cat("==============================================\n")
for (i in 1:nrow(bird_pair_counts)) {
  bird <- bird_pair_counts$Bird_ID[i]
  bird_data <- df_sig[df_sig$Bird_ID == bird, ]
  pairs <- unique(bird_data$Syllable_Pair)
  
  cat(sprintf("\n  %s : %d pairs\n", bird, bird_pair_counts$Unique_Syl_Pairs[i]))
  cat("    All pairs:", paste(pairs, collapse=", "), "\n")
  
  # Find pairs that occur at more than one distance
  pair_dist_counts <- aggregate(Distance ~ Syllable_Pair, data = bird_data, FUN = function(x) length(unique(x)))
  multi_dist_pairs <- pair_dist_counts[pair_dist_counts$Distance > 1, ]
  
  if (nrow(multi_dist_pairs) > 0) {
    cat(sprintf("    Pairs at >1 distance (%d):\n", nrow(multi_dist_pairs)))
    for (j in 1:nrow(multi_dist_pairs)) {
      cat(sprintf("      %-30s : %d distances\n", 
                  multi_dist_pairs$Syllable_Pair[j], 
                  multi_dist_pairs$Distance[j]))
    }
  } else {
    cat("    No pairs occur at more than one distance\n")
  }
}
cat(sprintf("\n  Total birds: %d\n", nrow(bird_pair_counts)))
cat("==============================================\n\n")
#######################################
# Create scatter plot
plot <- ggplot() +
  # Significant points (green circles)
  geom_jitter(data = df_sig,
              aes(x = Distance, y = Abs_Correlation),
              color = "#009E73", shape = 16, width = 0.1, height = 0, alpha = 0.9, size = 5) +
  # Non-significant points (grey squares)
  geom_jitter(data = df[df$Significance == "non-significant", ],
              aes(x = Distance, y = Abs_Correlation),
              color = "#706f6f", shape = 15, width = 0.1, height = 0, alpha = 0.9, size = 5) +
  labs(title = "",
       x = "Distance (number of intervening phrases)",
       y = "|Correlation|") +
  theme_minimal(base_size = 20) +
  theme(
    legend.position = "none",
    panel.grid = element_blank(),
    axis.line = element_line(color = "black", size = 1.5),
    axis.ticks = element_line(color = "black", size = 1.5),
    axis.ticks.length = unit(0.3, "cm"),
    axis.title = element_text(size = 26, color = 'black'),
    axis.text = element_text(size = 26, color = 'black'),
    plot.title = element_text(size = 26)
  )

print(plot)

#######################################
# DHARMa diagnostic plots for model assumptions
#######################################
library(DHARMa)

# Simulate residuals
simulated_residuals <- simulateResiduals(fittedModel = lmm_model)

# Create diagnostic plot
print("DHARMa ASSUMPTION CHECK PLOTS")
plot(simulated_residuals)

#######################################
# Summary of significant vs non-significant
#######################################
print("SUMMARY: SIGNIFICANT VS NON-SIGNIFICANT PAIRS")

n_sig <- nrow(df_sig)
n_nonsig <- nrow(df[df$Significance == "non-significant", ])
n_other <- nrow(df[df$Significance == "other", ])

print(paste("Significant pairs (used in model):", n_sig))
print(paste("Non-significant pairs (plotted as grey):", n_nonsig))
print(paste("Other pairs (not fully significant):", n_other))
print(paste("TOTAL rows analyzed:", nrow(df)))

# Count unique pairs analyzed and their distance distribution
df_all_original <- read.csv(csv_path)  # Read unfiltered to get all pairs for counting

# Count unique syllable pairs
unique_pairs <- n_distinct(df_all_original$Syllable_Pair)

# Count how many unique pairs appear at single vs multiple distances
pair_distance_counts <- df_all_original %>%
  group_by(Syllable_Pair) %>%
  summarise(n_distances = n_distinct(Distance), .groups = 'drop')

single_distance_pairs <- nrow(pair_distance_counts[pair_distance_counts$n_distances == 1, ])
multiple_distance_pairs <- nrow(pair_distance_counts[pair_distance_counts$n_distances > 1, ])


print(paste("\nTotal unique syllable pairs analyzed:", unique_pairs))
print(paste("  - Appeared at single distance:", single_distance_pairs))
print(paste("  - Appeared at multiple distances:", multiple_distance_pairs))

## ...existing code...
