# rf_by_phrase.py
# This script performs Random Forest analysis per phrase type.
#
# Context window: n=4, meaning each phrase block uses the 4 syllables before (-1 to -4)
# and the 4 syllables after (+1 to +4) as context features, along with their repeat numbers.
#
# For each syllable, the script:
#   - Extracts features from compressed song data
#   - Builds a Random Forest regression model
#   - Computes and saves SHAP and permutation importances
#   - Aggregates SHAP importances by feature group (e.g., previous, next, target syllable)
#   - Saves all outputs in a per-syllable folder


# === Imports ===

# === Import required libraries ===
import sys
import pandas as pd
import numpy as np
import os
import re
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score  # Cross-validation
from sklearn.ensemble import RandomForestRegressor  # Random Forest model
from sklearn.metrics import mean_squared_error, r2_score  # Model evaluation metrics
from sklearn.preprocessing import OneHotEncoder  # Encoding categorical variables
from sklearn.inspection import permutation_importance  # Permutation feature importance (not used in this script)
import shap  # SHAP for model interpretability

# Import custom function to load and process compressed song data
from get_compressed_syntax import load_and_process_data
from get_repeat_data import REPO_ROOT

# === User input for bird number ===

# === Prompt user for bird number and validate ===
bird_num = input("Enter bird number: ").strip()

# === Validate bird_num and get unique syllables for this bird ===

# Dictionary of valid bird numbers and their unique syllables
bird_syllables = {
    "1": ['a', 'u', 'g', 'h', 'e', 'b', 'q'],
    "2": ['b', 'c', 'e'],
    "3": ['b', 'c', 'd', 'e'],
    "4": ['b', 'e', 'k'],
    "5": ['b', 'e', 'f', 'g', 'h', 'm', 'i'],
    "6": ['a', 'b', 'e', 'f', 'h', 'm', 'i'],
}
# Check if bird_num is valid
if bird_num not in bird_syllables:
    print("INVALID Bird Number!!!")
    sys.exit()
unique_syllables = bird_syllables[bird_num]  # List of valid syllables for this bird

# === Load processed data for the selected bird ===

# === Load processed data for the selected bird ===
print("Loading data...")
df = load_and_process_data(bird_num)  # DataFrame with song and metadata


# === Extract time information from wav file names (bird-specific patterns) ===
# Each bird has a different file naming convention, so we use regex to extract date and time
if bird_num == "2":
    df['date_str'] = df['wav_file'].str.extract(r'_(\d{6})_')[0]
    df['time_str'] = df['wav_file'].str.extract(r'_(\d{6})\.\d+\.wav$')[0]
elif bird_num == "6":
    df['date_str'] = df['wav_file'].str.extract(r'_(\d{6})_')[0]
    df['time_str'] = df['wav_file'].str.extract(r'_(\d{8,})_part\d+\.wav$')[0]
elif bird_num in ["1", "3", "4", "5"]:
    df['date_str'] = df['wav_file'].str.extract(r'_(\d{6})_')[0]
    df['time_str'] = df['wav_file'].str.extract(r'_(\d{6})\.-?\d+\.cbin')[0]

# Drop rows with missing time information (i.e., where regex failed)
df = df.dropna(subset=['time_str'])
# Extract hour as integer (first 2 digits of HHMMSS format)
df['hour'] = df['time_str'].str[:2].astype(int)


# === Feature engineering settings (fixed for BV) ===
# Target syllable is always dropped: it is constant within per-syllable models so adds no information
drop_target_syl = True
# Always use one-hot encoding for categorical syllable variables
use_onehot = True
# Always include recording hour as a temporal feature
include_temporal = True
print("Using one-hot encoding. Target syllable dropped. Recording hour included.")


# === Syllables to exclude as targets for each bird ===
# Some syllables are not valid targets for modeling (e.g., rare or problematic syllables)
exclude_target_syllables = {
    "1": ['q', 'l'],
    "2": ['d', 'f', 'g', 'a'],
    "3": ['i', 'f'],
    "4": ['i'],
    "5": ['i', 'g', 'm', 'h'],
    "6": ['f', 'a', 'i'],
}


# === Helper: Check if compressed song string is valid ===
# Only keep rows where the compressed song string contains at least one letter+number pair
def is_valid_compressed_song(s):
    return bool(re.search(r"[a-zA-Z]\d+", str(s)))
valid_rows = df["compressed_song"].apply(is_valid_compressed_song)
df = df[valid_rows].copy()


# === Helper: Calculate total number of syllables in expanded song from compressed format ===
# Each compressed song is a string like 'a3b2c1', meaning 3 a's, 2 b's, 1 c, etc.
# This function returns the total number of syllables in the song.
def calculate_song_length(compressed_song):
    matches = re.findall(r"([a-zA-Z])(\d+)", compressed_song)
    return sum(int(repeat_count) for _, repeat_count in matches)


# === Helper: Extract features for each block in a compressed song string ===
# For each block (syllable) in the song, extract:
#   - The target syllable and its repeat number
#   - The occurrence number of this syllable in the song
#   - The relative position of the block in the song
#   - The previous n syllables and their repeat numbers
#   - The next syllable and its repeat number
def format_data(s, n):
    matches = re.findall(r"([a-zA-Z])(\d+)", s)
    letters, numbers = zip(*matches)
    numbers = list(map(int, numbers))
    total_song_length = sum(numbers)
    syllable_occurrence_count = {}
    target_syl = []
    rpt_num_target = []
    target_occurrence_num = []
    target_relative_pos = []
    prev_syls = []
    prev_rpt_nums = []
    next_syls = []  # List of lists for next n syllables
    next_rpt_nums = []  # List of lists for next n repeat numbers
    for i in range(n, len(letters) - n):
        target_syllable = letters[i]
        syllables_before = sum(numbers[:i])
        block_start_pos = syllables_before + 1
        relative_position = block_start_pos / total_song_length
        if target_syllable not in syllable_occurrence_count:
            syllable_occurrence_count[target_syllable] = 0
        syllable_occurrence_count[target_syllable] += 1
        target_syl.append(target_syllable)
        rpt_num_target.append(numbers[i])
        target_occurrence_num.append(syllable_occurrence_count[target_syllable])
        target_relative_pos.append(relative_position)
        previous_syllables = [letters[i - j - 1] for j in range(n)]
        previous_repeat_nums = [numbers[i - j - 1] for j in range(n)]
        prev_syls.append(previous_syllables)
        prev_rpt_nums.append(previous_repeat_nums)
        # Next n syllables and their repeat numbers
        next_syllables = [letters[i + j + 1] for j in range(n)]
        next_repeat_nums = [numbers[i + j + 1] for j in range(n)]
        next_syls.append(next_syllables)
        next_rpt_nums.append(next_repeat_nums)
    return target_syl, rpt_num_target, target_occurrence_num, target_relative_pos, prev_syls, prev_rpt_nums, next_syls, next_rpt_nums

song_ids = []
recording_hours = []
song_lengths = []

# === Feature extraction for all songs: build a row for each block in each song ===
# For each song, extract features for every block (syllable) using format_data().
# Each block becomes a row in the modeling DataFrame.
n = 4  # Number of previous syllables to include as context
all_target_syl = []  # All target syllables (across all songs)
all_rpt_num_target = []  # All repeat numbers for targets
all_target_occurrence_num = []  # All occurrence numbers
all_target_relative_pos = []  # All relative positions
all_prev_syls = []  # All previous n syllables
all_prev_rpt_nums = []  # All previous n repeat numbers
all_next_syls = []  # All next syllables
all_next_rpt_nums = []  # All next repeat numbers
song_ids = []  # Song index for each block
recording_hours = []  # Recording hour for each block
song_lengths = []  # Song length for each block
for song_id, (idx, row) in enumerate(df.iterrows()):
    song = row["compressed_song"]
    hour = row["hour"]
    song_length = calculate_song_length(song)
    # Extract features for all blocks in this song
    target_syl, rpt_num_target, target_occurrence_num, target_relative_pos, prev_syls, prev_rpt_nums, next_syls, next_rpt_nums = format_data(song, n)
    all_target_syl.extend(target_syl)
    all_rpt_num_target.extend(rpt_num_target)
    all_target_occurrence_num.extend(target_occurrence_num)
    all_target_relative_pos.extend(target_relative_pos)
    all_prev_syls.extend(prev_syls)
    all_prev_rpt_nums.extend(prev_rpt_nums)
    all_next_syls.extend(next_syls)
    all_next_rpt_nums.extend(next_rpt_nums)
    song_ids.extend([song_id] * len(target_syl))
    recording_hours.extend([hour] * len(target_syl))
    song_lengths.extend([song_length] * len(target_syl))



# === Build the main DataFrame for modeling ===
tree_data = pd.DataFrame({
    "song_id": song_ids,
    "recording_hour": recording_hours,
    "song_length": song_lengths,
    "target_syl": all_target_syl,
    "target_occurrence_num": all_target_occurrence_num,
    "target_relative_pos": all_target_relative_pos,
    "rpt_num_target": all_rpt_num_target
})
# Add previous syllables and repeat numbers as separate columns
for i in range(n):
    tree_data[f"prev_syl_{i+1}"] = [syls[i] for syls in all_prev_syls]
    tree_data[f"rpt_num_prev_{i+1}"] = [nums[i] for nums in all_prev_rpt_nums]
# Add next syllables and repeat numbers as separate columns
for i in range(n):
    tree_data[f"next_syl_{i+1}"] = [syls[i] for syls in all_next_syls]
    tree_data[f"next_rpt_num_{i+1}"] = [nums[i] for nums in all_next_rpt_nums]


# Exclude target syllables that are not valid for this bird
if bird_num in exclude_target_syllables and exclude_target_syllables[bird_num]:
    tree_data = tree_data[~tree_data['target_syl'].isin(exclude_target_syllables[bird_num])]


# === Per-syllable analysis: build, evaluate, and interpret a model for each syllable ===
results = []  # Store results for each syllable
shap_results = []  # (Unused, but could store SHAP results)
min_rows = 20  # Minimum data points required to model a syllable
for syl in sorted(tree_data['target_syl'].unique()):
    # Select only rows for the current syllable
    syl_mask = tree_data['target_syl'] == syl
    syl_data = tree_data[syl_mask].copy()
    # Skip syllables with too little data
    if len(syl_data) < min_rows:
        print(f"Skipping syllable {syl} (not enough data: {len(syl_data)})")
        continue
    # Target variable: repeat number for this syllable
    y = syl_data['rpt_num_target']
    # Grouping variable for cross-validation (song ID)
    groups = syl_data['song_id']
    # Build feature matrix X: drop song_id, target, and target_syl (constant within per-syllable model)
    X = syl_data.drop(columns=['song_id', 'rpt_num_target', 'target_syl'])

    # Print the first few rows of the feature matrix and target for inspection (after X and y are defined)
    print(f"\n[INFO] First 5 rows of features (X) for syllable '{syl}':")
    print(X.head())
    print(f"[INFO] First 5 rows of target (y) for syllable '{syl}':")
    print(y.head())
    # Identify which columns are syllable identity columns (to encode)
    syllable_cols = [f"prev_syl_{i+1}" for i in range(n)] + [f"next_syl_{i+1}" for i in range(n)]
    # One-hot encode each syllable column, drop first category to avoid collinearity
    encoded_dfs = [X[[col for col in X.columns if col not in syllable_cols]]]
    for col in syllable_cols:
        onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
        encoded_col = onehot_encoder.fit_transform(X[[col]])
        col_names = [f"{col}_{category}" for category in onehot_encoder.categories_[0][1:]]
        encoded_df = pd.DataFrame(encoded_col, columns=col_names, index=X.index)
        encoded_dfs.append(encoded_df)
    X = pd.concat(encoded_dfs, axis=1)
    # Cross-validation: grouped by song, 5 folds
    gkf_cv = GroupKFold(n_splits=5)
    cv_r2_scores = []
    cv_mse_scores = []
    cv_rmse_scores = []

    # --- Grid Search for best RF parameters per syllable ---
    from sklearn.model_selection import GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [4, 8, 12, None],
        'max_features': ['sqrt', 'log2', None]
    }
    rf_for_grid = RandomForestRegressor(random_state=42, oob_score=True)
    grid = GridSearchCV(rf_for_grid, param_grid, cv=gkf_cv, scoring='r2', n_jobs=-1)
    grid.fit(X, y, groups=groups)
    best_params = grid.best_params_
    print(f"Best RF params for syllable {syl}: {best_params}")

    # Use best params for final CV and model
    for train_idx, test_idx in gkf_cv.split(X, y, groups=groups):
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]
        rf_cv = RandomForestRegressor(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            max_features=best_params['max_features'],
            random_state=42, oob_score=True
        )
        rf_cv.fit(X_train_cv, y_train_cv)
        y_pred_cv = rf_cv.predict(X_test_cv)
        r2_cv = r2_score(y_test_cv, y_pred_cv)
        mse_cv = mean_squared_error(y_test_cv, y_pred_cv)
        rmse_cv = np.sqrt(mse_cv)
        cv_r2_scores.append(r2_cv)
        cv_mse_scores.append(mse_cv)
        cv_rmse_scores.append(rmse_cv)
    # Train/test split: group shuffle (20% test)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    # Fit Random Forest model with best params
    rf_model = RandomForestRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        max_features=best_params['max_features'],
        random_state=42, oob_score=True
    )
    rf_model.fit(X_train, y_train)
    # Predict and evaluate
    y_pred_rf = rf_model.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    oob_score = rf_model.oob_score_
    rmse_rf = np.sqrt(mse_rf)
    # Store results for this syllable
    results.append({
        "Bird_ID": bird_num,
        "Syllable": syl,
        "N_Syllables_Back": n,
        "Encoding_Type": "one-hot",
        "Drop_Target_Syl": True,
        "Include_Temporal": include_temporal,
        "Sample_Size": len(X),
        "Target_RptNum_Std": np.std(y),
        "MSE": mse_rf,
        "RMSE": rmse_rf,
        "R2": r2_rf,
        "OOB_Score": oob_score,
        "CV_R2_Mean": np.mean(cv_r2_scores),
        "CV_R2_Std": np.std(cv_r2_scores),
        "CV_R2_Scores": str(cv_r2_scores),
        "CV_MSE_Mean": np.mean(cv_mse_scores),
        "CV_MSE_Std": np.std(cv_mse_scores),
        "CV_MSE_Scores": str(cv_mse_scores),
        "CV_RMSE_Mean": np.mean(cv_rmse_scores),
        "CV_RMSE_Std": np.std(cv_rmse_scores),
        "CV_RMSE_Scores": str(cv_rmse_scores),
        "Best_n_estimators": best_params['n_estimators'],
        "Best_max_depth": best_params['max_depth'],
        "Best_max_features": best_params['max_features']
    })

    # --- SHAP analysis and grouped SHAP aggregation (model interpretability) ---
    # Helper: group one-hot features by their base (e.g., prev_syl_1_a, prev_syl_1_b -> prev_syl_1)
    def group_individual_features(features_list):
        import re
        from collections import defaultdict
        grouped = defaultdict(list)
        syllable_pattern = r'^(target_syl|prev_syl_\d+|next_syl_\d+)_[a-zA-Z]$'
        for feature in features_list:
            if re.match(syllable_pattern, feature):
                base_name = '_'.join(feature.split('_')[:-1])
                grouped[base_name].append(feature)
            else:
                grouped[feature].append(feature)
        return grouped

    # Fit Random Forest on all data for SHAP
    rf_model_shap = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, oob_score=True)
    rf_model_shap.fit(X, y)
    # Compute SHAP values (feature importances)
    explainer = shap.Explainer(rf_model_shap, X)
    shap_values = explainer(X)

    # Set output directory for this syllable (all outputs go here)
    base_output = os.path.join(REPO_ROOT, 'output', 'RF results', 'per_phrase_rf_models', f'bird_{bird_num}', syl)
    outdir = os.path.abspath(base_output)
    print(f"[DEBUG] Creating output directory: {outdir}")
    os.makedirs(outdir, exist_ok=True)

    # Save SHAP values as CSV
    shap_df = pd.DataFrame({
        "Feature": X.columns,
        "SHAP_Importance": np.mean(np.abs(shap_values.values), axis=0)
    })
    shap_df = shap_df.sort_values("SHAP_Importance", ascending=False).reset_index(drop=True)

    # Save SHAP importances as CSV for this syllable
    shap_csv_path = os.path.join(outdir, f"bird_{bird_num}_{syl}_shap_importance.csv")
    print(f"[DEBUG] Saving SHAP CSV to: {os.path.abspath(shap_csv_path)}")
    shap_df.to_csv(shap_csv_path, index=False)

    # Save model results (metrics and settings) as Excel for this syllable
    rf_results_path = os.path.join(outdir, f"bird_{bird_num}_{syl}_rf_results.xlsx")
    print(f"[DEBUG] Saving RF results to: {os.path.abspath(rf_results_path)}")
    pd.DataFrame([results[-1]]).to_excel(rf_results_path, index=False)

    # Save a sample of the feature matrix and targets (for inspection/debugging)
    sample_size = min(100, len(X))
    X_sample = X.head(sample_size)
    y_sample = y.head(sample_size)
    sample_data = X_sample.copy()
    sample_data['target_rpt_num'] = y_sample.values
    sample_data_path = os.path.join(outdir, f"bird_{bird_num}_{syl}_sample_data.csv")
    print(f"[DEBUG] Saving sample data to: {os.path.abspath(sample_data_path)}")
    sample_data.to_csv(sample_data_path, index=False)

    # Print summary for this syllable
    print(f"Finished syllable {syl}: n={len(X)}, R2={r2_rf:.3f}, OOB={oob_score:.3f}")


# === All syllable models complete: save summary results ===
print("\nAll syllable models complete.")
# Save summary of all syllable results as Excel file
summary_base = os.path.join(REPO_ROOT, 'output', 'RF results', 'per_phrase_rf_models', f'bird_{bird_num}')
summary_dir = os.path.abspath(summary_base)
print(f"[DEBUG] Creating summary directory: {summary_dir}")
os.makedirs(summary_dir, exist_ok=True)
summary_path = os.path.join(summary_dir, f"bird_{bird_num}_per_syllable_rf_summary.xlsx")
print(f"[DEBUG] Saving summary results to: {summary_path}")
pd.DataFrame(results).to_excel(summary_path, index=False)
print(f"Summary results saved to {summary_path}")

# === END OF WORKFLOW ===
# This script extracts features from compressed song data, builds and interprets a Random Forest model for each syllable, and saves all results and plots in organized folders for further analysis.
