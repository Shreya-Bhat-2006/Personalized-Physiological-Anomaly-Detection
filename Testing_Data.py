import pandas as pd
import numpy as np

# -------------------------
# Load Dataset
# -------------------------

df = pd.read_csv("feature_engineered_dataset.csv")

print("="*70)
print("DATASET INFORMATION")
print("="*70)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

# -------------------------
# Feature Set B
# -------------------------

features = [
    "HR_Zscore",
    "Sleep_Zscore",
    "Steps_Zscore",

    "Deep_Sleep",
    "Calories",
    "Moderate_Activity",
    "Vigorous_Activity",
    "Fluid_Intake",

    "Stress",
    "Fatigue",
    "Mood",
    "Readiness",
    "Restlessness"
]

X = df[features]

# -------------------------
# Missing Values
# -------------------------

print("\n" + "="*70)
print("MISSING VALUES")
print("="*70)

print(X.isnull().sum())

# -------------------------
# Descriptive Statistics
# -------------------------

print("\n" + "="*70)
print("DESCRIPTIVE STATISTICS")
print("="*70)

stats = pd.DataFrame({
    "Mean": X.mean(),
    "Median": X.median(),
    "Std": X.std(),
    "Min": X.min(),
    "25%": X.quantile(0.25),
    "75%": X.quantile(0.75),
    "Max": X.max()
})

print(stats)

# -------------------------
# Save Statistics
# -------------------------

stats.to_csv("dataset_statistics.csv")

print("\nStatistics saved as dataset_statistics.csv")

# -------------------------
# Correlation Matrix
# -------------------------

print("\n" + "="*70)
print("CORRELATION MATRIX")
print("="*70)

corr = X.corr()

print(corr.round(2))

corr.to_csv("feature_correlations.csv")

print("\nCorrelation matrix saved as feature_correlations.csv")

# -------------------------
# Feature Ranges
# -------------------------

print("\n" + "="*70)
print("FEATURE RANGES")
print("="*70)

for feature in features:

    print(f"\n{feature}")

    print(f"Mean   : {X[feature].mean():.3f}")
    print(f"Std    : {X[feature].std():.3f}")
    print(f"Min    : {X[feature].min():.3f}")
    print(f"Max    : {X[feature].max():.3f}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)



import pandas as pd
import numpy as np

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv("feature_engineered_dataset.csv")

# --------------------------------------------------
# Feature Set B
# --------------------------------------------------

features = [
    "HR_Zscore",
    "Sleep_Zscore",
    "Steps_Zscore",
    "Deep_Sleep",
    "Calories",
    "Moderate_Activity",
    "Vigorous_Activity",
    "Fluid_Intake",
    "Stress",
    "Fatigue",
    "Mood",
    "Readiness",
    "Restlessness"
]

real_data = df[features].copy()

# --------------------------------------------------
# Feature Statistics
# --------------------------------------------------

feature_std = real_data.std()

feature_min = real_data.min()

feature_max = real_data.max()

# --------------------------------------------------
# Random Seed
# --------------------------------------------------

np.random.seed(42)

# --------------------------------------------------
# Generate Dataset A
# --------------------------------------------------

synthetic_rows = []

NUM_DAYS = 50

for i in range(NUM_DAYS):

    # Pick one real day
    row = real_data.sample(n=1).iloc[0].copy()

    new_row = {}

    for feature in features:

        value = row[feature]

        # Small perturbation (10% of feature std)
        noise = np.random.normal(
            0,
            feature_std[feature] * 0.10
        )

        value = value + noise

        # Keep inside real range
        value = np.clip(
            value,
            feature_min[feature],
            feature_max[feature]
        )

        new_row[feature] = value

    synthetic_rows.append(new_row)

# --------------------------------------------------
# Create DataFrame
# --------------------------------------------------

synthetic_df = pd.DataFrame(synthetic_rows)

# --------------------------------------------------
# Round Discrete Features
# --------------------------------------------------

discrete_features = [
    "Stress",
    "Fatigue",
    "Mood",
    "Readiness",
    "Fluid_Intake"
]

for col in discrete_features:

    synthetic_df[col] = synthetic_df[col].round()

# --------------------------------------------------
# Ensure Integer Type
# --------------------------------------------------

synthetic_df[discrete_features] = synthetic_df[
    discrete_features
].astype(int)

# --------------------------------------------------
# Save
# --------------------------------------------------

synthetic_df.to_csv(
    "synthetic_datasetA.csv",
    index=False
)

print("="*60)
print("Synthetic Dataset A Generated")
print("="*60)

print()

print("Shape:")
print(synthetic_df.shape)

print()

print(synthetic_df.head())

print()

print("Saved as:")
print("synthetic_datasetA.csv")

# --------------------------------------------------
# VALIDATE SYNTHETIC DATASET A
# --------------------------------------------------

print("\n")
print("="*70)
print("VALIDATING SYNTHETIC DATASET A")
print("="*70)

real = real_data.copy()
synthetic = synthetic_df.copy()

comparison = pd.DataFrame(index=features)

comparison["Real Mean"] = real.mean()
comparison["Synthetic Mean"] = synthetic.mean()

comparison["Real Std"] = real.std()
comparison["Synthetic Std"] = synthetic.std()

comparison["Real Min"] = real.min()
comparison["Synthetic Min"] = synthetic.min()

comparison["Real Max"] = real.max()
comparison["Synthetic Max"] = synthetic.max()

comparison["Mean Difference %"] = (
    abs(comparison["Synthetic Mean"] - comparison["Real Mean"])
    / abs(comparison["Real Mean"] + 1e-8)
) * 100

comparison["Std Difference %"] = (
    abs(comparison["Synthetic Std"] - comparison["Real Std"])
    / abs(comparison["Real Std"] + 1e-8)
) * 100

print("\nFeature Comparison:\n")
print(comparison.round(3))

comparison.to_csv("datasetA_validation_statistics.csv")

print("\nValidation statistics saved as:")
print("datasetA_validation_statistics.csv")

# --------------------------------------------------
# CORRELATION COMPARISON
# --------------------------------------------------

print("\n")
print("="*70)
print("CORRELATION COMPARISON")
print("="*70)

real_corr = real.corr()
synthetic_corr = synthetic.corr()

corr_difference = (synthetic_corr - real_corr).abs()

print("\nAverage Correlation Difference :")

avg_corr = corr_difference.values.mean()

print(round(avg_corr,3))

corr_difference.to_csv("datasetA_correlation_difference.csv")

print("\nSaved:")
print("datasetA_correlation_difference.csv")

# --------------------------------------------------
# STRONGEST CORRELATIONS
# --------------------------------------------------

print("\n")
print("="*70)
print("TOP CORRELATIONS (REAL DATA)")
print("="*70)

corr_pairs = real_corr.unstack()

corr_pairs = corr_pairs[
    corr_pairs.index.get_level_values(0)
    !=
    corr_pairs.index.get_level_values(1)
]

corr_pairs = corr_pairs.abs()

corr_pairs = corr_pairs.sort_values(
    ascending=False
)

printed = set()

count = 0

for pair, value in corr_pairs.items():

    a, b = pair

    key = tuple(sorted([a,b]))

    if key in printed:
        continue

    printed.add(key)

    print(f"{a:25} <-> {b:25} : {value:.3f}")

    count += 1

    if count == 10:
        break

# --------------------------------------------------
# FINAL MESSAGE
# --------------------------------------------------

print("\n")
print("="*70)
print("DATASET A VALIDATION COMPLETE")
print("="*70)