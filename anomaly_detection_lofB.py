import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor


# -------------------------
# Load Dataset
# -------------------------

df = pd.read_csv(
    "feature_engineered_dataset.csv"
)

print("Dataset Shape:")
print(df.shape)


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

print("\nSelected Features:")
print(X.head())


# -------------------------
# Missing Values Check
# -------------------------

print("\nMissing Values:")

print(
    X.isnull().sum()
)


# -------------------------
# Feature Scaling
# -------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nScaled Data Shape:")
print(X_scaled.shape)


# -------------------------
# Local Outlier Factor
# -------------------------

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.05
)

predictions = lof.fit_predict(X_scaled)


# -------------------------
# Predictions
# -------------------------

df["Anomaly"] = predictions

# 1 = Normal
# -1 = Anomaly


# -------------------------
# Anomaly Counts
# -------------------------

print("\nAnomaly Counts:")

print(
    df["Anomaly"].value_counts()
)


# -------------------------
# LOF Score
# -------------------------

df["LOF_Score"] = -lof.negative_outlier_factor_

print("\nTop 10 Most Abnormal Days:")

print(
    df.sort_values(
        by="LOF_Score",
        ascending=False
    )[
        [
            "Date",
            "Anomaly",
            "LOF_Score"
        ]
    ].head(10)
)


# -------------------------
# Show Anomalies
# -------------------------

anomalies = df[
    df["Anomaly"] == -1
]

print("\nDetected Anomalies:")

print(
    anomalies[
        [
            "Date",

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

            "LOF_Score"
        ]
    ]
)


# -------------------------
# Save Results
# -------------------------

df.to_csv(
    "lofB_results.csv",
    index=False
)

print(
    "\nResults saved as lofB_results.csv"
)