import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


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
# Isolation Forest
# -------------------------

iso_model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

iso_model.fit(X_scaled)


# -------------------------
# Predictions
# -------------------------

df["Anomaly"] = iso_model.predict(
    X_scaled
)

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
# Anomaly Score
# -------------------------

df["Anomaly_Score"] = (
    iso_model.decision_function(X_scaled)
)

print("\nTop 10 Most Abnormal Days:")

print(
    df.sort_values(
        by="Anomaly_Score"
    )
    [
        [
            "Date",
            "Anomaly",
            "Anomaly_Score"
        ]
    ]
    .head(10)
)


# -------------------------
# Show Anomalous Days
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

            "Anomaly_Score"
        ]
    ]
)


# -------------------------
# Save Results
# -------------------------

df.to_csv(
    "isoB_results.csv",
    index=False
)

print(
    "\nResults saved as isoB_results.csv"
)