import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


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
# DBSCAN
# -------------------------

dbscan = DBSCAN(
    eps=3.5,
    min_samples=5
)

dbscan.fit(X_scaled)


# -------------------------
# Predictions
# -------------------------

df["Cluster"] = dbscan.labels_

# -1 = Noise (Anomaly)


# -------------------------
# Cluster Counts
# -------------------------

print("\nCluster Counts:")

print(
    df["Cluster"].value_counts().sort_index()
)


# -------------------------
# Anomaly Count
# -------------------------

print("\nNumber of Anomalies:")

print(
    (df["Cluster"] == -1).sum()
)


# -------------------------
# Show Anomalies
# -------------------------

anomalies = df[
    df["Cluster"] == -1
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

            "Cluster"
        ]
    ]
)


# -------------------------
# Save Results
# -------------------------

df.to_csv(
    "dbscanB_results.csv",
    index=False
)

print(
    "\nResults saved as dbscanB_results.csv"
)