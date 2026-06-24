import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


# -------------------------
# Load Dataset
# -------------------------

df = pd.read_csv(
    "feature_engineered_dataset.csv"
)

print(df.shape)


# -------------------------
# New Day
# -------------------------

new_day = {
    "Date": "2020-04-01",
    "Sleep_Score": 35,
    "Deep_Sleep": 5,
    "Restlessness": 0.18,
    "Resting_HR": 72,
    "Steps": 2500,
    "Calories": 2200,
    "Moderate_Activity": 0,
    "Vigorous_Activity": 0,
    "Fatigue": 5,
    "Mood": 1,
    "Readiness": 2,
    "Stress": 5,
    "Fluid_Intake": 3
}

new_df = pd.DataFrame([new_day])

combined = pd.concat(
    [df, new_df],
    ignore_index=True
)


# -------------------------
# Date Conversion
# -------------------------

combined["Date"] = pd.to_datetime(
    combined["Date"]
)

combined = combined.sort_values(
    "Date"
)


# -------------------------
# Recalculate Baselines
# -------------------------

combined["HR_7day_avg"] = (
    combined["Resting_HR"]
    .rolling(7)
    .mean()
)

combined["Sleep_7day_avg"] = (
    combined["Sleep_Score"]
    .rolling(7)
    .mean()
)

combined["Steps_7day_avg"] = (
    combined["Steps"]
    .rolling(7)
    .mean()
)


# -------------------------
# Recalculate Deviations
# -------------------------

combined["HR_Deviation"] = (
    combined["Resting_HR"]
    - combined["HR_7day_avg"]
)

combined["Sleep_Deviation"] = (
    combined["Sleep_Score"]
    - combined["Sleep_7day_avg"]
)

combined["Steps_Deviation"] = (
    combined["Steps"]
    - combined["Steps_7day_avg"]
)


# -------------------------
# Recalculate STD
# -------------------------

combined["HR_7day_std"] = (
    combined["Resting_HR"]
    .rolling(7)
    .std()
)

combined["Sleep_7day_std"] = (
    combined["Sleep_Score"]
    .rolling(7)
    .std()
)

combined["Steps_7day_std"] = (
    combined["Steps"]
    .rolling(7)
    .std()
)


# -------------------------
# Recalculate Z Scores
# -------------------------

combined["HR_Zscore"] = (
    combined["HR_Deviation"]
    / combined["HR_7day_std"]
)

combined["Sleep_Zscore"] = (
    combined["Sleep_Deviation"]
    / combined["Sleep_7day_std"]
)

combined["Steps_Zscore"] = (
    combined["Steps_Deviation"]
    / combined["Steps_7day_std"]
)


# -------------------------
# Split History / Today
# -------------------------

history = combined.iloc[:-1].copy()

today = combined.iloc[[-1]].copy()


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


# -------------------------
# Scaling
# -------------------------

scaler = StandardScaler()

X_history = scaler.fit_transform(
    history[features]
)

X_today = scaler.transform(
    today[features]
)


# -------------------------
# Isolation Forest
# -------------------------

iso_model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

iso_model.fit(X_history)


# -------------------------
# Prediction
# -------------------------

prediction = iso_model.predict(
    X_today
)

score = iso_model.decision_function(
    X_today
)


# -------------------------
# Output
# -------------------------

print("\nToday's Data:")

print(
    today[
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
            "Readiness"
        ]
    ]
)

print("\nPrediction:")

if prediction[0] == -1:
    print("ANOMALY DETECTED")
else:
    print("NORMAL DAY")

print(
    "Anomaly Score:",
    score[0]
)