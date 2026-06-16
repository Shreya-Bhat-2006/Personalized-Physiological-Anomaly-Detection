import pandas as pd

master_df = pd.read_csv(
    "master_dataset_v1.csv"
)



master_df["Date"] = pd.to_datetime(
    master_df["Date"]
)



# Resting Heart Rate

master_df["HR_7day_avg"] = (
    master_df["Resting_HR"]
    .rolling(7)
    .mean()
)


# Sleep Score

master_df["Sleep_7day_avg"] = (
    master_df["Sleep_Score"]
    .rolling(7)
    .mean()
)


# Steps

master_df["Steps_7day_avg"] = (
    master_df["Steps"]
    .rolling(7)
    .mean()
)


# HR deviation from personal baseline

master_df["HR_Deviation"] = (
    master_df["Resting_HR"]
    - master_df["HR_7day_avg"]
)


# Sleep deviation

master_df["Sleep_Deviation"] = (
    master_df["Sleep_Score"]
    - master_df["Sleep_7day_avg"]
)


# Steps deviation

master_df["Steps_Deviation"] = (
    master_df["Steps"]
    - master_df["Steps_7day_avg"]
)


#std

master_df["HR_7day_std"] = (
    master_df["Resting_HR"]
    .rolling(7)
    .std()
)

master_df["Sleep_7day_std"] = (
    master_df["Sleep_Score"]
    .rolling(7)
    .std()
)

master_df["Steps_7day_std"] = (
    master_df["Steps"]
    .rolling(7)
    .std()
)



master_df["HR_Zscore"] = (
    master_df["HR_Deviation"]
    / master_df["HR_7day_std"]
)

master_df["Sleep_Zscore"] = (
    master_df["Sleep_Deviation"]
    / master_df["Sleep_7day_std"]
)

master_df["Steps_Zscore"] = (
    master_df["Steps_Deviation"]
    / master_df["Steps_7day_std"]
)




master_df = master_df.dropna()


master_df.to_csv(
    "feature_engineered_dataset.csv",
    index=False
)