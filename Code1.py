import pandas as pd
import json
import numpy as np


#sleep data

# Load sleep data
sleep = pd.read_csv(
    "pmdata/p01/fitbit/sleep_score.csv"
)


# Keep required columns
sleep = sleep[
    [
        "timestamp",
        "overall_score",
        "deep_sleep_in_minutes",
        "restlessness"
    ]
]


# Rename columns
sleep = sleep.rename(
    columns={
        "timestamp": "Date",
        "overall_score": "Sleep_Score",
        "deep_sleep_in_minutes": "Deep_Sleep",
        "restlessness": "Restlessness"
    }
)


# Convert Date column
sleep["Date"] = pd.to_datetime(
    sleep["Date"]
).dt.date


#-----------------------------------

#resting_heart_rate

## Load resting heart rate data

with open(
    "pmdata/p01/fitbit/resting_heart_rate.json",
    "r"
) as f:
    resting_hr = json.load(f)


# Extract required fields

rows = []

for record in resting_hr:

    rows.append({
        "Date": record["dateTime"],
        "Resting_HR": record["value"]["value"]
    })


# Convert list to DataFrame

resting_hr = pd.DataFrame(rows)


# Convert Date column

resting_hr["Date"] = pd.to_datetime(
    resting_hr["Date"]
).dt.date





# Merge sleep and resting HR

master_df = pd.merge(
    sleep,
    resting_hr,
    on="Date",
    how="inner"
)




#-----------------

#steps

# Load steps data

with open(
    "pmdata/p01/fitbit/steps.json",
    "r"
) as f:
    steps = json.load(f)


# Convert to DataFrame

steps = pd.DataFrame(steps)


# Rename columns

steps = steps.rename(
    columns={
        "dateTime": "Date",
        "value": "Steps"
    }
)


# Convert data types

steps["Date"] = pd.to_datetime(
    steps["Date"]
)

steps["Steps"] = steps["Steps"].astype(int)


# Extract only date

steps["Date"] = steps["Date"].dt.date


# Aggregate minute-wise steps into daily steps

steps = (
    steps.groupby("Date")["Steps"]
    .sum()
    .reset_index()
)


# Merge steps with master_df


master_df = pd.merge(
    master_df,
    steps,
    on="Date",
    how="inner"
)


print(master_df.head())

print(master_df.shape)

print(master_df.isnull().sum())