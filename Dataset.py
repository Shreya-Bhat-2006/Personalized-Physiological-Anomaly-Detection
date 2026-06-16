from sqlite3 import Date

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







#-----------------
#calories



# Load calories data
with open(
    "pmdata/p01/fitbit/calories.json",
    "r"
) as f:
    Calories=json.load(f)



# Convert to DataFrame
Calories=pd.DataFrame(Calories)



# Rename columns
Calories=Calories.rename(
    columns={
        "dateTime":"Date",
        "value":"Calories"
    }

)


# Convert date
Calories["Date"]= pd.to_datetime(
    Calories["Date"]
)


# Convert calories to float
Calories["Calories"]=Calories["Calories"].astype(float)


# Keep only date

Calories["Date"] = Calories["Date"].dt.date

# Sum calories per day

Calories = (
    Calories.groupby("Date")["Calories"]
    .sum()
    .reset_index()
)

# Merge calories

master_df = pd.merge(
    master_df,
    Calories,
    on="Date",
    how="inner"
)


#-----------------
# moderate activity 

# Load moderate activity data

with open(
    "pmdata/p01/fitbit/moderately_active_minutes.json",
    "r"
) as f:
    moderate = json.load(f)


moderate = pd.DataFrame(moderate)

moderate = moderate.rename(
    columns={
        "dateTime": "Date",
        "value": "Moderate_Activity"
    }
)


moderate["Date"] = pd.to_datetime(
    moderate["Date"]
).dt.date


moderate["Moderate_Activity"] = moderate["Moderate_Activity"].astype(int)




master_df = pd.merge(
    master_df,
    moderate,
    on="Date",
    how="inner"
)






#---------------
#vigorous activity


# Load vigorous activity data

with open(
    "pmdata/p01/fitbit/very_active_minutes.json",
    "r"
) as f:
    vigorous = json.load(f)


# Convert to DataFrame

vigorous = pd.DataFrame(vigorous)


# Rename columns

vigorous = vigorous.rename(
    columns={
        "dateTime": "Date",
        "value": "Vigorous_Activity"
    }
)


# Convert Date

vigorous["Date"] = pd.to_datetime(
    vigorous["Date"]
).dt.date


# Convert datatype

vigorous["Vigorous_Activity"] = (
    vigorous["Vigorous_Activity"]
    .astype(int)
)


# Merge

master_df = pd.merge(
    master_df,
    vigorous,
    on="Date",
    how="inner"
)


#-----------------
#wellness



# Load wellness data

wellness = pd.read_csv(
    "pmdata/p01/pmsys/wellness.csv"
)


# Keep required columns

wellness = wellness[
    [
        "effective_time_frame",
        "fatigue",
        "mood",
        "readiness",
        "stress"
    ]
]


# Rename columns

wellness = wellness.rename(
    columns={
        "effective_time_frame": "Date",
        "fatigue": "Fatigue",
        "mood": "Mood",
        "readiness": "Readiness",
        "stress": "Stress"
    }
)


# Convert Date column

wellness["Date"] = pd.to_datetime(
    wellness["Date"]
).dt.date





# Merge with master dataset

master_df = pd.merge(
    master_df,
    wellness,
    on="Date",
    how="inner"
)



#-----------------
#reporting data

# Load reporting data

reporting = pd.read_csv(
    "pmdata/p01/googledocs/reporting.csv"
)


# Keep required columns

reporting = reporting[
    [
        "date",
        "weight",
        "glasses_of_fluid"
    ]
]


# Rename columns

reporting = reporting.rename(
    columns={
        "date": "Date",
        "weight": "Weight",
        "glasses_of_fluid": "Fluid_Intake"
    }
)


# Convert Date

reporting["Date"] = pd.to_datetime(
    reporting["Date"],
    dayfirst=True
).dt.date

reporting = (
    reporting.groupby("Date")
    .agg({
        "Weight": "mean",
        "Fluid_Intake": "mean"
    })
    .reset_index()
)


# Merge with master dataset

master_df = pd.merge(
    master_df,
    reporting,
    on="Date",
    how="left"
)



master_df = master_df.drop(
    columns=["Weight"]
)

# Fill missing Fluid_Intake with median

master_df["Fluid_Intake"] = (
    master_df["Fluid_Intake"]
    .fillna(
        master_df["Fluid_Intake"].median()
    )
)

# Verify no missing values remain

print(master_df.isnull().sum())


# Check final shape

print(master_df.shape)



# Save final dataset

master_df.to_csv(
    "master_dataset_v1.csv",
    index=False
)