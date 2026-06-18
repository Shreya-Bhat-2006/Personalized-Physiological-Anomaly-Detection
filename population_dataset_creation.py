import pandas as pd
import json
import numpy as np
from datetime import datetime
import os

# List of participants
participants = [
    'p01', 'p02', 'p03', 'p04', 'p05', 'p06', 'p07', 'p08', 
    'p09', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16'
]

# Initialize empty list to store all participant dataframes
all_participants_df = []

def create_master_dataset(participant):
    """Create master dataset for a single participant"""
    
    try:
        # Sleep data
        sleep = pd.read_csv(f"pmdata/{participant}/fitbit/sleep_score.csv")
        sleep = sleep[["timestamp", "overall_score", "deep_sleep_in_minutes", "restlessness"]]
        sleep = sleep.rename(columns={
            "timestamp": "Date",
            "overall_score": "Sleep_Score",
            "deep_sleep_in_minutes": "Deep_Sleep",
            "restlessness": "Restlessness"
        })
        sleep["Date"] = pd.to_datetime(sleep["Date"]).dt.date
        
        # Resting heart rate
        try:
            with open(f"pmdata/{participant}/fitbit/resting_heart_rate.json", "r") as f:
                resting_hr = json.load(f)
            
            rows = []
            for record in resting_hr:
                rows.append({
                    "Date": record["dateTime"],
                    "Resting_HR": record["value"]["value"]
                })
            resting_hr = pd.DataFrame(rows)
            resting_hr["Date"] = pd.to_datetime(resting_hr["Date"]).dt.date
            
            # Merge sleep and resting HR
            master_df = pd.merge(sleep, resting_hr, on="Date", how="inner")
            
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Warning: {participant} missing resting_heart_rate.json - skipping participant")
            return None
        
        # Steps
        try:
            with open(f"pmdata/{participant}/fitbit/steps.json", "r") as f:
                steps = json.load(f)
            
            steps = pd.DataFrame(steps)
            steps = steps.rename(columns={"dateTime": "Date", "value": "Steps"})
            steps["Date"] = pd.to_datetime(steps["Date"])
            steps["Steps"] = steps["Steps"].astype(int)
            steps["Date"] = steps["Date"].dt.date
            steps = steps.groupby("Date")["Steps"].sum().reset_index()
            
            master_df = pd.merge(master_df, steps, on="Date", how="inner")
            
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Warning: {participant} missing steps.json - skipping participant")
            return None
        
        # Calories
        try:
            with open(f"pmdata/{participant}/fitbit/calories.json", "r") as f:
                calories = json.load(f)
            
            calories = pd.DataFrame(calories)
            calories = calories.rename(columns={"dateTime": "Date", "value": "Calories"})
            calories["Date"] = pd.to_datetime(calories["Date"])
            calories["Calories"] = calories["Calories"].astype(float)
            calories["Date"] = calories["Date"].dt.date
            calories = calories.groupby("Date")["Calories"].sum().reset_index()
            
            master_df = pd.merge(master_df, calories, on="Date", how="inner")
            
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Warning: {participant} missing calories.json - skipping participant")
            return None
        
        # Moderate activity
        try:
            with open(f"pmdata/{participant}/fitbit/moderately_active_minutes.json", "r") as f:
                moderate = json.load(f)
            
            moderate = pd.DataFrame(moderate)
            moderate = moderate.rename(columns={"dateTime": "Date", "value": "Moderate_Activity"})
            moderate["Date"] = pd.to_datetime(moderate["Date"]).dt.date
            moderate["Moderate_Activity"] = moderate["Moderate_Activity"].astype(int)
            
            master_df = pd.merge(master_df, moderate, on="Date", how="inner")
            
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Warning: {participant} missing moderately_active_minutes.json - skipping participant")
            return None
        
        # Vigorous activity
        try:
            with open(f"pmdata/{participant}/fitbit/very_active_minutes.json", "r") as f:
                vigorous = json.load(f)
            
            vigorous = pd.DataFrame(vigorous)
            vigorous = vigorous.rename(columns={"dateTime": "Date", "value": "Vigorous_Activity"})
            vigorous["Date"] = pd.to_datetime(vigorous["Date"]).dt.date
            vigorous["Vigorous_Activity"] = vigorous["Vigorous_Activity"].astype(int)
            
            master_df = pd.merge(master_df, vigorous, on="Date", how="inner")
            
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Warning: {participant} missing very_active_minutes.json - skipping participant")
            return None
        
        # Wellness data
        try:
            wellness = pd.read_csv(f"pmdata/{participant}/pmsys/wellness.csv")
            wellness = wellness[["effective_time_frame", "fatigue", "mood", "readiness", "stress"]]
            wellness = wellness.rename(columns={
                "effective_time_frame": "Date",
                "fatigue": "Fatigue",
                "mood": "Mood",
                "readiness": "Readiness",
                "stress": "Stress"
            })
            wellness["Date"] = pd.to_datetime(wellness["Date"]).dt.date
            
            master_df = pd.merge(master_df, wellness, on="Date", how="inner")
            
        except (FileNotFoundError, KeyError) as e:
            print(f"Warning: {participant} missing wellness.csv - skipping participant")
            return None
        
        # Reporting data
        try:
            reporting = pd.read_csv(f"pmdata/{participant}/googledocs/reporting.csv")
            reporting = reporting[["date", "weight", "glasses_of_fluid"]]
            reporting = reporting.rename(columns={
                "date": "Date",
                "weight": "Weight",
                "glasses_of_fluid": "Fluid_Intake"
            })
            reporting["Date"] = pd.to_datetime(reporting["Date"], dayfirst=True).dt.date
            reporting = reporting.groupby("Date").agg({
                "Weight": "mean",
                "Fluid_Intake": "mean"
            }).reset_index()
            
            master_df = pd.merge(master_df, reporting, on="Date", how="left")
            master_df = master_df.drop(columns=["Weight"])
            
            # Fill missing Fluid_Intake with median
            master_df["Fluid_Intake"] = master_df["Fluid_Intake"].fillna(master_df["Fluid_Intake"].median())
            
        except (FileNotFoundError, KeyError) as e:
            print(f"Warning: {participant} missing reporting.csv - skipping participant")
            return None
        
        # Verify no missing values remain
        if master_df.isnull().sum().sum() > 0:
            print(f"Warning: {participant} has missing values - skipping participant")
            return None
        
        return master_df
        
    except Exception as e:
        print(f"Error processing {participant}: {str(e)}")
        return None

def perform_feature_engineering(df):
    """Perform feature engineering on the dataset"""
    
    # Ensure Date is datetime
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Calculate 7-day rolling averages
    df["HR_7day_avg"] = df["Resting_HR"].rolling(7).mean()
    df["Sleep_7day_avg"] = df["Sleep_Score"].rolling(7).mean()
    df["Steps_7day_avg"] = df["Steps"].rolling(7).mean()
    
    # Calculate deviations
    df["HR_Deviation"] = df["Resting_HR"] - df["HR_7day_avg"]
    df["Sleep_Deviation"] = df["Sleep_Score"] - df["Sleep_7day_avg"]
    df["Steps_Deviation"] = df["Steps"] - df["Steps_7day_avg"]
    
    # Calculate 7-day rolling standard deviations
    df["HR_7day_std"] = df["Resting_HR"].rolling(7).std()
    df["Sleep_7day_std"] = df["Sleep_Score"].rolling(7).std()
    df["Steps_7day_std"] = df["Steps"].rolling(7).std()
    
    # Calculate Z-scores
    df["HR_Zscore"] = df["HR_Deviation"] / df["HR_7day_std"]
    df["Sleep_Zscore"] = df["Sleep_Deviation"] / df["Sleep_7day_std"]
    df["Steps_Zscore"] = df["Steps_Deviation"] / df["Steps_7day_std"]
    
    # Drop rows with NaN (from rolling calculations)
    df = df.dropna()
    
    return df

# Process each participant
for participant in participants:
    print(f"Processing {participant}...")
    
    try:
        # Create master dataset
        master_df = create_master_dataset(participant)
        
        if master_df is not None:
            # Perform feature engineering
            engineered_df = perform_feature_engineering(master_df)
            
            # Add participant column
            engineered_df["Participant"] = participant
            
            # Append to list
            all_participants_df.append(engineered_df)
            print(f"Successfully processed {participant} - {len(engineered_df)} rows")
        else:
            print(f"Skipped {participant} due to missing data")
            
    except Exception as e:
        print(f"Error processing {participant}: {str(e)}")
        continue

# Combine all participants
if all_participants_df:
    final_df = pd.concat(all_participants_df, ignore_index=True)
    
    # Ensure correct column order
    column_order = [
        "Date", "Sleep_Score", "Deep_Sleep", "Restlessness", "Resting_HR",
        "Steps", "Calories", "Moderate_Activity", "Vigorous_Activity",
        "Fatigue", "Mood", "Readiness", "Stress", "Fluid_Intake",
        "HR_7day_avg", "Sleep_7day_avg", "Steps_7day_avg",
        "HR_Deviation", "Sleep_Deviation", "Steps_Deviation",
        "HR_7day_std", "Sleep_7day_std", "Steps_7day_std",
        "HR_Zscore", "Sleep_Zscore", "Steps_Zscore",
        "Participant"
    ]
    
    final_df = final_df[column_order]
    
    # Save final dataset
    final_df.to_csv("population_feature_engineered_dataset.csv", index=False)
    print(f"\nSuccessfully created population_feature_engineered_dataset.csv")
    print(f"Total rows: {len(final_df)}")
    print(f"Total participants: {final_df['Participant'].nunique()}")
    print(f"Participants included: {sorted(final_df['Participant'].unique())}")
else:
    print("No data was processed successfully. Check your data files.")


