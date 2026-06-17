import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "feature_engineered_dataset.csv"
)

df["Date"] = pd.to_datetime(
    df["Date"]
)

#RH


plt.figure(figsize=(12, 5))

plt.plot(
    df["Date"],
    df["Resting_HR"],
    label="Actual HR"
)

plt.plot(
    df["Date"],
    df["HR_7day_avg"],
    label="HR Baseline"
)

plt.title("Resting Heart Rate vs Personalized Baseline")
plt.xlabel("Date")
plt.ylabel("Heart Rate")

plt.legend()

plt.show()



#sleep

plt.figure(figsize=(12, 5))

plt.plot(
    df["Date"],
    df["Sleep_Score"],
    label="Actual Sleep"
)

plt.plot(
    df["Date"],
    df["Sleep_7day_avg"],
    label="Sleep Baseline"
)

plt.title("Sleep Score vs Personalized Baseline")
plt.xlabel("Date")
plt.ylabel("Sleep Score")

plt.legend()

plt.show()


#steps
plt.figure(figsize=(12, 5))

plt.plot(
    df["Date"],
    df["Steps"],
    label="Actual Steps"
)

plt.plot(
    df["Date"],
    df["Steps_7day_avg"],
    label="Steps Baseline"
)

plt.title("Steps vs Personalized Baseline")
plt.xlabel("Date")
plt.ylabel("Steps")

plt.legend()

plt.show()


#