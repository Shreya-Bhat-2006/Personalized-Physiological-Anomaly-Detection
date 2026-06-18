import numpy as np
import pandas as pd
p1 = pd.read_csv("feature_engineered_dataset.csv")

pop = pd.read_csv(
    "population_feature_engineered_dataset.csv"
)

pop_p01 = (
    pop[pop["Participant"] == "p01"]
    .drop(columns=["Participant"])
    .reset_index(drop=True)
)

for col in p1.columns:

    if col == "Date":
        continue

    same = np.allclose(
        p1[col],
        pop_p01[col],
        equal_nan=True
    )

    print(col, same)