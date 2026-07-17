import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "silica_concentrate"

FEATURES = [
    "ave_flot_air_flow",
    "ave_flot_level",
    "iron_feed",
    "starch_flow",
    "amina_flow",
    "ore_pulp_flow",
    "ore_pulp_pH",
    "ore_pulp_density"
]

df = pd.read_csv("data/raw_data/raw.csv")

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.to_csv("data/processed_data/X_train.csv", index=False)
X_test.to_csv("data/processed_data/X_test.csv", index=False)
y_train.to_csv("data/processed_data/y_train.csv", index=False)
y_test.to_csv("data/processed_data/y_test.csv", index=False)