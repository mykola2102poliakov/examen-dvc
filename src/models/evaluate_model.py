import json
import joblib
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score

X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv").squeeze()

model = joblib.load("models/model.pkl")

pred = model.predict(X_test)

pd.DataFrame({"prediction": pred}).to_csv(
    "data/prediction.csv",
    index=False,
)

scores = {
    "mse": float(mean_squared_error(y_test, pred)),
    "r2": float(r2_score(y_test, pred))
}

with open("metrics/scores.json", "w") as f:
    json.dump(scores, f, indent=4)