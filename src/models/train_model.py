import joblib
import pandas as pd

from xgboost import XGBRegressor

X = pd.read_csv("data/processed_data/X_train_scaled.csv")
y = pd.read_csv("data/processed_data/y_train.csv").squeeze()

params = joblib.load("models/best_params.pkl")

model = XGBRegressor(
    **params,
    random_state=42,
    objective="reg:squarederror"
)

model.fit(X, y)

joblib.dump(model, "models/model.pkl")