import joblib
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


X = pd.read_csv(
    "data/processed_data/X_train_scaled.csv"
)

y = pd.read_csv(
    "data/processed_data/y_train.csv"
).squeeze()


params = {
    "n_estimators": [100, 200],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}


grid = GridSearchCV(
    XGBRegressor(
        random_state=42,
        objective="reg:squarederror"
    ),
    params,
    cv=3,
    scoring="r2",
    n_jobs=-1
)

grid.fit(X, y)

print(grid.best_params_)
print(grid.best_score_)

joblib.dump(grid.best_params_, "models/best_params.pkl")