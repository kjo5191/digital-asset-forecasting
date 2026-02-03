# models.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def train_random_forest(df_ml: pd.DataFrame, features: list[str]):
	X = df_ml[features]
	y = df_ml["price"]

	split_idx = int(len(df_ml) * 0.8)

	X_train = X.iloc[:split_idx]
	y_train = y.iloc[:split_idx]
	X_test = X.iloc[split_idx:]
	y_test = y.iloc[split_idx:]

	model = RandomForestRegressor(
		n_estimators=200,
		n_jobs=-1,
		random_state=42
	)
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)

	rmse = np.sqrt(mean_squared_error(y_test, y_pred))
	r2 = r2_score(y_test, y_pred)

	return model, y_test, y_pred, split_idx, rmse, r2
