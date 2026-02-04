# models/lightgbm_model.py

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score


def train(X_train, y_train):
	model = LGBMRegressor(
		n_estimators=500,
		learning_rate=0.05,
		max_depth=-1,
		subsample=0.8,
		colsample_bytree=0.8,
		random_state=42,
		n_jobs=-1
	)

	model.fit(X_train, y_train)
	return model


def predict(model, X_test):
	return model.predict(X_test)


def evaluate(y_true, y_pred):
	rmse = np.sqrt(mean_squared_error(y_true, y_pred))
	r2 = r2_score(y_true, y_pred)

	return {
		"rmse": rmse,
		"r2": r2
	}


def get_name():
	return "LightGBM"
