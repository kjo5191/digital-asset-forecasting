# models/lightgbm_model.py

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score

from .base import BasePriceModel


class LightGBMPriceModel(BasePriceModel):
	def __init__(self):
		self.model = None
		self.df = None
		self.features = None
		self.split_idx = None
		self.y_test = None
		self.y_pred = None
		self.rmse = None
		self.r2 = None

	def train(self, df: pd.DataFrame, features: list[str]):
		"""
		RandomForestPriceModel 이랑 인터페이스 맞춤
		- df: feature + price + date 를 포함한 전체 데이터프레임
		- features: 학습에 사용할 컬럼 리스트
		"""
		self.df = df
		self.features = features

		X = df[features]
		y = df["price"]

		# 시계열 유지: 앞 80% train, 뒤 20% test
		self.split_idx = int(len(df) * 0.8)

		X_train = X.iloc[:self.split_idx]
		y_train = y.iloc[:self.split_idx]
		X_test = X.iloc[self.split_idx:]
		y_test = y.iloc[self.split_idx:]

		self.model = LGBMRegressor(
			n_estimators=500,
			learning_rate=0.05,
			max_depth=-1,
			subsample=0.8,
			colsample_bytree=0.8,
			random_state=42,
			n_jobs=-1,
		)

		self.model.fit(X_train, y_train)

		self.y_test = y_test
		self.y_pred = self.model.predict(X_test)

		self.rmse = np.sqrt(mean_squared_error(y_test, self.y_pred))
		self.r2 = r2_score(y_test, self.y_pred)

	def predict_test(self):
		"""
		테스트 구간 평가 결과 반환
		(RandomForestPriceModel 과 동일한 리턴형 유지)
		"""
		return (
			self.y_test,
			self.y_pred,
			self.split_idx,
			self.rmse,
			self.r2,
		)

	def predict_future(self, steps: int):
		"""
		단순 반복 예측 (RandomForestPriceModel 의 forecast_future 로직 재사용)
		- 마지막 row 기준으로 features 업데이트 없이 그대로 예측 반복
		"""
		last_row = self.df.iloc[-1].copy()
		future_rows = []

		current_row = last_row.copy()

		for _ in range(steps):
			X_current = current_row[self.features].values.reshape(1, -1)
			pred_price = self.model.predict(X_current)[0]

			# 예측 값을 현재 row의 price로 갱신
			current_row["price"] = pred_price
			current_row["date"] = current_row["date"] + pd.Timedelta(minutes=10)

			future_rows.append(
				{
					"date": current_row["date"],
					"price": pred_price,
				}
			)

		return pd.DataFrame(future_rows)
