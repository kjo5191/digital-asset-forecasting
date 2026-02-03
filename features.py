# features.py
# 아이템 필터 + RSI/Bollinger + 학습용 데이터 생성

import pandas as pd


def filter_item(df_final: pd.DataFrame, target_keyword: str, target_grade: str | None):
	mask = df_final["name"].str.contains(target_keyword)

	if target_grade and target_grade != "전체":
		mask = mask & (df_final["grade"] == target_grade)

	df_target = df_final[mask].copy().sort_values("date")

	if len(df_target) == 0:
		return None

	# 데이터가 제일 많은 아이템 하나 자동 선택
	top_item = df_target["name"].value_counts().idxmax()
	df_target = df_target[df_target["name"] == top_item]

	return df_target, top_item


def calculate_rsi(series: pd.Series, window: int = 14):
	delta = series.diff()
	gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
	loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
	rs = gain / loss
	return 100 - (100 / (1 + rs))


def calculate_bollinger(series: pd.Series, window: int = 20):
	sma = series.rolling(window=window).mean()
	std = series.rolling(window=window).std()
	return sma + (std * 2), sma - (std * 2)


def make_ml_dataset(df_target: pd.DataFrame):
	df_ml = df_target.copy()

	# (1) Lag features
	df_ml["lag_10m"] = df_ml["price"].shift(1)
	df_ml["lag_1h"] = df_ml["price"].shift(6)
	df_ml["lag_24h"] = df_ml["price"].shift(144)

	# (2) RSI, Bollinger
	df_ml["rsi"] = calculate_rsi(df_ml["price"])
	df_ml["bb_upper"], df_ml["bb_lower"] = calculate_bollinger(df_ml["price"])

	# (3) 상태 정보
	df_ml["is_overbought"] = (df_ml["price"] > df_ml["bb_upper"]).astype(int)
	df_ml["is_oversold"] = (df_ml["price"] < df_ml["bb_lower"]).astype(int)

	# (4) 시간 정보
	df_ml["hour"] = df_ml["date"].dt.hour
	df_ml["day_of_week"] = df_ml["date"].dt.dayofweek

	# NaN 제거
	df_ml = df_ml.dropna()

	features = [
		"lag_10m", "lag_1h", "lag_24h",
		"rsi", "is_overbought", "is_oversold",
		"hour", "day_of_week"
	]

	return df_ml, features
