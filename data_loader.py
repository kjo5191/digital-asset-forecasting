# data_loader.py

import pandas as pd
from sqlalchemy import create_engine
import streamlit as st

DB_HOST = "projectl-db.cpusekkcm87u.ap-northeast-2.rds.amazonaws.com"
DB_PORT = 3306
DB_USER = "admin"
DB_PASSWORD = "rjqnr0824**"  # ⚠ 실제로는 환경변수나 .env 로 빼는게 좋음
DB_NAME = "projectl"


@st.cache_resource
def get_engine():
	db_connection_str = (
		f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}"
		f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
	)
	engine = create_engine(db_connection_str)
	return engine


@st.cache_data
def load_merged_data():
	engine = get_engine()

	df_logs = pd.read_sql("SELECT * FROM market_price_logs", engine)
	df_items = pd.read_sql("SELECT id, name, grade, category_code FROM market_items", engine)

	df_merged = pd.merge(
		df_logs,
		df_items,
		left_on="item_id",
		right_on="id",
		how="left"
	)

	df_final = df_merged.drop(columns=["id_y", "id_x"])
	df_final = df_final.rename(columns={
		"current_min_price": "price",
		"logged_at": "date"
	})

	df_final["date"] = pd.to_datetime(df_final["date"])
	df_final = df_final.sort_values("date")

	return df_final
