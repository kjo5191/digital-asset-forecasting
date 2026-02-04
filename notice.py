# notice.py

import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

@st.cache_resource
def get_engine():
	db_connection_str = (
		f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}"
		f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
	)
	engine = create_engine(db_connection_str)
	return engine

@st.cache_data
def load_notice_all():
	engine = get_engine()
	
	df_notice = pd.read_sql("SELECT * FROM raw_notices", engine)

	return df_notice

@st.cache_data
def load_notice_content():
	engine = get_engine()
	
	df_notice_content = pd.read_sql("SELECT content FROM raw_notices", engine)

	return df_notice_content

# if __name__ == "__main__":
# 	df = load_notice()
# 	print(df.head())
# 	print("rows:", len(df))

notice = load_notice_content()
print(notice)
