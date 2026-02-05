# models/io.py

import os
from pathlib import Path
from typing import Optional

import joblib

from .factory import get_model


# ---------------------------------------------------------------------
# 1. 모델 저장 기본 경로
#    예) trained_models/rf/item_123.pkl
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # 프로젝트 루트 기준
MODEL_DIR = BASE_DIR / "trained_models"


def _ensure_model_dir():
	if not MODEL_DIR.exists():
		MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _model_filename(model_key: str, item_id: Optional[int] = None) -> Path:
	"""
	모델 키 + 아이템 ID 조합으로 파일 경로 생성.
	item_id가 없으면 공통 모델로 취급.
	"""
	_ensure_model_dir()

	subdir = MODEL_DIR / model_key
	if not subdir.exists():
		subdir.mkdir(parents=True, exist_ok=True)

	if item_id is not None:
		filename = f"{model_key}_item_{item_id}.pkl"
	else:
		filename = f"{model_key}_global.pkl"

	return subdir / filename


# ---------------------------------------------------------------------
# 2. 저장 / 로드 헬퍼
# ---------------------------------------------------------------------
def save_model(model_key: str, item_id: Optional[int], price_model) -> Path:
	"""
	RF / LGBM / LSTM PriceModel 인스턴스를 그대로 joblib으로 저장.
	(LSTM은 Keras 버전/환경에 따라 pickle 문제가 날 수 있음 → 안 되면 LSTM만 예외 처리)
	"""
	path = _model_filename(model_key, item_id)
	joblib.dump(price_model, path)
	return path


def load_model(model_key: str, item_id: Optional[int]):
	"""
	기존에 저장된 모델을 로드. 없으면 None 반환.
	"""
	path = _model_filename(model_key, item_id)
	if not path.exists():
		return None

	price_model = joblib.load(path)
	return price_model


# ---------------------------------------------------------------------
# 3. Streamlit에서 쓸 "load or train" 헬퍼
# ---------------------------------------------------------------------
def load_or_train_model(
	model_key: str,
	item_id: Optional[int],
	df_ml,
	features,
	force_retrain: bool = False,
):
	if not force_retrain:
		existing = load_model(model_key, item_id)
		if existing is not None:
			return existing, "loaded"

	price_model = get_model(model_key)
	price_model.train(df_ml, features)

	try:
		save_model(model_key, item_id, price_model)
	except Exception as e:
		print(f"[WARN] 모델 저장 실패: {e}")

	return price_model, "trained"

