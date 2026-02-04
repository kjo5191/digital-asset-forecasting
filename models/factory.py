# models/factory.py

from .random_forest_model import RandomForestPriceModel
from .lightgbm_model import LightGBMPriceModel
from .lstm_model import LSTMPriceModel

_MODEL_REGISTRY = {
	"rf": RandomForestPriceModel,
	"lgbm": LightGBMPriceModel,
	"lstm": LSTMPriceModel,
	# "prophet": ProphetPriceModel,
}


def get_model(model_name: str):
	try:
		model_class = _MODEL_REGISTRY[model_name]
	except KeyError:
		raise ValueError(f"Unknown model: {model_name}")

	return model_class()
