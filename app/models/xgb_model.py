from pathlib import Path
import xgboost as xgb

path_model = Path(__file__).parent / "mnist-xgboost-v1.json"

model = xgb.XGBClassifier()

model.load_model(path_model)
