import numpy as np
import joblib

def load_model(model_path: str):
    return joblib.load(model_path)

def predict_admission(model, input_features: list) -> str:
    input_array = np.array([input_features])
    result = model.predict(input_array)
    return "Admitted" if result[0] == 1 else "Not Admitted"