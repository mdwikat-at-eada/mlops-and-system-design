from datetime import datetime
import joblib
from metadata import MODEL_NAME, MODELS_FOLDER


def save_model(model):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_path = f"{MODELS_FOLDER}/{MODEL_NAME}-{timestamp}.joblib"

    joblib.dump(model, model_path)
    return model_path
