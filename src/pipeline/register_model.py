import os
import json
import logging
import mlflow
from mlflow.tracking import MlflowClient

# ------------------- Logger -------------------
logger = logging.getLogger("model_registration")
logger.setLevel(logging.DEBUG)

log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../log"))
os.makedirs(log_dir, exist_ok=True)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(log_dir, "model_registration.log"))
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ------------------- Functions -------------------
def load_model_info(file_path: str) -> dict:
    try:
        with open(file_path, "r") as f:
            model_info = json.load(f)
        logger.debug(f"Model info loaded from {file_path}")
        return model_info
    except Exception as e:
        logger.error(f"Error loading model info: {e}")
        raise

def register_model(model_name: str, model_info: dict):
    try:
        model_uri = f"runs:/{model_info['mlflow_run_id']}/{model_info['model_path']}"
        logger.debug(f"Registering model from URI: {model_uri}")

        model_version = mlflow.register_model(model_uri, model_name)

        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        logger.info(f"Model '{model_name}' version {model_version.version} registered and moved to Staging.")
    except Exception as e:
        logger.error(f"Error registering model: {e}")
        raise

# ------------------- Main -------------------
def main():
    try:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")

        model_name = "customer_churn_rf_model"
        model_info_path = os.path.join(log_dir, "experiment_info.json")

        model_info = load_model_info(model_info_path)
        register_model(model_name, model_info)

    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
