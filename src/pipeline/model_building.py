import logging
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import os
import yaml


logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise
    finally:
        logger.debug('Finished loading parameters')


def load_processed_train_data(data_path: str):
    try:
        processed_data_path = os.path.join(data_path, "processed")
        logger.debug(f"Loading data from {processed_data_path}")
        
        X_train_scaled = np.load(os.path.join(processed_data_path, 'X_train_scaled.npy'))
        y_train = np.load(os.path.join(processed_data_path, 'y_train.npy'))

        logger.info("Processed data loaded successfully")
        return X_train_scaled, y_train
    except Exception as e:
        logger.error(f"Error loading processed data: {e}")
        raise e


def train_rf(X_train_scaled, y_train, params: dict):
    try:
        logger.debug("Training Random Forest model...")
        rf = RandomForestClassifier(**params)
        rf.fit(X_train_scaled, y_train)
        logger.info("Random Forest model trained successfully")
        return rf
    except Exception as e:
        logger.error(f"Error training Random Forest model: {e}")
        raise e


def save_model(model, model_path: str):
    try:
        # ✅ اتأكد إن فولدر models موجود
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        logger.debug(f"Saving model to {model_path}")
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        logger.info("Model saved successfully")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise e
    finally:
        logger.debug('Finished saving model')


def get_src_directory() -> str:
    """Get the src directory (one level up from this script's location)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../'))


def main():
    try:
        logger.debug("Starting model building...")
        params = load_params("params.yaml")
        
        X_train_scaled, y_train = load_processed_train_data("data")
        rf = train_rf(X_train_scaled, y_train, params["model_building"])

        # ✅ Save model in src directory
        model_path = os.path.join(get_src_directory(), "models", "rf_model.pkl")
        save_model(rf, model_path)
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise e
    finally:
        logger.debug("Model building completed")


if __name__ == '__main__':
    main()
