import os
import pickle
import numpy as np
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn

# ------------------- Logger -------------------
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../log"))
os.makedirs(log_dir, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_dir, 'model_evaluation.log'))
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ------------------- Functions -------------------
def load_model(model_path: str):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"Model loaded from {model_path}")
    return model

def load_data(data_path: str):
    processed_data_path = os.path.join(data_path, "processed")
    X_valid = np.load(os.path.join(processed_data_path, "X_valid_scaled.npy"))
    y_valid = np.load(os.path.join(processed_data_path, "y_valid.npy"))
    X_test = np.load(os.path.join(processed_data_path, "X_test_scaled.npy"))
    y_test = np.load(os.path.join(processed_data_path, "y_test.npy"))
    logger.info("Validation and test data loaded successfully")
    return X_valid, y_valid, X_test, y_test

def evaluate(model, X, y):
    y_pred = model.predict(X)
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1_score": f1_score(y, y_pred, zero_division=0)
    }
    return metrics, y_pred

def plot_confusion_matrix(y_true, y_pred, dataset_name: str, output_dir: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{dataset_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{dataset_name}_confusion_matrix.png")
    plt.savefig(file_path)
    plt.close()
    logger.info(f"Confusion matrix saved at {file_path}")
    return file_path

def log_metrics_and_artifacts_mlflow(metrics: dict, dataset_name: str, model, cm_path: str):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Customer_Churn_RF")
    artifact_path = f"{dataset_name}_model"
    with mlflow.start_run(run_name=f"Evaluation_{dataset_name}") as run:
        for key, value in metrics.items():
            mlflow.log_metric(f"{dataset_name}_{key}", value)
        mlflow.sklearn.log_model(model, artifact_path=artifact_path)
        mlflow.log_artifact(cm_path, artifact_path=f"{dataset_name}_artifacts")
        run_id = run.info.run_id
    return run_id, artifact_path

def save_experiment_info(metrics_valid, metrics_test, cm_valid_path, cm_test_path, run_id, model_path, output_dir="log"):
    os.makedirs(output_dir, exist_ok=True)
    experiment_info = {
        "validation": metrics_valid,
        "test": metrics_test,
        "validation_confusion_matrix": cm_valid_path,
        "test_confusion_matrix": cm_test_path,
        "mlflow_run_id": run_id,
        "model_path": model_path
    }
    file_path = os.path.join(output_dir, "experiment_info.json")
    with open(file_path, "w") as f:
        json.dump(experiment_info, f, indent=4)
    logger.info(f"Experiment info saved to {file_path}")

# ------------------- Main -------------------
def main():
    try:
        logger.debug("Starting model evaluation...")
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        model_path = os.path.join(root_dir, "models", "rf_model.pkl")
        data_path = os.path.join(root_dir, "Data")

        model = load_model(model_path)
        X_valid, y_valid, X_test, y_test = load_data(data_path)

        valid_metrics, valid_pred = evaluate(model, X_valid, y_valid)
        test_metrics, test_pred = evaluate(model, X_test, y_test)

        cm_valid_path = plot_confusion_matrix(y_valid, valid_pred, "validation", log_dir)
        cm_test_path = plot_confusion_matrix(y_test, test_pred, "test", log_dir)

        run_id, model_artifact_path = log_metrics_and_artifacts_mlflow(valid_metrics, "validation", model, cm_valid_path)
        # يمكن تسجيل test metrics لو تحب
        _ = log_metrics_and_artifacts_mlflow(test_metrics, "test", model, cm_test_path)

        save_experiment_info(valid_metrics, test_metrics, cm_valid_path, cm_test_path, run_id, model_artifact_path, output_dir=log_dir)

        logger.info(f"Validation metrics: {valid_metrics}")
        logger.info(f"Test metrics: {test_metrics}")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise
    finally:
        logger.debug("Model evaluation completed")

if __name__ == "__main__":
    main()
