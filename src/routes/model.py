from fastapi import APIRouter, HTTPException, status
import logging
import os
import pickle
import numpy as np
from routes.schemes.data import SimpleChurnResponse, ProcessReuest
from controllers.ProcessController import ProcessController

logger = logging.getLogger(__name__)

model_router = APIRouter(
    prefix="/api/v1/model",
    tags=["api_v1", "model", "prediction"]
)

# Load model
try:
    model_path = os.path.join("models", "rf_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None


EXPECTED_FEATURES = [
    "Age",
    "Gender",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Subscription Type",
    "Contract Length",
    "Total Spent",
    "Last Interaction"
]


@model_router.post("/predict", response_model=SimpleChurnResponse)
async def predict_churn(request: ProcessReuest):
    """
    Predict customer churn for a single customer
    """
    try:
     
        raw_data = request.model_dump(by_alias=True)

  
        processed_data = ProcessController().process_data(raw_data)

     
        input_values = list(processed_data.values())
        input_array = np.array(input_values).reshape(1, -1)

        prediction = model.predict(input_array)[0]
        return SimpleChurnResponse(churn_prediction=int(prediction))

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making prediction: {str(e)}"
        )
