from fastapi import FastAPI 
from routes import base
from routes import data
from routes import model

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using machine learning",
    version="1.0.0"
)

app.include_router(base.base_router)
app.include_router(data.data_router)
app.include_router(model.model_router)