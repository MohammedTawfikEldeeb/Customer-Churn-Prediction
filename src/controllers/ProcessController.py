import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from routes.schemes.data import ProcessReuest


class ProcessController:
    def __init__(self, artifacts_dir="artifacts"):
        self.artifacts_dir = artifacts_dir

      
        encoders_path = os.path.join(artifacts_dir, "label_encoders.pkl")
        if os.path.exists(encoders_path):
            with open(encoders_path, "rb") as f:
                self.label_encoders = pickle.load(f)
        else:
         
            raise FileNotFoundError(f"Encoders not found at {encoders_path}")

      
        scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
        else:
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")

    def process_data(self, data):
        
        data_dict = data

    
        for col, encoder in self.label_encoders.items():
    
            encoded_value = encoder.transform([data_dict[col]])[0]
            data_dict[col] = float(encoded_value)  # خليها float عشان scaler

        
        all_cols = list(data_dict.keys())
        values = np.array([data_dict[col] for col in all_cols], dtype=float).reshape(1, -1)
        scaled_values = self.scaler.transform(values)

     
        for col, val in zip(all_cols, scaled_values[0]):
            data_dict[col] = float(val)

        return data_dict
    
