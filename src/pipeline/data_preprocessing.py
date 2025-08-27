import pandas as pd
import numpy as np
import os
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('preprocessing_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Apply label encoding and combine data"""
    df_categorical = train_df.select_dtypes(include=['object']).columns
    label_encoders = {}

    try:
        for col in df_categorical:
            lbl = LabelEncoder()
            train_df[col] = lbl.fit_transform(train_df[col])
            test_df[col] = lbl.transform(test_df[col])
            label_encoders[col] = lbl  # نخزن encoder لكل عمود

        # دمج الداتا
        df = pd.concat([train_df, test_df], ignore_index=True)
        df = shuffle(df, random_state=42).reset_index(drop=True)

        # حفظ الـ encoders
        os.makedirs("src/artifacts", exist_ok=True)
        with open("src/artifacts/label_encoders.pkl", "wb") as f:
            pickle.dump(label_encoders, f)

        logger.debug(f"df shape: {df.shape}")
        logger.debug("Saving full data...")
        os.makedirs("./Data/raw", exist_ok=True)
        df.to_csv("./Data/raw/full_data.csv", index=False)

    except Exception as e:
        logger.error(f"Error in preprocess_data: {e}")
        raise e
    return df


def split_data(df: pd.DataFrame):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full,
        test_size=0.25,
        random_state=42,
        stratify=y_train_full
    )

    scaler = StandardScaler()
    scaler.fit(X_train)

    # حفظ الـ scaler
    os.makedirs("src/artifacts", exist_ok=True)
    with open("src/artifacts/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    X_train_scaled = scaler.transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test


def save_data(X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test, data_path):
    try:
        processed_data_path = os.path.join(data_path, "processed")
        logger.debug(f"Creating directory {processed_data_path}")
        
        os.makedirs(processed_data_path, exist_ok=True)
        logger.debug(f"Directory {processed_data_path} created")

        np.save(os.path.join(processed_data_path, 'X_train_scaled.npy'), X_train_scaled)
        np.save(os.path.join(processed_data_path, 'X_valid_scaled.npy'), X_valid_scaled)
        np.save(os.path.join(processed_data_path, 'X_test_scaled.npy'), X_test_scaled)
        np.save(os.path.join(processed_data_path, 'y_train.npy'), y_train)
        np.save(os.path.join(processed_data_path, 'y_valid.npy'), y_valid)
        np.save(os.path.join(processed_data_path, 'y_test.npy'), y_test)

        logger.debug(f"Data saved to {processed_data_path}")
        logger.debug(f"X_train_scaled shape: {X_train_scaled.shape}")
        logger.debug(f"X_valid_scaled shape: {X_valid_scaled.shape}")
        logger.debug(f"X_test_scaled shape: {X_test_scaled.shape}")
        logger.debug(f"y_train shape: {y_train.shape}")
        logger.debug(f"y_valid shape: {y_valid.shape}")
        logger.debug(f"y_test shape: {y_test.shape}")
    except Exception as e:
        logger.error(f"Error in save_data: {e}")
        raise e


def main():
    try:
        logger.debug("Starting data preprocessing...")
        train_df = pd.read_csv("./Data/raw/train.csv")
        test_df = pd.read_csv("./Data/raw/test.csv")
        df = preprocess_data(train_df, test_df)
        X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test = split_data(df)
        save_data(X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test, "./Data")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise e
    finally:
        logger.debug("Data preprocessing completed")


if __name__ == '__main__':
    main()
