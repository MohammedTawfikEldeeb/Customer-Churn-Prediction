import os
import pandas as pd
from sklearn.utils import shuffle
import logging

logger = logging.getLogger("data_loading")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

training_data_path = 'Data\customer_churn_dataset-training-master.csv'
testing_data_path = 'Data\customer_churn_dataset-testing-master.csv'


def load_data(data_path):
    try:
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        logger.debug("Data loaded from %s" , data_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {data_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"Empty data file: {data_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {e}")
        raise

def preprocess_data(df: pd.DataFrame):
    try:
        logger.info("Preprocessing data")
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df.drop(columns=["CustomerID"] , inplace=True)
        logger.info("Data preprocessed successfully")
        return df
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise
def save_data(train_data: pd.DataFrame , test_data: pd.DataFrame , data_path: str):
    try:
        logger.info(f"Saving data to {data_path}")
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)

        train_data_path = os.path.join(raw_data_path, 'train.csv')
        test_data_path = os.path.join(raw_data_path, 'test.csv')

        train_data.to_csv(train_data_path, index=False)
        test_data.to_csv(test_data_path, index=False)   

        logger.info(f"Data saved successfully. Train shape: {train_data.shape}, Test shape: {test_data.shape}")
        logger.info(f"Train data saved to {train_data_path}")
        logger.info(f"Test data saved to {test_data_path}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

def main():
    try:
        data_path = 'Data'
        df_train = load_data(training_data_path)
        df_test = load_data(testing_data_path)


        df_train = preprocess_data(df_train)
        df_test = preprocess_data(df_test)

        save_data(df_train, df_test, data_path)
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == '__main__':
    main()

