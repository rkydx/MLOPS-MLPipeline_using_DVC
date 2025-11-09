import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging
import yaml

# Ensure the 'logs' directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

### Configure logging for logging events in data ingestion
logger = logging.getLogger('data_ingestion')        # Create a logger for data ingestion by the name 'data_ingestion', nothing related to .py file name
logger.setLevel('DEBUG')                            # Set the logging level to DEBUG, have levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
                                                    
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler(os.path.join(log_dir, 'data_ingestion.log'))
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

### Function to load data, data can be from local or remote location
def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file into a pandas DataFrame."""
    try:
        logger.info(f"Loading data from {data_url}")
        data = pd.read_csv(data_url)
        logger.info(f"Data loaded successfully with shape {data.shape}")
        return data
    except pd.errors.ParserError as e:
        logger.error(f"Parsing error while loading data: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

### Function to preprocess data, specific to the dataset being used
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data (example: handle missing values, drop columns, rename columns, remove duplicates etc.)."""
    try:
        logger.info("Starting data preprocessing")
        data.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace = True) # Drop unnecessary columns
        data.rename(columns = {'v1': 'target', 'v2': 'text'}, inplace = True)           # Rename columns for better understanding
        # data = data.dropna()                                                          # Drop rows with missing values
        # logger.info(f"Data after dropping missing values has shape {data.shape}")
        data.drop_duplicates(keep='first', inplace = True)                                # Drop duplicate rows
        logger.info("Data preprocessing completed")
        return data
    except KeyError as e:
        logger.error(f"Key error during data preprocessing: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in data preprocessing: {e}")
        raise

