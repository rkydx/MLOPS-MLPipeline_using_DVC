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


### Function to load configuration/parameters from a YAML file
def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file.
    
    :param config_path: Path to the YAML configuraton file
    :return: Dictionary containing the configuration parameters
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f'Configuration parameters loaded from {config_path}')
        return config
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        raise
    except yaml.YAMLErrora as e:
        logger.error(f"Error parsing configuration file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

### Function to load data, data can be from local or remote location
def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file into a pandas DataFrame.
    
    :param data_url: URL or local path to the CSV file
    :return: DataFrame containing the loaded data
    """
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
def cleaning_data(data: pd.DataFrame) -> pd.DataFrame:
    """Cleaning the data (example: handle missing values, drop columns, rename columns, remove duplicates etc.).
    
    :param data: DataFrame containing the raw data
    :return: DataFrame containing the cleaned data
    """
    try:
        logger.info("Starting data cleaning process")
        data.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace = True) # Drop unnecessary columns
        data.rename(columns = {'v1': 'target', 'v2': 'text'}, inplace = True)           # Rename columns for better understanding
        # data = data.dropna()                                                          # Drop rows with missing values
        # logger.info(f"Data after dropping missing values has shape {data.shape}")
        data.drop_duplicates(keep='first', inplace = True)                                # Drop duplicate rows
        logger.info("Data cleaning process completed - dropped unnecessary columns and duplicates and renamed columns")
        return data
    except KeyError as e:
        logger.error(f"Key error during data cleaning: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in data cleaning: {e}")
        raise

### Function to split data into train and test sets
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test data to specified paths.
    
    :param train_data: DataFrame containing the training data
    :param test_data: DataFrame containing the testing data
    :param data_path: Path to save the train and test data
    :return: None
    """
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)      
        logger.info(f"Train and test data saved to {raw_data_path}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

### Main function to execute the data ingestion pipeline
def main():
    try:
        params = load_config(config_path="params.yaml")
        test_size = params['data_ingestion']['test_size']
        #test_size = 0.2
        random_state = 2                # Random state 2 means every time you run the code, you will get the same split of data into train and test sets
        data_path = "https://raw.githubusercontent.com/rkydx/datasets_repo/refs/heads/main/spam.csv"

        # Load data
        data = load_data(data_path)
        # Cleaning the data
        final_data = cleaning_data(data)
        # Split data into train and test sets
        train_data, test_data = train_test_split(final_data, test_size=test_size, random_state=random_state)
        # Save the train and test data
        save_data(train_data, test_data, data_path="./data")
    except Exception as e:
        logger.critical(f"Critical error in data ingestion pipeline: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()


