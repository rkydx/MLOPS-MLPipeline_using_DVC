import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os
import logging
import pickle
import yaml

# Ensure the 'logs' directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

### Configure logging for logging events in model building
logger = logging.getLogger('model_building')        # Create a logger for model building by the name 'model_building', nothing related to .py file name
logger.setLevel('DEBUG')                                 # Set the logging level to DEBUG, have levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
                                                    
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler(os.path.join(log_dir, 'model_building.log'))
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file into a pandas DataFrame.
    
    :param file_path: Path to the CSV file
    :return: DataFrame containing the loaded data
    """
    try:
        logger.info(f"Loading data from {file_path}")
        data = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}with shape {data.shape}")
        return data
    except pd.errors.ParserError as e:
        logger.error(f"Parsing error while loading data: {e}")
        raise
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
    """ 
    Train a RandomForestClassifier model with the given training data and parameters.
    
    :param X_train: Training features
    :param y_train: Training labels
    :param params: Dictionary of hyperparameters for the RandomForestClassifier
    :return: Trained RandomForestClassifier model
    """
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Number of samples in X_train and y_train do not match.")

        logger.info('Initializing the RandomForestClassifier with parameters: {}'.format(params))
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])

        logger.info('Model training started with training data of shape: {}'.format(X_train.shape))
        clf.fit(X_train, y_train)
        logger.info('Model training completed successfully.')

        return clf
    except ValueError as ve:
        logger.error(f"Value error during model training: {ve}")
        raise
    except Exception as e:
        logger.error(f'Error during model training: {e}')
        raise

def save_model(model: RandomForestClassifier, model_path: str) -> None:
    """
    Save the trained model to a file using pickle.

    :param model: Trained RandomForestClassifier model
    :param model_path: Path to save the model file
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f'Model saved successfully at {model_path}')
    except FileNotFoundError as e:
        logger.error(f'File not found while saving model: {e}')
        raise
    except Exception as e:
        logger.error('Error saving model: {e}')
        raise

def main():
    """Main function to execute the model building process."""
    try:
        params = {
            'n_estimators': 25,
            'random_state': 2
        }
        # Load the vectorized training data
        train_data = load_data('./data/processed/train_tfidf.csv')
        X_train = train_data.drop(columns=['label']).values  # All columns except the last one, here 'label' is the last column
        y_train = train_data['label'].values                 # Only the last column, here 'label' is the last column
        # X_train = train_data.iloc[:, :-1].values           # All columns except the last one
        # y_train = train_data.iloc[:, -1].values            # Only the last column

        # Train the model
        model = train_model(X_train, y_train, params)

        # Save the trained model
        model_save_path = os.path.join('./models', 'random_forest_model.pkl')
        save_model(model, model_save_path)
    
    except Exception as e:
        logger.error(f'Error in model building process: {e}')
        raise

if __name__ == '__main__':
    main()
    
        
