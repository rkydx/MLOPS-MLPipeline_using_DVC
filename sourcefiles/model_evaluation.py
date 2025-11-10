import os
import pandas as pd
import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import pickle
import json
import yaml

# Ensure the 'logs' directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

### Configure logging for logging events in model evaluation
logger = logging.getLogger('model_evaluation')        # Create a logger for model evaluation by the name 'model_evaluation', nothing related to .py file name
logger.setLevel('DEBUG')                              # Set the logging level to DEBUG, have levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
                                                    
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler(os.path.join(log_dir, 'model_evaluation.log'))
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(model_path: str):
    """Load the trained model from a file.
    
    :param model_path: Path to the model file
    :return: Loaded model object
    """
    try:
        logger.info(f'Loading model from {model_path}')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info('Model loaded successfully')
        return model
    except FileNotFoundError as e:
        logger.error(f'Model file not found: {e}')
        raise
    except Exception as e:
        logger.error(f'Error loading model: {e}')
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a processed CSV file.

    :param file_path: Path to the CSV file
    :return: DataFrame containing the loaded data
    """
    try:
        data = pd.read_csv(file_path)
        logger.info(f'Data loaded successfully from {file_path}')
        return data
    except pd.errors.ParserError as e:
        logger.error(f'Parsing error while loading data: {e}')
        raise
    except FileNotFoundError as e:
        logger.error(f'File not found: {e}')
        raise
    except Exception as e:
        logger.error(f'Error loading data: {e}')
        raise

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate the model on the test data and compute various metrics.
    
    :param model: Trained model object
    :param X_test: Test features
    :param y_test: Test labels
    :return: Dictionary containing evaluation metrics
    """
    try:
        logger.info('Starting model evaluation')
        y_pred = model.predict(X_test)                          # Predicted labels
        y_pred_proba = model.predict_proba(X_test)[:, 1]        # For ROC-AUC calculation, means probability estimates of the positive class

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        #f1 = f1_score(y_test, y_pred)
        #conf_matrix = confusion_matrix(y_test, y_pred)    

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'roc_auc': auc,
            # 'f1': f1,
            # 'confusion_matrix': conf_matrix
        }

        logger.info('Model evaluation metrics computed successfully')
        return metrics
    except Exception as e:
        logger.error(f'Error during model evaluation: {e}')
        raise   

def save_evaluation_report(metrics: dict, report_path: str) -> None:
    """
    Save the evaluation metrics to a JSON file.
    
    :param metrics: Dictionary containing evaluation metrics
    :param report_path: Path to save the evaluation report
    :return: None
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f'Evaluation report saved successfully at {report_path}')
    except Exception as e:
        logger.error(f'Error saving evaluation report: {e}')
        raise

def main():
    """
    Main function to execute the model evaluation process.
    """
    try:
        # Load the trained model
        model = load_model('./models/random_forest_model.pkl')

        # Load the vectorized test data
        test_data = load_data('./data/processed/test_tfidf.csv')

        X_test = test_data.drop(columns=['label']).values  # All columns except the last one, here 'label' is the last column
        y_test = test_data['label'].values                 # Only the last column, here 'label' is the last column
        # X_test = test_data.iloc[:, :-1].values           # All columns except the last one
        # y_test = test_data.iloc[:, -1].values            # Only the last column   

        # Evaluate the model
        metrics = evaluate_model(model, X_test, y_test)

        # Save the evaluation report
        save_evaluation_report(metrics, './reports/metrics_evaluation_report.json')
    except Exception as e:
        logger.error(f'Error in main: {e}')
        raise

if __name__ == '__main__':
    main()