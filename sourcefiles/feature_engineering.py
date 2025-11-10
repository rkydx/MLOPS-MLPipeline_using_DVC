import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import yaml

# Ensure the 'logs' directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

### Configure logging for logging events in feature engineering
logger = logging.getLogger('feature_engineering')        # Create a logger for feature engineering by the name 'feature_engineering', nothing related to .py file name
logger.setLevel('DEBUG')                                 # Set the logging level to DEBUG, have levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
                                                    
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler(os.path.join(log_dir, 'feature_engineering.log'))
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file into a pandas DataFrame."""
    try:
        logger.info(f"Loading data from {file_path}")
        data = pd.read_csv(file_path)
        data.fillna('', inplace=True)                      # Fill missing values with empty strings
        logger.info(f"Data loaded successfully from {file_path} and NaNs filled with empty strings, shape: {data.shape}")
        return data
    except pd.errors.ParserError as e:
        logger.error(f"Parsing error while loading data: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def apply_tfidf_vectorization(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """Apply TF-IDF vectorization to the data."""
    try:
        logger.info("Starting TF-IDF vectorization")
        vectorizer = TfidfVectorizer(max_features=max_features)

        X_train = train_data['text'].values
        y_train = train_data['target'].values
        X_test = test_data['text'].values
        y_test = test_data['target'].values

        # Fit the vectorizer on the training data and transform both train and test data
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Convert the sparse matrices to DataFrames for easier handling and add target labels back
        train_tfidf_df = pd.DataFrame(X_train_tfidf.toarray())
        train_tfidf_df['label'] = y_train

        test_tfidf_df = pd.DataFrame(X_test_tfidf.toarray())
        test_tfidf_df['label'] = y_test

        logger.info("TF-IDF vectorization completed successfully")
        return train_tfidf_df, test_tfidf_df
    except Exception as e:
        logger.error(f"Error during TF-IDF vectorization: {e}")
        raise    

def save_vectorized_data(data: pd.DataFrame, file_path: str) -> None:
    """Save the vectorized data to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data.to_csv(file_path, index=False)
        logger.info(f"Vectorized data saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error saving vectorized data: {e}")
        raise

def main():
    """Main function to execute the feature engineering process."""
    try:
        max_features = 50

        # Load the preprocessed data
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        # Apply TF-IDF vectorization
        train_vectorized, test_vectorized = apply_tfidf_vectorization(train_data, test_data, max_features)

        # Save the vectorized data
        save_vectorized_data(train_vectorized, os.path.join("./data", "processed", "train_tfidf.csv"))
        save_vectorized_data(test_vectorized, os.path.join("./data", "processed", "test_tfidf.csv"))
    except Exception as e:
        logger.error(f"Error in feature engineering process: {e}")
        raise

if __name__ == '__main__':
    main()    
    
