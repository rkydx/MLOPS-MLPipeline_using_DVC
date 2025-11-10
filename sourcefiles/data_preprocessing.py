# pip install nltk
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os
import logging
nltk.download('punkt')
nltk.download('stopwords')


# Ensure the 'logs' directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

### Configure logging for logging events in data preprocessing
logger = logging.getLogger('data_preprocessing')        # Create a logger for data preprocessing by the name 'data_preprocessing', nothing related to .py file name
logger.setLevel('DEBUG')                                # Set the logging level to DEBUG, have levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
                                                    
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler(os.path.join(log_dir, 'data_preprocessing.log'))
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def transform_text(text):
    """
    Transform the input text by converting to lowercase, tokenizing, removing stopwords and punctuation, and stemming.
    """
    try:
        # logger.info("Starting text transformation")              # Commented to reduce log clutter for each text transformation
        # Convert to lowercase
        text = text.lower()
        
        # Tokenization - splitting text into individual words
        words = nltk.word_tokenize(text)
        
        # Remove stopwords and punctuation and non-alphabetic characters, and stemming
        ps = PorterStemmer()
        cleaned_words = []
        for word in words:
            if word not in stopwords.words('english') and word not in string.punctuation and word.isalpha():
                # Stemming
                stemmed_word = ps.stem(word)
                cleaned_words.append(stemmed_word)
        
        transformed_text = ' '.join(cleaned_words)
        # logger.info("Text transformation completed")              # Commented to reduce log clutter for each text transformation
        return transformed_text
    except Exception as e:
        logger.error(f"Error in text transformation: {e}")
        raise

def preprocess_data(data, text_column='text', target_column='target'):
    """
    Preprocess the data by transforming text column and encoding target column.
    """
    try:
        logger.info("Starting data preprocessing")

        # Transforming the specified text column
        data[text_column] = data[text_column].apply(transform_text)
        logger.info("Text column transformation applied")
        
        # Encode the target labels
        le = LabelEncoder()
        data[target_column] = le.fit_transform(data[target_column])
        logger.info("Target column label encoding applied")
        
        return data
    except KeyError as e:
        logger.error(f"Column not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in data preprocessing: {e}")
        raise    


def main(text_column='text', target_column='target'):
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.info("Raw data loaded successfully.")

        # Preprocess the data
        train_processed_data = preprocess_data(train_data, text_column, target_column)
        test_processed_data = preprocess_data(test_data, text_column, target_column)
        logger.info("Data preprocessing completed successfully.")

        # Save the processed data to data/interim
        data_path = os.path.join('./data', 'interim')
        os.makedirs(data_path, exist_ok=True)
        train_processed_data.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
        test_processed_data.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)
        logger.info(f"Processed data saved to {data_path}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Parsing error: {e}")
        raise    
    except Exception as e:
        logger.error(f"Failed to complete data preprocessing: {e}")
        raise

if __name__ == "__main__":
    main()    