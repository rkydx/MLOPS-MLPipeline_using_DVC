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
logger.setLevel('DEBUG')                                # Set the logging level to DEBUG, have levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
                                                    
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler(os.path.join(log_dir, 'feature_engineering.log'))
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)