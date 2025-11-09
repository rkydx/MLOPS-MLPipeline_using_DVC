import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging
import yaml

# Ensure the 'logs' directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

### Configure logging
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler(os.path.join(log_dir, 'data_ingestion.log'))
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

