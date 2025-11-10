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

