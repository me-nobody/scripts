#!/usr/bin/python

# import libraries
import sys
import os
import time
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.model_selection import train_test_split


import joblib

timestr = time.strftime("%Y%m%d-%H%M%S")

log_file = '/users/ad394h/Documents/nuclei_segment/logs/train_minmax_scaler_{}.txt'.format(timestr)

logging.basicConfig(filename=log_file, level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

scaler = MinMaxScaler()


DATA_PATH ="/users/ad394h/Documents/nuclei_segment/training_data/"

MODEL_PATH ="/users/ad394h/Documents/nuclei_segment/models/"



training_data = "final_GFP_train_df.csv"

scaler = MinMaxScaler()


def train_scaler(training_data):
    # training data contains the target column
    training_data = os.path.join(DATA_PATH,training_data)
    if os.path.isfile(training_data):
        logger.info("training data file accessed")

    # create dataframes from the file
    train_df = pd.read_csv(training_data)          
    
    # Identify input and target columns
    input_cols, target_col = train_df.columns[:-1], train_df.columns[-1]
    input_df, input_targets = train_df[input_cols].copy(), train_df[target_col].copy()
    # target column is encoded as 'tumor':1,'normal':0
    
    # Create training and validation sets
    X_train, X_test, train_targets, test_targets = train_test_split(input_df, input_targets, test_size=0.25, random_state=42)
    # Impute and scale numeric columns
    scaler = MinMaxScaler()
    scaler_path = os.path.join(MODEL_PATH,"scaler_pretrained.joblib")
    with open(scaler_path,"wb") as f_pre:
        joblib.dump(scaler,f_pre,protocol=5)

    scaler = scaler.fit(X_train)
    scaler_path = os.path.join(MODEL_PATH,"scaler_fitted.joblib")
    with open(scaler_path,"wb") as f:
        joblib.dump(scaler,f,protocol=5)
    
    
def test_scaler(training_data):
    # get the saved scaler
    scaler_path = os.path.join(MODEL_PATH,"scaler_fitted.joblib")
    with open(scaler_path,"rb") as f:
        scaler = joblib.load(f)    
    training_data = os.path.join(DATA_PATH,training_data)
    train_df = pd.read_csv(training_data)
    sample_data = train_df.sample(n=200)
    sample_data = sample_data[train_df.columns[:-1]]
    transformed_data = scaler.transform(sample_data)
    transformed_data = pd.DataFrame(transformed_data)
    transformed_data.columns = sample_data.columns
    sample_data = sample_data.iloc[:,1:5]
    transformed_data = transformed_data.iloc[:,1:5]
    # some stats to check the transformation
    max_sample_data = sample_data.max()
    max_transformed_data = transformed_data.max()
    logger.info(f"sample data {max_sample_data}")
    logger.info(f"transformed data {max_transformed_data}")



if __name__ == "__main__":
    # train_scaler(training_data)
    test_scaler(training_data)
    