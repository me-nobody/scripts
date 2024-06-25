# import libraries
from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import os
import time
import glob
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from skimage import io
import joblib

timestr = time.strftime("%Y%m%d-%H%M%S")

log_file = '/users/ad394h/Documents/nuclei_segment/logs/classify_nuclei_{}.txt'.format(timestr)

logging.basicConfig(filename=log_file, level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

scaler = MinMaxScaler()


DATA_PATH ="/users/ad394h/Documents/nuclei_segment/data/training_data/"

MODEL_PATH ="/users/ad394h/Documents/nuclei_segment/models/"



scaler = MinMaxScaler()


training_data = "final_GFP_train_df.csv"

def prepare_data(training_data):
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
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # save the train and test dataframes
    xtrain_df = pd.DataFrame(X_train)
    xtest_df = pd.DataFrame(X_test)
    xtrain_df.columns = input_cols
    xtest_df.columns = input_cols
    xtest_df['target'] = test_targets
    xtrain_df.to_csv(os.path.join(DATA_PATH,"X_train_{}.csv".format(timestr)),index=False)
    xtest_df.to_csv(os.path.join(DATA_PATH,"X_test_{}.csv".format(timestr)),index=False)
    logger.info(f"train data shape {xtrain_df.shape[0]},{xtrain_df.shape[1]}")
    logger.info(f"test data shape {xtest_df.shape[0]},{xtest_df.shape[1]}")
    return X_train, X_test,train_targets,test_targets,scaler,xtrain_df



def predict_with_saved_model(X_test,test_targets,scaler):
    # predict class has to identify which image it is predicting for
    logger.info("predicting class")
    # get the features list
    saved_model_path = os.path.join(MODEL_PATH,"adaboost_classify_nuclei.joblib")
    with open(saved_model_path, "rb") as f_model:
        clf = joblib.load(f_model)      

    # get feature importance
    
    feature_importance = clf.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    fig = plt.figure(figsize=(22, 16))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), x_train_df.columns)
    plt.title('Feature Importance')
    plt.savefig(os.path.join(MODEL_PATH,"feature_importances.png"))
    
    # accuracy
    test_accuracy = clf.score(X_test, test_targets)
    # F-score
    predicted_test_targets = clf.predict(X_test)
    test_f1 = f1_score(test_targets,predicted_test_targets, average='macro')
    
    logger.info(f"test accuracy with saved model {test_accuracy}")
    logger.info(f"test F1 score with saved model {test_f1}")



if __name__ == "__main__":
    logger.info("start train classifier and classify nuclei")
    logger.info("preparing data ....")
    
    # get the feature label pair list
    X_train, X_test,train_targets,test_targets,scaler,x_train_df = prepare_data(training_data=training_data) 
    predict_with_saved_model(X_test,test_targets,scaler)   
    
    
