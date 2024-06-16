# import libraries
import os
import time
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
from skimage import io
import joblib
import pickle

timestr = time.strftime("%Y%m%d-%H%M%S")

log_file = '/users/ad394h/Documents/nuclei_segment/logs/train_classifier_{}.txt'.format(timestr)

logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

scaler = MinMaxScaler()

DATA_PATH ="/users/ad394h/Documents/nuclei_segment/data/"

MODEL_PATH = "/users/ad394h/Documents/nuclei_segment/models/"

tumor_csv = "image3_45_16_tumor.csv"
normal_csv = "image3_7_13_normal.csv"

def prepare_data(*files):
    for file in files:
        # check tumor file
        if "tumor" in file:            
            tumor_file = os.path.join(DATA_PATH,file)
            if os.path.isfile(tumor_file):
                logger.info(f"tumor csv file read {file}")
        # check normal file
        elif "normal" in file:
            normal_file = os.path.join(DATA_PATH,file)
            if os.path.isfile(tumor_file):
                logger.info(f"normal csv file read {file}")
        else:
            logger.info("not the desired file")      
    # create dataframes from the file
    tumor_df = pd.read_csv(tumor_file)
    normal_df = pd.read_csv(normal_file)          
    # add the target label
    tumor_df[['type']] = 'tumor'
    normal_df[['type']] = 'normal'
    # create combined dataframe
    combined = pd.concat([tumor,normal],axis=0)
    # remove Label column from dataset for further machine learning
    combined = combined.drop('Label',axis=1)
    # Identify input and target columns
    input_cols, target_col = combined.columns[:-1], combined.columns[-1]
    input_df, input_targets = combined[input_cols].copy(), combined[target_col].copy()
    # we will one-hot encode the target column
    input_targets_num = input_targets.map({'tumor':1,'normal':0})
    # Create training and validation sets
    X_train, X_test, train_targets, test_targets = train_test_split(input_df, input_targets_num, test_size=0.25, random_state=42)
    # Impute and scale numeric columns
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # save the train and test dataframes
    xtrain_df = pd.DataFrame(X_train)
    xtest_df = pd.DataFrame(X_test)
    xtrain_df.columns = input_cols
    xtest_df.columns = input_cols
    xtrain_df.to_csv(os.path.join(DATA_PATH,"X_train.csv"),index=False)
    xtest_df.to_csv(os.path.join(DATA_PATH,"X_test.csv"),index=False)
    return X_train, X_test,train_targets,test_targets
    
def train_data(*vars):
    X_train, X_test,train_targets,test_targets = vars
    from sklearn.ensemble import AdaBoostClassifier
    model = AdaBoostClassifier(n_estimators=100, algorithm="SAMME", random_state=0)
    model.fit(X_train, train_targets)
    # get the scores
    # accuracy
    train_accuracy = model.score(X_train, train_targets)
    test_accuracy = model.score(X_test, test_targets)
    # F-score
    predicted_train_targets = model.predict(X_train)
    predicted_test_targets = model.predict(X_test)
    train_f1 = f1_score(train_targets,predicted_train_targets, average='macro')
    test_f1 = f1_score(test_targets,predicted_test_targets, average='macro')



    logger.info(f"train accuracy {train_accuracy}")
    logger.info(f"test accuracy {test_accuracy}")
    logger.info(f"train F1 score {train_f1}")
    logger.info(f"test F1 score {test_f1}")


if __name__ == "__main__":
    logger.info("preparing data ....")
    X_train, X_test,train_targets,test_targets = prepare_data(tumor_csv,normal_csv)    
    logger.info("training data ...")
    train_data(X_train, X_test,train_targets,test_targets)
  