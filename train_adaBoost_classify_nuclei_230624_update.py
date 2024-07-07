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


FEATURE_PATH = "/users/ad394h/Documents/nuclei_segment/data/val_he_images_features/"

LABEL_PATH = "/users/ad394h/Documents/nuclei_segment/data/val_he_image_labels/"


input_csv = "nuclei_features.csv"

training_data = "final_GFP_train_df.csv"

scaler = MinMaxScaler()


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
    return X_train, X_test,train_targets,test_targets,scaler
    
def train_data(*vars):
    X_train, X_test,train_targets,test_targets = vars
    from sklearn.ensemble import AdaBoostClassifier
    model = AdaBoostClassifier(n_estimators=100, algorithm="SAMME", random_state=0)
    model_path = os.path.join(MODEL_PATH,"adaboost_classify_nuclei_pretrained.joblib")
    with open(model_path,"wb") as f_pre:
        joblib.dump(model,f_pre,protocol=5)
    model.fit(X_train, train_targets)
    # save the model
    model_path = os.path.join(MODEL_PATH,"adaboost_classify_nuclei.joblib")
    with open(model_path,"wb") as f:
        joblib.dump(model,f,protocol=5)
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

    return model


def read_label(LBL_IMG):
    logger.info("reading file")
    img_file = os.path.join(IN,LBL_IMG)
    labels = io.imread(img_file)
    return labels

# function to pair features df and label image
def create_feature_label_pair(FEATURE_PATH,LABEL_PATH):
    feature_file_list = os.listdir(FEATURE_PATH)
    label_file_list = os.listdir(LABEL_PATH)
    logger.info(f"number of feature files {len(feature_file_list)}")
    logger.info(f"number of label files {len(label_file_list)}")
    assert len(feature_file_list)>0,"no files present"
    assert len(label_file_list)>0,"no files present"
    if all(feature[-4:] == ".csv" for feature in feature_file_list):
        logger.info("all csv files in folder")
    if all(img[-4:] == ".png" for img in label_file_list):
        logger.info("all png files in folder")
    # matched image and label
    feature_label_pair_list = [(a,b) for a in feature_file_list for b in label_file_list if a[:21] == b[:21]]
    logger.info(f"{len(feature_label_pair_list)} feature_label_pairs_created")    
    return feature_label_pair_list    


def predict_class(input_csv=None,scaler=None,model=None):
    # predict class has to identify which image it is predicting for
    logger.info("predicting class")
    # get the features list
    csv_file = os.path.join(OUT,input_csv)
    csv_file_name = input_csv[:-4]
    if os.path.isfile(csv_file):
        logger.info(f"input csv file read")
    test_img_ft = pd.read_csv(csv_file)
    # extract the image labels
    test_img_ft_labels = test_img_ft[["Label"]]
    test_img_ft.drop('Label',axis=1,inplace=True)
    test_img_ft = scaler.transform(test_img_ft)
    
    # # get predictions
    predictions = model.predict(test_img_ft)
    predictions = pd.DataFrame(predictions,columns=["class"])
    test_classes = pd.merge(test_img_ft_labels,predictions,left_index=True,right_index=True)
    test_classes.to_csv(os.path.join(OUT,"test_classes.csv"),header=True,index=False)
    logger.info(f"columns are {test_classes.columns}")
    return test_class_df

def predict_with_saved_model(X_test,test_targets,scaler):
    # predict class has to identify which image it is predicting for
    logger.info("predicting class")
    # get the features list
    saved_model_path = os.path.join(MODEL_PATH,"adaboost_classify_nuclei.joblib")
    with open(saved_model_path, "rb") as f_model:
        clf = joblib.load(f_model)      
    # accuracy
    test_accuracy = clf.score(X_test, test_targets)
    # F-score
    predicted_test_targets = clf.predict(X_test)
    test_f1 = f1_score(test_targets,predicted_test_targets, average='macro')
    
    logger.info(f"test accuracy with saved model {test_accuracy}")
    logger.info(f"test F1 score with saved model {test_f1}")



def cell_types(test_class_df):
    class_dict = {0:'normal',1:'tumor'}
    test_class_df['class'] = test_class_df['class'].map(class_dict)
    for a in test_class_df['class'].value_counts().items():
        logger.info(f"class {a[0]} has {a[1]} nuclei")
        percent_tumor =0.0
        if a[0] == 'tumor':
            percent_tumor = (int(a[1])/test_class_df.shape[0])*100
            logger.info(f"percentage of tumor cells {percent_tumor}")    



def relabel_image(class_df,label_image):
    # extract the 2 columns of the label dataframe as arrays
    label_objects = class_df.loc[:,'Label']
    labels_class = class_df.loc[:,'class']+1 # upindex the classes to remove 0 as a class
    # create a dictionary with labels as key and class as value
    label_dict ={}
    for label_,class_ in zip(label_objects,labels_class):
        label_dict[label_]=class_
    # create a new array of the same dimensions as label image
    new_class_labels = np.zeros_like(label_image)
    # the label image and its copy are 2-D. flatten them to reduce search space while re-assigning the array
    flat_labels=label_image.flatten()
    flat_new_labels = new_class_labels.flatten()
    # assign the class in dictionary to the label in the label image. here idx is the actual value in the flattened
    # label image and count is the position of that value
    for count,idx in enumerate(flat_labels):
        if idx > 0 and idx in label_dict.keys():
            flat_new_labels[count] = label_dict[idx]      # set the value of new labels as the value of label dictionary
    # reshape the new label image to the original shape
    new_labels =flat_new_labels.reshape(label_image.shape)   
    return new_labels



if __name__ == "__main__":
    logger.info("start train classifier and classify nuclei")
    logger.info("preparing data ....")
    # create_feature_label_pair(FEATURE_PATH,LABEL_PATH)
    X_train, X_test,train_targets,test_targets,scaler = prepare_data(training_data=training_data)    
    logger.info("training data ...")
    # model = train_data(X_train, X_test,train_targets,test_targets)
    try:
        model_path = os.path.join(MODEL_PATH,"adaboost_classify_nuclei.joblib")
    except FileNotFoundError:
        print("model file not found")

    if os.path.exists(model_path) and isinstance(model_path,AdaBoostClassifier):
        logger.info(f"adaboost classifier obtained")
    else:
        logger.info("please check model is AdaBoostClassifier")    
    predict_with_saved_model(X_test,test_targets,scaler)    
    # try:
    #     os.path.join(OUT,LBL_IMG)
    #     logger.info(f"{LBL_IMG} image file recongnized")
    # except AssertionError as err:
    #     logger.info(f"label image not found")
    # label_image = read_label(LBL_IMG)
    # assert input_csv is not None,logger.info(f"input csv not found")
    # test_class_df = predict_class(input_csv=input_csv,scaler=scaler,model=model)
    # new_label_image = relabel_image(test_class_df,label_image)
    # assert new_label_image is not None,logger.info(f"relablled image missing")    
    # io.imsave(fname=os.path.join(OUT,f"{IMG[:-4]}_classified_image.png"),arr=new_label_image)
    # try:
    #     isinstance(test_class_df,pd.DataFrame)
    #     logger.info(f"test dataframe detected")
    # except AssertionError as err:
    #     logger.info(f"test dataframe not found")    
    # cell_types(test_class_df)
    
    
