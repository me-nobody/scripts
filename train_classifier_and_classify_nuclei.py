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

log_file = '/users/ad394h/Documents/nuclei_segment/logs/classify_nuclei.txt'

logging.basicConfig(filename=log_file, level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

scaler = MinMaxScaler()

IN ="/users/ad394h/Documents/nuclei_segment/data/"

OUT = "/users/ad394h/Documents/nuclei_segment/data/"

DATA_PATH ="/users/ad394h/Documents/nuclei_segment/data/"

MODEL_PATH = "/users/ad394h/Documents/nuclei_segment/models/"

IMG = glob.glob("/users/ad394h/Documents/nuclei_segment/data/*.jpg")[0]

LBL_IMG = f"{IMG[:-4]}_predicted_image_label.tiff"
input_csv = "nuclei_features.csv"


tumor_csv = "image3_45_16_tumor.csv"
normal_csv = "image3_7_13_normal.csv"

scaler = MinMaxScaler()


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
    combined = pd.concat([tumor_df,normal_df],axis=0)
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
    # save the train and test dataframes
    xtrain_df = pd.DataFrame(X_train)
    xtest_df = pd.DataFrame(X_test)
    xtrain_df.columns = input_cols
    xtest_df.columns = input_cols
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)    
    # xtrain_df.to_csv(os.path.join(DATA_PATH,"X_train.csv"),index=False)
    # xtest_df.to_csv(os.path.join(DATA_PATH,"X_test.csv"),index=False)
    return X_train, X_test,train_targets,test_targets,scaler
    
def train_data(*vars):
    X_train, X_test,train_targets,test_targets = vars
    from sklearn.ensemble import AdaBoostClassifier
    model = AdaBoostClassifier(n_estimators=100, algorithm="SAMME", random_state=0)
    model.fit(X_train, train_targets)
    # save the model
    joblib.dump(model,os.path.join(MODEL_PATH,"adaboost_tumor.joblib"))
    # get the scores
    # accuracy
    train_accuracy = model.score(X_train, train_targets)
    test_accuracy = model.score(X_test, test_targets)
    # F-score
    predicted_train_targets = model.predict(X_train)
    predicted_test_targets = model.predict(X_test)
    train_f1 = f1_score(train_targets,predicted_train_targets, average='macro')
    test_f1 = f1_score(test_targets,predicted_test_targets, average='macro')
    return model


def read_label(LBL_IMG):
    logger.info("reading file")
    img_file = os.path.join(IN,LBL_IMG)
    labels = io.imread(img_file)
    return labels

def predict_class(input_csv=None,scaler=None,model=None):
    logger.info("predicting class")
    # get the features list
    csv_file = os.path.join(OUT,input_csv)
    if os.path.isfile(csv_file):
        logger.info(f"input csv file read")
    test_img_ft = pd.read_csv(csv_file)
    # extract the image labels
    test_img_ft_labels = test_img_ft[["Label"]]
    test_img_ft.drop('Label',axis=1,inplace=True)
    # scale the variables
    # train = os.path.join(OUT,x_train)
    # train = pd.read_csv(train)
    # train.columns = test_img_ft.columns
    # if isinstance(train,pd.DataFrame):
    #     logger.info(f"scaling train csv file read")
    # train = scaler.fit(train)
    test_img_ft = scaler.transform(test_img_ft)
    # load the classifier
    # model = joblib.load(os.path.join(MODEL_PATH,model))
    # assert model is not None,f"model not loaded properly"
    # # get predictions
    predictions = model.predict(test_img_ft)
    predictions = pd.DataFrame(predictions,columns=["class"])
    test_classes = pd.merge(test_img_ft_labels,predictions,left_index=True,right_index=True)
    test_classes.to_csv(os.path.join(OUT,"test_classes.csv"),header=True,index=False)
    logger.info(f"columns are {test_classes.columns}")
    return test_classes

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
    X_train, X_test,train_targets,test_targets,scaler = prepare_data(tumor_csv,normal_csv)    
    logger.info("training data ...")
    model = train_data(X_train, X_test,train_targets,test_targets)
    try:
        os.path.join(OUT,LBL_IMG)
        logger.info(f"{LBL_IMG} image file recongnized")
    except AssertionError as err:
        logger.info(f"label image not found")
    label_image = read_label(LBL_IMG)
    assert input_csv is not None,logger.info(f"input csv not found")
    test_class_df = predict_class(input_csv=input_csv,scaler=scaler,model=model)
    new_label_image = relabel_image(test_class_df,label_image)
    assert new_label_image is not None,logger.info(f"relablled image missing")    
    io.imsave(fname=os.path.join(OUT,f"{IMG[:-4]}_classified_image.png"),arr=new_label_image)
    try:
        isinstance(test_class_df,pd.DataFrame)
        logger.info(f"test dataframe detected")
    except AssertionError as err:
        logger.info(f"test dataframe not found")    
    cell_types(test_class_df)
    
