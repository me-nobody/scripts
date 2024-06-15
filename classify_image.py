
# import libraries
from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import os
import time
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.ensemble import AdaBoostClassifier


from skimage import io
import joblib
import pickle
from joblib import load

timestr = time.strftime("%Y%m%d-%H%M%S")

log_file = '/users/ad394h/Documents/nuclei_segment/logs/classify_image_{}.txt'.format(timestr)

logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

scaler = MinMaxScaler()

IN ="/users/ad394h/Documents/nuclei_segment/data/"

OUT = "/users/ad394h/Documents/nuclei_segment/data/"

MODEL_PATH = "/users/ad394h/Documents/nuclei_segment/models/"

x_train = "X_train.csv"

LBL_IMG = "predicted_image_label.tiff"
input_csv = "tumor_features.csv"

model = "adaboost_tumor_pkl.pkl"

model = os.path.join(MODEL_PATH,model)


def read_label(LBL_IMG):
    logger.info("reading file")
    img_file = os.path.join(IN,LBL_IMG)
    labels = io.imread(img_file)
    return labels

def predict_class(input_csv,model):
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
    train = os.path.join(OUT,x_train)
    train = pd.read_csv(train)
    train.columns = test_img_ft.columns
    if isinstance(train,pd.DataFrame):
        logger.info(f"scaling train csv file read")
    train = scaler.fit_transform(train)
    test_img_ft = scaler.transform(test_img_ft)
    # load the classifier
    with open(model,'rb') as f:
        model = pickle.load(f)
    assert model is not None,f"model not loaded properly"
    # get predictions
    predictions = model.predict(test_img_ft)
    predictions = pd.DataFrame(predictions,columns=["class"])
    test_classes = pd.merge(test_img_ft_labels,predictions,left_index=True,right_index=True)
    test_classes.to_csv(os.path.join(OUT,"test_classes.csv"))
    logger.info(f"columns are {test_classes.columns}")
    return test_classes

def cell_types(test_class_df):
    class_dict = {0:'normal',1:'tumor'}
    test_class_df['class'] = test_class_df['class'].map(class_dict)
    for a in test_class_df['class'].value_counts().items():
        logger.info(f"class{a[0]} has {a[1]} nuclei")
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
    logger.info("start")
    try:
        os.path.join(OUT,LBL_IMG)
        logger.info(f"{LBL_IMG} image file recongnized")
    except AssertionError as err:
        logger.info(f"label image not found")
    label_image = read_label(LBL_IMG)
    assert input_csv is not None,logger.info(f"input csv not found")
    test_class_df = predict_class(input_csv,model)
    new_label_image = relabel_image(test_class_df,label_image)
    assert new_label_image is not None,logger.info(f"relablled image missing")    
    io.imsave(fname=os.path.join(OUT,"classfied_image.png"),arr=new_label_image)
    try:
        isinstance(test_class_df,pd.DataFrame)
        logger.info(f"test dataframe detected")
    except AssertionError as err:
        logger.info(f"test dataframe not found")    
    cell_types(test_class_df)
    
