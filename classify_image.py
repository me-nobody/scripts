
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

logging.basicConfig(filename=log_file, filemode='a',level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

scaler = MinMaxScaler()

IN ="/users/ad394h/Documents/nuclei_segment/data/"

OUT = "/users/ad394h/Documents/nuclei_segment/data/"

MODEL_PATH = "/users/ad394h/Documents/nuclei_segment/models/"

LBL_IMG = "predicted_image_label.tiff"
input_csv = "tumor_features.csv"

model = "adaboost_tumor.joblib"

model = os.path.join(MODEL_PATH,model)




def read_label(LBL_IMG):
    file = os.path.join(IN,LBL_IMG)
    labels = io.imread(file)
    assert labels is not None,f"image not loaded properly"
    return labels

def predict_class(input_csv):
    # get the features list
    file = os.path.join(OUT,input_csv)
    test_img_ft = pd.read_csv(file)
    # extract the image labels
    test_img_ft_labels = test_img_ft[["label"]]
    test_img_ft.drop('Label',axis=1,inplace=True)
    # scale the variables
    test_img_ft = scaler.transform(test_img_ft)
    # load the classifier
    model = load(model)
    assert model is not None,f"model not loaded properly"
    # get predictions
    predictions = model.predict(test_img_ft)
    predictions = pd.DataFrame(predictions,columns=["class"])
    test_classes = pd.merge(test_img_ft_labels,predictions,left_index=True,right_index=True)
    return test_classes

def cell_types(test_class_df):
    class_dict = {0:'normal',1:'tumor'}
    test_class_df['class'] = test_class_df['class'].map(class_dict)
    for a in df['class'].value_counts().items():
        logger.info(f"class{a[0]} has {a[1]} nuclei")
        percent_tumor =0.0
        if a[0] == 'tumor':
            percent_tumor = (int(a[1])/test_class_df.shape[0])*100
        logger.info(f"percentage of tumor cells {percent_tumor}")    


def relabel_image(class_df):
    # extract the 2 columns of the label dataframe as arrays
    label_objects = test_classes.loc[:,'Label']
    labels_class = test_classes.loc[:,'prediction']+1 # upindex the classes to remove 0 as a class
    # create a dictionary with labels as key and class as value
    label_dict ={}
    for label_,class_ in zip(label_objects,labels_class):
        label_dict[label_]=class_
    # create a new array of the same dimensions as label image
    new_class_labels = np.zeros_like(labels)
    # the label image and its copy are 2-D. flatten them to reduce search space while re-assigning the array
    flat_labels=labels.flatten()
    flat_new_labels = new_class_labels.flatten()
    # assign the class in dictionary to the label in the label image. here idx is the actual value in the flattened
    # label image and count is the position of that value
    for count,idx in enumerate(flat_labels):
        if idx > 0 and idx in label_dict.keys():
            flat_new_labels[count] = label_dict[idx]      # set the value of new labels as the value of label dictionary
    # reshape the new label image to the original shape
    new_labels =flat_new_labels.reshape(labels.shape)   
    return new_labels

if __name__ == "__main__":
    label_image = read_label(LBL_IMG)
    test_class_df = predict_class(input_csv)
    new_label_image = relabel_image(test_class_df)
    cell_types(test_class_df)
    io.imsave(fname=os.path.join(OUT,"classfied_image.png"),arr=new_label_image)
