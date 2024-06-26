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

from multiprocessing import Pool


import seaborn as sns
import matplotlib.pyplot as plt
from skimage import io
import joblib

timestr = time.strftime("%Y%m%d-%H%M%S")

log_file = '/users/ad394h/Documents/nuclei_segment/logs/classify_nuclei_{}.txt'.format(timestr)

logging.basicConfig(filename=log_file, level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

scaler = MinMaxScaler()


DATA_PATH ="/users/ad394h/Documents/nuclei_segment/data/training_data/"

MODEL_PATH ="/users/ad394h/Documents/nuclei_segment/models/"


FEATURE_PATH = "/users/ad394h/Documents/nuclei_segment/data/karin_he_images_features/"

LABEL_PATH = "/users/ad394h/Documents/nuclei_segment/data/karin_he_image_labels/"

PREDICT_PATH = "/users/ad394h/Documents/nuclei_segment/data/karin_he_images_predictions/"

CLASSIFIED_IMG_PATH = "/users/ad394h/Documents/nuclei_segment/data/karin_he_classified_images/"


training_data = "final_GFP_train_df.csv"

scaler = MinMaxScaler()



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
    feature_label_pair_list = [(a,b) for a in feature_file_list for b in label_file_list if a[:-20] == b[:-21]]
    logger.info(f"{len(feature_label_pair_list)} feature_label_pairs_created")    
    return feature_label_pair_list    


def predict_class(feature_pair):
    features,label = feature_pair
    # get the features list
    csv_file_name = features[:-20]+"_predicted_classes.csv"
    feature_file = os.path.join(FEATURE_PATH,features)
    
    if os.path.isfile(feature_file):
        logger.info(f"input csv file read")
    test_img_ft = pd.read_csv(feature_file)
    
    
    # load the scaler
    scaler_path = os.path.join(MODEL_PATH,"scaler_fitted.joblib")
    with open(scaler_path,"rb") as f:
        scaler = joblib.load(f)
    # load the model 
    model_path = os.path.join(MODEL_PATH,"adaboost_classify_nuclei.joblib")
    with open(model_path, "rb") as f_model:
        model = joblib.load(f_model)

    # extract the image labels
    test_img_ft_labels = test_img_ft[["Label"]]
    test_img_ft.drop('Label',axis=1,inplace=True)
    col_names = test_img_ft.columns
   
    # scale the data
    test_img_ft = scaler.transform(test_img_ft)
    test_img_ft = pd.DataFrame(test_img_ft)
    test_img_ft.columns = col_names

    # get predictions
    predictions = model.predict(test_img_ft)
    predictions = pd.DataFrame(predictions,columns=["class"])
    test_class_df = pd.merge(test_img_ft_labels,predictions,left_index=True,right_index=True)
    test_class_df.to_csv(os.path.join(PREDICT_PATH,csv_file_name),header=True,index=False)
    logger.info(f"columns are {test_class_df.columns}")
    return test_class_df

def percent_nuclei():
    prediction_dfs = os.listdir(PREDICT_PATH)
    percent_tumor_list = []
    for csv in prediction_dfs:
        percent_tumor = 0.0
        df = pd.read_csv(os.path.join(PREDICT_PATH,csv))
        if isinstance(df,pd.DataFrame):
            tumor = df['class'] == 1.0
            count_tumor = len(df[tumor])
            percent_tumor = (count_tumor/df.shape[0])*100
            percent_tumor_list.append(percent_tumor)     
            logger.info(f"% tumor in {csv[:-20] is {percent_tumor}}")
        else:
            logger.info("csv not formatted properly")    
    percent_tumor_array = np.array(percent_tumor_list)      

    plt.figure(figsize=(12,12))
    sns.set_style('whitegrid')
    sns.set(font_scale=0.8)     
    sns.histplot(data=percent_tumor_array,binrange=(0,100),legend=False)
    plt.savefig(os.path.join(PREDICT_PATH,"percent_tumor.png"))


def relabel_image(feature_pair,class_df):
    features, labels = feature_pair
    if isinstance(class_df,pd.DataFrame):
        logger.info("reading file")
        img_file = os.path.join(LABEL_PATH,labels)
        labels = io.imread(img_file)
        # extract the 2 columns of the label dataframe as arrays
        label_objects = class_df.loc[:,'Label']
        labels_class = class_df.loc[:,'class']+100 # upindex the classes to remove 0 as a class
        # create a dictionary with labels as key and class as value
        label_dict ={}
        for label_,class_ in zip(label_objects,labels_class):
            label_dict[label_]=class_
        # create a new array of the same dimensions as label image
        new_class_labels = np.zeros_like(labels)
        # the label image and its copy are 2-D. flatten them to reduce search space while re-assigning the array
        flat_labels = labels.flatten()
        flat_new_labels = new_class_labels.flatten()
        # assign the class in dictionary to the label in the label image. here idx is the actual value in the flattened
        # label image and count is the position of that value
        for count,idx in enumerate(flat_labels):
            if idx > 0 and idx in label_dict.keys():
                flat_new_labels[count] = label_dict[idx]      # set the value of new labels as the value of label dictionary
        # reshape the new label image to the original shape
        new_labels = flat_new_labels.reshape(labels.shape)   
        io.imsave(fname=os.path.join(CLASSIFIED_IMG_PATH,f"{features[:-20]}_classified_image.png"),arr=new_labels)
        
    else:
        logger.info(f"{features} classified images not generated")


def multiparallel_predict(feature_pair):
    test_class_df = predict_class(feature_pair)
    new_label_image = relabel_image(feature_pair,class_df=test_class_df)



if __name__ == "__main__":
    logger.info("start train classifier and classify nuclei")
    # pool = Pool(10)
    # # get the feature label pair list
    # feature_label_pair_list = create_feature_label_pair(FEATURE_PATH,LABEL_PATH)
    
    # # multiprocessing
    # results = pool.map(multiparallel_predict, feature_label_pair_list)
    # pool.close()
    # pool.join()
    
    percent_nuclei()
        
    
    
