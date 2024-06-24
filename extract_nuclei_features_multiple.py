# HistomicsTK called via module

import sys
import time
import numpy as np
import skimage as ski
from skimage import io

import os
import time
import glob
import logging

from multiprocessing import Pool


import histomicstk as htk
import pandas as pd

timestr = time.strftime("%Y%m%d-%H%M%S")

log_file = '/users/ad394h/Documents/nuclei_segment/logs/extract_features_{}.txt'.format(timestr)


logging.basicConfig(filename=log_file, level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



IN_IMG = "/users/ad394h/Documents/nuclei_segment/data/val_he_images/"
LBL_IMG = "/users/ad394h/Documents/nuclei_segment/data/val_he_image_labels/"
OUT = "/users/ad394h/Documents/nuclei_segment/data/val_he_images_features/"

# create pairs for sending to histomics image feature generator
def create_image_label_pair(IN_IMG,LBL_IMG):
    image_file_list = os.listdir(IN_IMG)
    label_file_list = os.listdir(LBL_IMG)
    logger.info(f"number of image files {len(image_file_list)}")
    logger.info(f"number of label files {len(label_file_list)}")
    assert len(image_file_list)>0,"no files present"
    assert len(label_file_list)>0,"no files present"
    assert all(img[-4:] == ".jpg" for img in image_file_list),"non jpeg files"
    assert all(img[-4:] == ".png" for img in label_file_list),"non png files"
    # matched image and label
    image_file_pair_list = [(a,b) for a in image_file_list for b in label_file_list if a[:21] == b[:21]]
        
    return image_file_pair_list    

def extract_features(image_pair):
    nuclei_features_dict = {}
    image_name,label_name = image_pair
    image_path = os.path.join(IN_IMG,image_name)
    label_path = os.path.join(LBL_IMG,label_name)
    image = io.imread(image_path)
    label = io.imread(label_path)        
    image = image[:,:,0]
    image_name = image_name[:21]
    # logger.info(f"image name {image_name}")
    # logger.info(f"image shape {image.shape} label shape {label.shape}")
    nuclei_features = htk.features.compute_nuclei_features(label,image)    
    # nuclei_features.to_csv(os.path.join(OUT,"{image_name}_nuclei_features.csv".format(image_name)),index=False)
    return nuclei_features,image_name   # when we want individual feature df for each image
    # return nuclei_features  # when we want to combine all the feature df

if __name__ == "__main__":
    pool = Pool(10)
    image_pair_list = create_image_label_pair(IN_IMG=IN_IMG,LBL_IMG=LBL_IMG)
    # parallely execute feature extraction
    results = pool.map(extract_features, image_pair_list)
    pool.close()
    pool.join()
    for result in results:
        features,name = result
        name = name + "_nuclei_features.csv"
        features.to_csv(os.path.join(OUT,name))
    # results_df = pd.concat(results)
    # logger.info(f"results_df shape {results_df.shape}")
    # results_df.to_csv(os.path.join(OUT,"gfp_negative_nuclei_features.csv"),index=False)
        



    
    
