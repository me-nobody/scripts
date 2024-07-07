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



import histomicstk as htk
import pandas as pd

timestr = time.strftime("%Y%m%d-%H%M%S")

log_file = '/users/ad394h/Documents/nuclei_segment/logs/extract_features_{}.txt'.format(timestr)


logging.basicConfig(filename=log_file, level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



IN_IMG = "/users/ad394h/Documents/nuclei_segment/data/anu_he_cd31_claudin_images_40X/"
LBL_IMG = "/users/ad394h/Documents/nuclei_segment/data/anu_he_cd31_claudin_image_labels_40X/"
OUT = "/users/ad394h/Documents/nuclei_segment/data/anu_he_cd31_claudin_images_40X_features/"



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
    # for a in image_file_list:
    #     logger.info(f"image {a[:-4]}")
    # for b in label_file_list:
    #     logger.info(f"label {b[:-21]}")    
    image_file_pair_list = [(a,b) for a in image_file_list for b in label_file_list if a[:-4] == b[:-21]]
        
    return image_file_pair_list    


def extract_hematoxylin(IMAGE):
    # img = ski.io.imread(os.path.join(IN_IMG,IMAGE))
    # create stain to color map
    stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map

    # specify stains of input image
    stains = ['hematoxylin',  # nuclei stain
              'eosin',        # cytoplasm stain
              'null']         # set to null if input contains only two stains

    # create stain matrix
    W = np.array([stain_color_map[st] for st in stains]).T

    # perform standard color deconvolution
    imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(IMAGE, W)

    deconv_image = imDeconvolved.Stains[:,:,0]

    return deconv_image


def extract_features(image_pair):
    nuclei_features_dict = {}
    image_name,label_name = image_pair
    image_path = os.path.join(IN_IMG,image_name)
    label_path = os.path.join(LBL_IMG,label_name)
    image = io.imread(image_path)
    label = io.imread(label_path)   
    if isinstance(image,np.ndarray) and isinstance(label,np.ndarray):
        hematoxylin_img = extract_hematoxylin(image)
        image_name = image_name[:-4]
        nuclei_features = htk.features.compute_nuclei_features(im_label=label,im_nuclei=hematoxylin_img)    
        # nuclei_features.to_csv(os.path.join(OUT,"{}_nuclei_features.csv".format(image_name)),index=False)
        return nuclei_features,image_name   # when we want individual feature df for each image
        # return nuclei_features  # when we want to combine all the feature df
    else:
        logger.info("image and label files not generated")

if __name__ == "__main__":
    
    image_pair_list = create_image_label_pair(IN_IMG=IN_IMG,LBL_IMG=LBL_IMG)
    logger.info(f"{len(image_pair_list)} image pairs created")
    # parallely execute feature extraction
    for pair in image_pair_list:
        try:
            nuclei_features,image_name = extract_features(pair)    
            nuclei_features.to_csv(os.path.join(OUT,"{}_nuclei_features.csv".format(image_name)),index=False)
        
        except IndexError as err:
            image_name,label_name = pair
            logger.info(f"{label_name} has calculation errors")
            continue




    
    
