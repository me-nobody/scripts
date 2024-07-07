# StarDist and HistomicsTK installed

import sys
import time
import numpy as np
import glob
import skimage as ski
from skimage import io
from skimage.color import rgb2hed, hed2rgb

from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible

from stardist import random_label_cmap, _draw_polygons, export_imagej_rois
from stardist.models import StarDist2D

np.random.seed(6)
lbl_cmap = random_label_cmap()


import os
import logging

timestr = time.strftime("%Y%m%d-%H%M%S")

log_file = '/users/ad394h/Documents/nuclei_segment/logs/predict_nuclei_{}.txt'.format(timestr)


logging.basicConfig(filename = log_file, level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


IN = "/users/ad394h/Documents/nuclei_segment/data/image3_gfp_positive_images/"

DECONV_OUT = "/users/ad394h/Documents/nuclei_segment/data/image3_gfp_positive_image_labels/"

LABEL_OUT = "/users/ad394h/Documents/nuclei_segment/data/image3_gfp_positive_image_labels/"


def extract_DAB(image):
    name = image[:-4]+"_deconv_image.jpg"
    image = ski.io.imread(os.path.join(IN,image))
    image = rgb2hed(image)
    dab_img = image[:,:,2]
    null = np.zeros_like(dab_img)
    dab_img = hed2rgb(np.stack((null, null, dab_img), axis=-1))
    dab_img = dab_img*255
    dab_img = dab_img.astype(np.uint8)    
    io.imsave(os.path.join(DECONV_OUT,name),dab_img)
    logger.info(f"extracted image shape {dab_img.shape[0]} and {dab_img.shape[1]}")
    return dab_img


def segment_nuclei(image,file):
    name = file[:-4]
    model = StarDist2D.from_pretrained('2D_versatile_he')    
    if not model:
        logger.info("model has not been loaded")
    else:
        logger.info("model exists")
    
    # mormalize the image 
    image = normalize(image, 1,99.8)
    # call the model
    img_label, _ = model.predict_instances(image)   # this should call unique instances of the model     
    num_nuclei = np.unique(img_label).shape[0]
    out_image = name+"_predicted_labels.png"
    if isinstance(out_image,str):
        io.imsave(os.path.join(LABEL_OUT,out_image),img_label)
        logger.info(f"image {file[:-4]} has {num_nuclei} nuclei")    
    # return num_nuclei,img_label

if __name__ == '__main__':  
    for file in os.listdir(IN):
        image = extract_DAB(file)
        segment_nuclei(image=image,file=file)

    
    
