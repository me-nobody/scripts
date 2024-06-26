# StarDist and HistomicsTK installed

import sys
import time
import numpy as np
import skimage as ski
from skimage import io
# from skimage.color import rgb2hed, hed2rgb

import os
import time
import glob
import logging

import histomicstk as htk
import pandas as pd

timestr = time.strftime("%Y%m%d-%H%M%S")

log_file = '/users/ad394h/Documents/nuclei_segment/logs/classify_nuclei.txt'


logging.basicConfig(filename=log_file, level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



IMAGE = "image.jpg"

LABEL = "label.png"

IN = "/users/ad394h/Documents/nuclei_segment/data/"
OUT = "/users/ad394h/Documents/nuclei_segment/data/"


def extract_hematoxylin(IMAGE):
    img = ski.io.imread(os.path.join(IN,IMAGE))
    # create stain to color map
    stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map

    # specify stains of input image
    stains = ['hematoxylin',  # nuclei stain
              'eosin',        # cytoplasm stain
              'null']         # set to null if input contains only two stains

    # create stain matrix
    W = np.array([stain_color_map[st] for st in stains]).T

    # perform standard color deconvolution
    imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(img, W)

    deconv_image = imDeconvolved.Stains[:,:,0]

    return deconv_image

# this is an alternate method to extract hematoxylin channel

# def extract_hematoxylin(IMAGE):
#     img = ski.io.imread(os.path.join(IN,IMAGE))
#     img = rgb2hed(img)
#     null = np.zeros_like(img[:,:,0])
#     deconv_img = hed2rgb(np.stack((img[:,:,0],null,null),axis=-1))
#     deconv_img = deconv_img*255
#     deconv_img = deconv_img.astype(np.uint8)
#     ski.io.imsave(OUT+f"{IMAGE[:-4]}_deconv_image.jpg",deconv_img)
#     return deconv_img


if __name__ == "__main__":
    # io.imsave(os.path.join(OUT,f"{IMAGE[:-4]}_nuclei_features.jpg"),image)
    label = io.imread(os.path.join(IN,LABEL))
    logger.info(f"label shape {label.shape}")
    deconv_image = extract_hematoxylin(IMAGE)
    logger.info(f"hematoxylin channel image shape {deconv_image.shape}")
    nuclei_features = htk.features.compute_nuclei_features(im_label=label,im_nuclei=deconv_image)
    nuclei_features.to_csv(os.path.join(OUT,"nuclei_features.csv"),index=False)
    
    
