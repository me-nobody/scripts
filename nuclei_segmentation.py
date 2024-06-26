# mamba environment in MARS/users/ad394h/miniforge-pypy3/envs/
# StarDist and HistomicsTK installed
from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np

import numpy as np
import skimage as ski
from skimage import io
from skimage.color import rgb2hed, hed2rgb
import os

from glob import glob
from tifffile import imread, TiffFile, imwrite
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible

from stardist import random_label_cmap, _draw_polygons, export_imagej_rois
from stardist.models import StarDist2D

np.random.seed(6)
lbl_cmap = random_label_cmap()

import os
import logging

logging.basicConfig(filename='/users/ad394h/Documents/nuclei_segment/logs/nuclei_segment.txt', filemode='w',level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


import pandas as pd

import multiprocessing as mp


IN = "/users/ad394h/Documents/nuclei_segment/data/nogfp_hematoxylin_images/"
OUT = "/users/ad394h/Documents/nuclei_segment/data/nogfp_hematoxylin_labels/"

def segment_nuclei(inp_image):
    # read the image
    img_path = os.path.join(IN,inp_image)
    image = io.imread(img_path)
    if isinstance(image,np.ndarray):
        logger.info(f"the image shape is {image.shape[0]} x {image.shape[1]}")    
    # mormalize the image
    image = normalize(image, 1,99.8)
    # call the model
    img_label, _ = model.predict_instances(image)  
    # inp_image = inp_image[:-8]+".tiff"
    # io.imsave(os.path.join(OUT,inp_image),img_label)  
    num_nuclei = np.unique(img_label).shape[0]
    return inp_image,num_nuclei,img_label

if __name__ == "__main__":
    model = StarDist2D.from_pretrained('2D_versatile_he')
    if not model:
        logger.info("model has not been loaded")
    else:
        logger.info("model exists")
    
    # access the image file names
    image_list = os.listdir(IN)

    for image in image_list:
        logger.info(f"inputting image {image}")
        result = segment_nuclei(image)
        logger.info(f"slide {result[0]} has {result[1]} nuclei")
    # pool = mp.Pool(processes=8)
    # results = pool.map(segment_nuclei, image_list)
    # for result in results:
    #     logger.info(f"slide {result[0]} has {result[1]} nuclei")

