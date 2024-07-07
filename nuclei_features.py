# mamba environment in MARS/users/ad394h/miniforge-pypy3/envs/
# StarDist and HistomicsTK installed

import sys
import numpy as np
import skimage as ski
from skimage import io

import os
import time
import logging

import histomicstk as htk
import pandas as pd

timestr = time.strftime("%Y%m%d-%H%M%S")

log_file = '/users/ad394h/Documents/nuclei_segment/logs/predict_features_{}.txt'.format(timestr)

logging.basicConfig(filename=log_file, filemode='a',level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

timestr = time.strftime("%Y%m%d-%H%M%S")


IMG = "test_image.jpg"
LABEL = "predicted_image_label.tiff"
IN = "/users/ad394h/Documents/nuclei_segment/data/"
OUT = "/users/ad394h/Documents/nuclei_segment/data/"

img_file = os.path.join(IN,IMG)
label_file = os.path.join(OUT,LABEL)

if __name__ == "__main__":
    image = io.imread(img_file)[:,:,0]
    io.imsave(os.path.join(OUT,"nuclei_features_image.jpeg"),image)
    label = io.imread(label_file)
    logger.info(f"image shape {image.shape}")
    logger.info(f"label shape {label.shape}")
    tumor_features = htk.features.compute_nuclei_features(label,image)
    tumor_features.to_csv(os.path.join(OUT,"tumor_features.csv"),index=False)
    
    
