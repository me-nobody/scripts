# mamba environment in MARS/users/ad394h/miniforge-pypy3/envs/
# StarDist and HistomicsTK installed

import sys
import time
import numpy as np
import skimage as ski
from skimage import io

import os
import time
import logging

import histomicstk as htk
import pandas as pd

timestr = time.strftime("%Y%m%d-%H%M%S")

log_file = '/users/ad394h/Documents/nuclei_segment/logs/classify_nuclei.txt'


logging.basicConfig(filename=log_file, level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



IMG = "image_4_GFP.ndpi_8_18.jpg"
LABEL = f"{IMG}_predicted_image_label.tiff"
IN = "/users/ad394h/Documents/nuclei_segment/data/"
OUT = "/users/ad394h/Documents/nuclei_segment/data/"

img_file = os.path.join(IN,IMG)
label_file = os.path.join(OUT,LABEL)

if __name__ == "__main__":
    image = io.imread(img_file)[:,:,0]
    io.imsave(os.path.join(OUT,f"{IMG}_nuclei_features.jpg"),image)
    label = io.imread(label_file)
    logger.info(f"image shape {image.shape}")
    logger.info(f"label shape {label.shape}")
    nuclei_features = htk.features.compute_nuclei_features(label,image)
    nuclei_features.to_csv(os.path.join(OUT,"nuclei_features.csv"),index=False)
    
    
