# mamba environment in MARS/users/ad394h/miniforge-pypy3/envs/
# StarDist and HistomicsTK installed

import sys
import time
import numpy as np
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

log_file = '/users/ad394h/Documents/nuclei_segment/logs/classify_nuclei.txt'


logging.basicConfig(filename = log_file, level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



IMG = "image_4_GFP.ndpi_8_18.jpg"

IN = "/users/ad394h/Documents/nuclei_segment/data/"
DECONV_OUT = "/users/ad394h/Documents/nuclei_segment/data/"

def extract_hematoxylin(IMG):
    img = ski.io.imread(os.path.join(IN,IMG))
    img = rgb2hed(img)
    null = np.zeros_like(img[:,:,0])
    deconv_img = hed2rgb(np.stack((img[:,:,0],null,null),axis=-1))
    deconv_img = deconv_img*255
    deconv_img = deconv_img.astype(np.uint8)
    ski.io.imsave(DECONV_OUT+f"{IMG}_deconv_image.jpg",deconv_img)
    return deconv_img


def segment_nuclei(inp_image):
    # mormalize the image
    image = normalize(inp_image, 1,99.8)
    # call the model
    img_label, _ = model.predict_instances(image)        
    num_nuclei = np.unique(img_label).shape[0]
    return num_nuclei,img_label

if __name__ == "__main__":
    model = StarDist2D.from_pretrained('2D_versatile_he')
    if not model:
        logger.info("model has not been loaded")
    else:
        logger.info("model exists")
    
    deconv_img = extract_hematoxylin(IMG)
    num_nuclei,img_label = segment_nuclei(deconv_img)
    inp_image = f"{IMG}_predicted_image_label.tiff"
    io.imsave(os.path.join(DECONV_OUT,inp_image),img_label)
    logger.info(f"slide {IMG} has {num_nuclei} nuclei")
    
