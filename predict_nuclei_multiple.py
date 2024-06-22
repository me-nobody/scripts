# mamba environment in MARS/users/ad394h/miniforge-pypy3/envs/
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

from multiprocessing import Pool

import os
import logging

timestr = time.strftime("%Y%m%d-%H%M%S")

log_file = '/users/ad394h/Documents/nuclei_segment/logs/predict_nuclei_{}.txt'.format(timestr)


logging.basicConfig(filename = log_file, level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


IN = "/users/ad394h/Documents/nuclei_segment/data/gfp_negative_images/"
DECONV_OUT = "/users/ad394h/Documents/nuclei_segment/data/gfp_negative_label_images/"



# def extract_hematoxylin(IMG):
#     img = ski.io.imread(os.path.join(IN,IMG))
#     img = rgb2hed(img)
#     null = np.zeros_like(img[:,:,0])
#     deconv_img = hed2rgb(np.stack((img[:,:,0],null,null),axis=-1))
#     deconv_img = deconv_img*255
#     deconv_img = deconv_img.astype(np.uint8)
#     # ski.io.imsave(DECONV_OUT+f"{IMG[:-4]}_deconv_image.jpg",deconv_img)
#     return deconv_img


def segment_nuclei(inp_image):
    model_dict ={} # this is an expensive way to create multiple images of the model. the error messages
                   # in the slurm cluster may be due to multiple processes trying to access the same model
    model_id = inp_image[64:-4]               
    model_dict[model_id] = StarDist2D.from_pretrained('2D_versatile_he')    
    if not model_dict[model_id]:
        logger.info("model has not been loaded")
    else:
        logger.info("model exists")
    # read the image
    image = io.imread(inp_image)
    # mormalize the image 
    image = normalize(image, 1,99.8)
    # call the model
    img_label, _ = model_dict[model_id].predict_instances(image)   # this should call unique instances of the model     
    num_nuclei = np.unique(img_label).shape[0]
    out_image = f"{inp_image[64:-4]}_predicted_labels.png"
    io.imsave(os.path.join(DECONV_OUT,out_image),img_label)
    logger.info(f"image {inp_image[:-4]} has {num_nuclei} nuclei")    
    return num_nuclei,img_label

if __name__ == '__main__':    
    pool = Pool(10)
    # Create a multiprocessing Pool
    pool.map(segment_nuclei, glob.glob("/users/ad394h/Documents/nuclei_segment/data/gfp_negative_images/*.jpg")) 

    
    
