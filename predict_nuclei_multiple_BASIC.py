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

# from multiprocessing import Pool

import os
import logging

timestr = time.strftime("%Y%m%d-%H%M%S")

log_file = '/users/ad394h/Documents/nuclei_segment/logs/predict_nuclei_{}.txt'.format(timestr)


logging.basicConfig(filename = log_file, level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


IN = "/users/ad394h/Documents/nuclei_segment/data/karin_he_images_40X/"
DECONV_OUT = "/users/ad394h/Documents/nuclei_segment/data/karin_he_image_labels_40X/"



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
    
    name = inp_image.split("/")[-1][:-4]    
     
    # read the image
    image = io.imread(inp_image)    
    # mormalize the image 
    image = normalize(image, 1,99.8)
    # call the model
    img_label, _ = model.predict_instances(image,nms_thresh=0.3,prob_thresh=0.56,sparse=False,scale=4)   # this should call unique instances of the model     
    num_nuclei = np.unique(img_label).shape[0]
    out_image = f"{name}_predicted_labels.png"
    io.imsave(os.path.join(DECONV_OUT,out_image),img_label)
    logger.info(f"image {inp_image[:-4]} has {num_nuclei} nuclei")    
    return name

if __name__ == '__main__':  
    model = StarDist2D.from_pretrained('2D_versatile_he')  
    for image in glob.glob("/users/ad394h/Documents/nuclei_segment/data/karin_he_images_40X/*.jpg"):
        name = segment_nuclei(image)
        logger.info(f"{name}")
      
    
