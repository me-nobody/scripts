import numpy as np
import skimage as ski
from skimage.color import rgb2hed, hed2rgb
import os
from skimage import exposure

import os
import logging


logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


REFERENCE = "/users/ad394h/Documents/nuclei_segment/data/Image_26_10x_REFERENCE.jpg"

IN = "/users/ad394h/Documents/nuclei_segment/gfp_negative_images/"
DECONV_OUT = "/users/ad394h/Documents/nuclei_segment/data/nogfp_hematoxylin_images/"
MATCHED_OUT = "/users/ad394h/Documents/nuclei_segment/data/nogfp_hist_matched_images/"

def extract_hematoxylin(IN):
    file_list = os.listdir(IN)
    for file in file_list:
        img = os.path.join(IN,file)
        img = ski.io.imread(img)
        img = rgb2hed(img)
        null = np.zeros_like(img[:,:,0])
        deconv_img = hed2rgb(np.stack((img[:,:,0],null,null),axis=-1))
        deconv_img = deconv_img*255
        deconv_img = deconv_img.astype(np.uint8)
        ski.io.imsave(DECONV_OUT+file,deconv_img)

def match_histograms(DECONV_OUT):
    file_list = os.listdir(DECONV_OUT)
    ref = ski.io.imread(REFERENCE)
    
    logger.info(f"shape of reference {ref.shape}")
    for file in file_list:
        src = os.path.join(DECONV_OUT,file)
        src = ski.io.imread(src)
        src = src[:,:,2]
        logger.info(f"shape of source {src.shape}")
        # determine if we are performing multichannel histogram matching
        # and then perform histogram matching itself
        multi = True if src.shape[-1] > 1 else False
        matched = exposure.match_histograms(src, ref, multichannel=False)
        ski.io.imsave(MATCHED_OUT+file,matched)    

if __name__ == "__main__":
    match_histograms(DECONV_OUT)
