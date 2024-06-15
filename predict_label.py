from __future__ import print_function, unicode_literals, absolute_import, division
# import sys
import numpy as np
# import matplotlib.pyplot as plt

from glob import glob
np.random.seed(6)

import argparse
import cv2


# from tifffile import imread, TiffFile, imwrite
# from csbdeep.utils import Path, normalize
# from csbdeep.io import save_tiff_imagej_compatible
# import csbdeep.utils
# from stardist import random_label_cmap, _draw_polygons, export_imagej_rois
# from stardist.models import StarDist2D
# lbl_cmap = random_label_cmap()


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())


# import histomicstk as htk

# from skimage import io
# import tensorflow as tf
# import torch
# import pandas as pd




def preprocess(image):
    # histogram equalization
    pass

def segmentation(image):
    # normalization

    # segmentation

    pass

def feature_extraction(image):
    
    pass

def classification(image,model):

    pass

if __name__ == "__main__":
    # load the image from disk via "cv2.imread" and then grab the spatial
    # dimensions, including width, height, and number of channels
    image = cv2.imread(args["image"])
    (h, w, c) = image.shape[:3]
    print(h,w,c)

    pass
