import time
import os
import logging
import shutil

timestr = time.strftime("%Y%m%d-%H%M%S")

log_file = '/users/ad394h/Documents/nuclei_segment/logs/check_empty_file_{}.txt'.format(timestr)

logging.basicConfig(filename= log_file, filemode='w',level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import numpy as np
import cv2

IN = "/users/ad394h/Documents/nuclei_segment/training_data/tumor/"
OUT = "/users/ad394h/Documents/nuclei_segment/discard/"


def is_empty(img):
   # Reading Image
   image = os.path.join(IN,img)
   image = cv2.imread(image, 0)
   np.reshape(image, (-1,1))
   u, count_unique = np.unique(image, return_counts =True)
   
   if count_unique.size < 120:
      logger.info(f"{img} Image is empty")
      shutil.move(IN+img,OUT+img)
   else:
      logger.info(f"{img} is not empty")

if __name__ == "__main__":
   for image in os.listdir(IN):
      is_empty(image)
      
