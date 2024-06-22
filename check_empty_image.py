import time
import os
import logging

timestr = time.strftime("%Y%m%d-%H%M%S")

log_file = '/users/ad394h/Documents/nuclei_segment/logs/check_empty_file_{}.txt'.format(timestr)

logging.basicConfig(filename= log_file, filemode='w',level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import numpy as np
import cv2

IN ="/users/ad394h/Documents/nuclei_segment/training_data/normal/"


def is_empty(img):
   # Reading Image
   img = os.path.join(IN,img)
   image = cv2.imread(img, 0)
   np.reshape(image, (-1,1))
   u, count_unique = np.unique(image, return_counts =True)
   
   if count_unique.size< 50:
      logger.info(f"{img} Image is empty")
   else:
      logger.info(f"{img} is not empty")

if __name__ == "__main__":
   for image in os.listdir(IN):
      is_empty(image)
      