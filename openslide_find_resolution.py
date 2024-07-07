import time
import os
import logging

timestr = time.strftime("%Y%m%d-%H%M%S")

log_file = '/users/ad394h/Documents/nuclei_segment/logs/slides_resolution{}.txt'.format(timestr)


logging.basicConfig(filename= log_file, filemode='w',level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


from openslide import open_slide
import openslide

from openslide.deepzoom import DeepZoomGenerator

#Load the slide file (svs) into an object.
IN = "/users/ad394h/Documents/nuclei_segment/data/val_he_ndpi/"

slides = os.listdir(IN)

for slide_img in slides:
    logger.info(slide_img)
    slide = open_slide(os.path.join(IN,slide_img))
    slide_props = slide.properties
    
    for key,value in slide_props.items():
        logger.info(f"{key} : {value}")
    
    # convert to deepzoom project
    tiles = DeepZoomGenerator(slide, tile_size=1965, overlap=0, limit_bounds=False)
    logger.info(f"The number of levels in the tiles object are: {tiles.level_count}")
    logger.info(f"The dimensions of data in each level are: {tiles.level_dimensions}")



