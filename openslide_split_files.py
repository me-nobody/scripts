# https://youtu.be/QntLBvUZR5c


import time
import os
import logging

timestr = time.strftime("%Y%m%d-%H%M%S")

log_file = '/users/ad394h/Documents/nuclei_segment/logs/openslide_split_files_{}.txt'.format(timestr)

logging.basicConfig(filename= log_file, filemode='w',level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
import openslide
from PIL import Image

#Load the slide file (svs) into an object.
IN ="/users/ad394h/Documents/nuclei_segment/training_data/karin/"
OUT = "/users/ad394h/Documents/nuclei_segment/training_data/normal/"

slides = os.listdir(IN)

#Generate object for tiles using the DeepZoomGenerator
for slide in slides:
    logger.info(slide)
    slide_name = slide
    #Load the slide file into an object.
    slide = open_slide(os.path.join(IN,slide))

    # convert to deepzoom project
    tiles = DeepZoomGenerator(slide, tile_size=1792, overlap=0, limit_bounds=False)
    logger.info(f"The number of levels in the tiles object are: {tiles.level_count}")

    logger.info(f"The dimensions of data in each level are: {tiles.level_dimensions}")

    # #Tile count at the highest resolution level (level 16 in our tiles)
    tile_Count = tiles.level_tiles[15] 
    logger.info(f"tiles are {tile_Count[0]} and {tile_Count[1]}")
    
    # Saving each tile to local directory
    cols, rows = tile_Count

    for row in range(rows):
        for col in range(cols):
            tile_name = slide_name+ "_" + '%d_%d' % (col, row)
            tile_name = os.path.join(OUT,tile_name)
            # logger.info(f"Now saving tile with title: {tile_name}")
            temp_tile = tiles.get_tile(15, (col, row))
            temp_tile_RGB = temp_tile.convert('RGB')
            temp_tile_RGB.save(tile_name+".jpg")
            


