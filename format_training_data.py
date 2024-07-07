import pandas as pd
import glob, os

INFILE = "/home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/Nuclei_segmentation/Training_Data/Anu_data/normal/anu_normal_features/*.csv"


df = pd.DataFrame()

for file in glob.glob(INFILE):
    print(file)