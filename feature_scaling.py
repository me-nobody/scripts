import numpy as np
import pandas as pd
import glob, os

from sklearn.preprocessing import MinMaxScaler

IN = "/home/anubratadas/Documents/tumor_normal_pairs/features/"
OUT = "/home/anubratadas/Documents/tumor_normal_pairs/scaled_features/"

scaler = MinMaxScaler()

for file in glob.glob(IN+"*.csv"):
    name = file.split("/")[-1]
    name = "scaled_"+name
    infile = pd.read_csv(file)
    label = infile[["Label"]]
    infile.drop("Label",axis=1,inplace=True)
    scaled_infile = scaler.fit_transform(infile)
    scaled_infile = pd.DataFrame(scaled_infile)
    scaled_infile.columns = infile.columns
    # print(f"infile shape before {scaled_infile.shape}")
    scaled_infile = pd.concat([label,scaled_infile],axis=1,join="inner")
    # print(f"infile shape after {scaled_infile.shape}")
    scaled_infile.to_csv(os.path.join(OUT,name),index=False)
    
