import pandas as pd
import glob, os

INFILE = "/home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/Nuclei_segmentation/Training_Data/Anu_data/tumor/anu_mixed_features/*.csv"
OUTFILE = "/home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/Nuclei_segmentation/Training_Data/Anu_data/"

df = pd.DataFrame()

for file in glob.glob(INFILE):
    new_df = pd.read_csv(file)
    new_df.drop("Label",inplace=True,axis=1)
    df = pd.concat([df,new_df],axis=0)

print(df.shape)
df = df.sample(n=100000,axis=0)
df.to_csv(os.path.join(OUTFILE,"anu_mixed_features_40X.csv"),index=False)