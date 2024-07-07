import os, sys, shutil

INFILE = "/home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/Nuclei_segmentation/Training_Data/Anu_data/normal/anu_normal_images/"

SCANFILE = "/home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/Nuclei_segmentation/anu_he_cd31_claudin_image_labels_40X/"

SCANFILE2 = "/home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/Nuclei_segmentation/anu_he_cd31_claudin_images_40X_features/"


OUTFILE = "/home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/Nuclei_segmentation/Training_Data/Anu_data/normal/anu_normal_labels/"

OUTFILE2 = "/home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/Nuclei_segmentation/Training_Data/Anu_data/normal/anu_normal_features/"


for file1 in os.listdir(INFILE):
    for file2 in os.listdir(SCANFILE2):
        # print(f"infile {file1[:-4]} scanfile {file2[:-20]}")        
        if file1[:-4] == file2[:-20]:
            # print(f"infile {file1} scanfile {file2}")
            shutil.copy(os.path.join(SCANFILE2,file2),os.path.join(OUTFILE2,file2))
        

print(f"{len(os.listdir(OUTFILE2))} files transferred")            
            
        
    
        