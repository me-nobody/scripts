#!/bin/bash

counter=0
for id in $(ls /home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/Nuclei_segmentation/Training_Data/Anu_data/tumor/anu_tumor_images);
do 
  if [[ -e /home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/Nuclei_segmentation/anu_he_cd31_claudin_image_labels_40X/${id:0:20}.png ]]; then
  echo ${id:0:20}.png
  # cp "be50.PASS/$id.VEP.vcf" "be50.MSI"
  else
  echo not found
  echo ${id}
  fi
  let counter++
  echo $counter

done

echo number of files
ls -lh   /home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/Nuclei_segmentation/anu_he_cd31_claudin_image_labels_40X| wc -l
# echo space on disk
# du -h .