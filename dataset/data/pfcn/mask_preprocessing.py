import numpy as np
from PIL import Image
import os,sys
import cv2
path = 'masks_png/'
path1 = 'masks_png_final/'
dirs = os.listdir(path)
count = 0
for item in dirs:
    print(item)
    count = count+1
    print(count)
    if os.path.isfile(path+item):
       im = cv2.imread(path + item)
       im[im != 0] = 1
       cv2.imwrite(path1 + item, im)

