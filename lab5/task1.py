## Tiny images representation + nearest neighbor classifier

import numpy as np
import cv2
import os, sys
from matplotlib import pyplot as plt


def ReadFile(Path):
    ImgIdx = []
    Dirs = [ d for d in os.listdir(Path) if not d.startswith(".") ]
    ImgNames = [ os.listdir(Path + d) for d in Dirs if not d.startswith(".") ]
    for Idx in range( len(ImgNames) ): 
        for name in ImgNames[Idx]:
            if not name.startswith("."):
                img = cv2.imread(Path + Dirs[Idx] + "/" + name, cv2.IMREAD_GRAYSCALE)
                ImgIdx.append( (img,Idx) )
    return ImgIdx




train_path = "./hw5_data/train/"
train_list = ReadFile(Path = train_path)
print(train_list)