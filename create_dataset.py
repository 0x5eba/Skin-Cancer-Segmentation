import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt

from Mask.config import Config
import Mask.utils as utils
import Mask.model as modellib
import Mask.visualize as visualize
from Mask.model import log

from Mask.magicwand import SelectionWindow


dir_path = os.path.dirname(os.path.realpath(__file__))
DIR_NUMPYDATA = dir_path + "/NumpyData/"
DIR_DATA = dir_path + "/Data/"

def get_mask(img, i):
    selection = SelectionWindow('Selection Window', img, i, DIR_NUMPYDATA, 8)
    selection.show()


for i in range(1367, 13786):

    num = str(i).zfill(7)
    data = json.load(open(DIR_DATA + f"Descriptions/ISIC_{num}"))

    if data["meta"]["clinical"]["benign_malignant"] is None:
        continue

    if data["meta"]["clinical"]["benign_malignant"] != "malignant" and data["meta"]["clinical"]["benign_malignant"] != "benign":
        continue

    img = cv2.imread(DIR_DATA + f"Images/ISIC_{num}.jpg")
    img = cv2.resize(img, (128, 128))

    get_mask(img, num)

    with open(DIR_NUMPYDATA + f"ISIC_{num}_IMG.npy", "w") as f:
        np.save(DIR_NUMPYDATA + f"ISIC_{num}_IMG", img)
    
