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


DIRECTORY = "~/DeepLearning/Uqido/NumpyData/"


def get_mask(img, i):
    selection = SelectionWindow('Selection Window', img, i, connectivity=8)
    selection.show()

    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    # brown = cv2.inRange(hsv, (5, 0, 0), (15, 255, 255))
    # magenta = cv2.inRange(hsv, (295, 0, 0), (305, 255, 255))
    # # blue = cv2.inRange(hsv, (230, 50, 0), (250, 150, 255))
    # mask = cv2.bitwise_or(brown, magenta)
    # target = cv2.bitwise_and(img, img, mask=mask)
    # black = cv2.inRange(target, (0, 0, 0), (0, 0, 255))
    # tot_black = cv2.countNonZero(black)
    # return target, tot_black


for i in range(1367, 13786):

    num = str(i).zfill(7)
    data = json.load(open(f"Data/Data/Descriptions/ISIC_{num}"))

    if data["meta"]["clinical"]["benign_malignant"] is None:
        continue

    if data["meta"]["clinical"]["benign_malignant"] != "malignant" and data["meta"]["clinical"]["benign_malignant"] != "benign":
        continue

    # if data["meta"]["clinical"]["benign_malignant"] == "benign":
    #     continue

    img = cv2.imread(f"Data/Data/Images/ISIC_{num}.jpg")
    img = cv2.resize(img, (128, 128))

    get_mask(img, num)
    
    # if all_black > 15500 or all_black < 9000:
    #     continue

    with open(DIRECTORY + f"ISIC_{num}_IMG.npy", "w") as f:
        np.save(DIRECTORY + f"ISIC_{num}_IMG", img)
    
    # with open(DIRECTORY + f"ISIC_00{num}_MASK.npy", "w") as f:
    #     np.save(DIRECTORY + f"ISIC_00{num}_MASK", mask)
    
    # print("created")

