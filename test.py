import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import glob

from Mask.config import Config
import Mask.utils as utils
import Mask.model as modellib
import Mask.visualize as visualize

np.set_printoptions(threshold=np.inf)

# path of the trained model
dir_path = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = dir_path + "/models/"
MODEL_PATH = input("Insert the path of your trained model [ Like models/moles.../mask_rcnn_moles_0030.h5 ]: ")
if os.path.isfile(MODEL_PATH) == False:
    raise Exception(MODEL_PATH + " Does not exists")

# path of Data that contain Descriptions and Images
path_data = input("Insert the path of Data [ Link /home/../ISIC-Archive-Downloader/Data/ ] : ")
if not os.path.exists(path_data):
    raise Exception(path_data + " Does not exists")


class CocoConfig(Config):
    ''' 
    MolesConfig:
        Contain the configuration for the dataset + those in Config
    '''
    NAME = "moles"
    NUM_CLASSES = 1 + 2 # background + (malignant , benign)
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MAX_INSTANCES = 3

# create and instance of config
config = CocoConfig()

# take the trained model
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(MODEL_PATH, by_name=True)

# background + (malignant , benign)
class_names = ["BG", "malignant", "benign"]

# find the largest number of image that you download
all_desc_path = glob.glob(path_data + "Descriptions/ISIC_*")
for filename in os.listdir(path_data+"Descriptions/"):
    data = json.load(open(path_data+"/Descriptions/"+filename))
    img = cv2.imread(path_data+"Images/"+filename+".jpg")
    img = cv2.resize(img, (128, 128))

    if not img:
        continue

    # ground truth of the class
    print(data["meta"]["clinical"]["benign_malignant"])
    
    # predict the mask, bounding box and class of the image
    r = model.detect([img])[0]
    visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])
