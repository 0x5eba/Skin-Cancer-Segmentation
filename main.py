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

MODEL_DIR = os.path.join(os.getcwd(), "logs")
COCO_MODEL_PATH = os.path.join("~/DeepLearning/Uqido/Mask/", "mask_rcnn_coco.h5")


class MolesConfig(Config):
    NAME = "moles"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 2  # background + malignant , benign
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 5
    RPN_TRAIN_ANCHORS_PER_IMAGE = 150

class Info:
    def __init__(self, meta, dataset, img, mask):
        self.meta = meta
        self.dataset = dataset
        self.img = img
        self.type = meta["clinical"]["benign_malignant"]  # malignant , benign
        self.mask = mask


class MoleDataset(utils.Dataset):

    def load_shapes(self, dataset, height, width):
        self.add_class("moles", 1, "malignant")
        self.add_class("moles", 2, "benign")

        for i, info in enumerate(dataset):
            height, width, channels = info.img.shape
            self.add_image(source="moles", image_id=i, path=None,
                           width=width, height=height,
                           img=info.img, shape=(info.type, channels, (height, width)),
                           mask=info.mask, extra=info)

    def load_image(self, image_id):
        return self.image_info[image_id]["img"]

    def image_reference(self, image_id):
        if self.image_info[image_id]["source"] == "moles":
            return self.image_info[image_id]["shape"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        shapes = info["shape"]
        mask = info["mask"].astype(np.uint8)
        class_ids=np.array([self.class_names.index(shapes[0])])
        return mask, class_ids.astype(np.int32)
    

config = MolesConfig()
all_info = []

for filename in os.listdir("~/DeepLearning/Uqido/NumpyData"):
    num = filename.split("_")[1]
    data = json.load(open(f"Data/Data/Descriptions/ISIC_{num}"))
    img = np.load(f"~/DeepLearning/Uqido/NumpyData/ISIC_{num}_IMG.npy")
    mask = np.load(f"~/DeepLearning/Uqido/NumpyData/ISIC_{num}_MASK.npy")
    info = Info(data["meta"], data["dataset"], img, mask)
    all_info.append(info)



percentual = (len(all_info)*30)//100
np.random.shuffle(all_info)
train_data = all_info[:-percentual]
val_data = all_info[percentual+1:]
del all_info

dataset_train = MoleDataset()
dataset_train.load_shapes(train_data, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

dataset_val = MoleDataset()
dataset_val.load_shapes(val_data, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()

del train_data
del val_data
# image_ids = np.random.choice(dataset_train.image_ids, 3)
# for image_id in image_ids:
#     image = dataset_train.load_image(image_id)
#     mask, class_ids = dataset_train.load_mask(image_id)
#     # print(mask, class_ids)
#     visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)


model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

model.load_weights(COCO_MODEL_PATH, by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                            "mrcnn_bbox", "mrcnn_mask"])
# model.load_weights(model.get_imagenet_weights(), by_name=True)

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=30,
            layers='heads')
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE/10,
            epochs=90,
            layers="all")



class InferenceConfig(MolesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


inference_config = InferenceConfig()
# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)

model.load_weights(model.find_last()[1], by_name=True)

# Test on a random image
for i in range(10):
    image_id = np.random.choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config,
                            image_id, use_mini_mask=False)

    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                dataset_train.class_names, figsize=(8, 8))

    r = model.detect([original_image], verbose=1)[0]

    def get_ax(rows=1, cols=1, size=8):
        _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
        return ax

    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                dataset_val.class_names, r['scores'], ax=get_ax())