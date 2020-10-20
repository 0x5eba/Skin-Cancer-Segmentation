import numpy as np
import os

from Mask.config.moles_config import MolesConfig
import Mask.model as modellib
import Mask.visualize as visualize
from Mask.data.read_files import read_all_files_async, read_all_files
from Mask.data.mole_dataset import MoleDataset

def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    MODEL_DIR = dir_path + "/models/"
    COCO_MODEL_PATH = dir_path + "/Mask/mask_rcnn_coco.h5"
    MODEL_PATH = "/models/mask_rcnn_moles_0074.h5"
    ITERATION = 74
    ASYNC_READ = True
    SHOW_SAMPLES = False
    if not os.path.isfile(COCO_MODEL_PATH):
        raise Exception("You have to download mask_rcnn_coco.h5 inside Mask folder \n\
        You can find it here: https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5")    

    config = MolesConfig()
    all_info = []

    # path of Data that contain Descriptions and Images
    path_data = './Data/'#input("Insert the path of Data [ Link /home/../ISIC-Archive-Downloader/Data/ ] : ")
    if not os.path.exists(path_data):
        raise Exception(path_data + " Does not exists")

    warning = True

    # Load all the images, mask and description of the Dataset
    print('start loading images...')
    all_info = read_all_files_async(path_data) if ASYNC_READ else read_all_files(path_data)


    # split the data into train and test
    percentual = (len(all_info)*30)//100
    np.random.shuffle(all_info)
    train_data = all_info[:-percentual]
    val_data = all_info[percentual+1:]
    del all_info

    # processing the data
    dataset_train = MoleDataset()
    dataset_train.load_shapes(train_data, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_train.prepare()
    dataset_val = MoleDataset()
    dataset_val.load_shapes(val_data, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_val.prepare()
    del train_data
    del val_data

    # Show some random images to verify that everything is ok
    if SHOW_SAMPLES:
        print(dataset_train.image_ids)
        image_ids = np.random.choice(dataset_train.image_ids, 3)
        for image_id in image_ids:
            image = dataset_train.load_image(image_id)
            mask, class_ids = dataset_train.load_mask(image_id)
            visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

    # Create the MaskRCNN model
    print('creating model...')
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

    # Use as start point the coco model
    print('loading weights...')
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])

    # Train the model on the train dataset
    # First only the header layers
    if ITERATION < 30:
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=30-ITERATION,
                    layers='heads')
    # After all the layers 
    if ITERATION < 90:
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE/10,
                    epochs=90-ITERATION,
                    layers="all")

    print("Trained finished!")

if __name__ == "__main__":
    main()
