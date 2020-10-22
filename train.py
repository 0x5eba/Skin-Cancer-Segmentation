import numpy as np
import os

import Mask.model as modellib
import Mask.visualize as visualize
from Mask.meta.serialize_data import serialize_dataset,deserialize_dataset

def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    MODEL_DIR = dir_path + "/models/"
    COCO_MODEL_PATH = dir_path + "/Mask/mask_rcnn_coco.h5"
    DATA_PATH = dir_path + "/Data/data_set.obj"
    MODEL_PATH = "/models/mask_rcnn_moles_0074.h5"
    ITERATION = 74
    SHOW_SAMPLES = False

    if not os.path.exists(DATA_PATH):
        print('No preprocessed version found')
        dataset_train, dataset_val = serialize_dataset(DATA_PATH)
    else:
        dataset_train, dataset_val = deserialize_dataset(DATA_PATH)

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
    model.load_weights(MODEL_PATH, by_name=True,
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
