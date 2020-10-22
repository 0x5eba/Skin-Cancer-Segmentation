import pickle
import os
import numpy as np
from Mask.meta.config.moles_config import MolesConfig
from Mask.meta.mole_dataset import MoleDataset
from Mask.meta.read_files import read_all_files_async, read_all_files


def serialize_dataset(file_path, ASYNC_READ=True):
    config = MolesConfig()
    all_info = []

    # path of Data that contain Descriptions and Images
    path_data = './Data/'  # input("Insert the path of Data [ Link /home/../ISIC-Archive-Downloader/Data/ ] : ")
    if not os.path.exists(path_data):
        raise Exception(path_data + " Does not exists")

    warning = True

    # Load all the images, mask and description of the Dataset
    print('start loading images...')
    all_info = read_all_files_async(path_data) if ASYNC_READ else read_all_files(path_data)

    # split the meta into train and test
    percentual = (len(all_info) * 30) // 100
    np.random.shuffle(all_info)
    train_data = all_info[:-percentual]
    val_data = all_info[percentual + 1:]
    del all_info

    # processing the meta
    dataset_train = MoleDataset()
    dataset_train.load_shapes(train_data, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_train.prepare()
    dataset_val = MoleDataset()
    dataset_val.load_shapes(val_data, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_val.prepare()
    del train_data
    del val_data


    with open(file_path) as f:
        pickle.dump([dataset_train, dataset_val], f)

    return dataset_train, dataset_val

def deserialize_dataset(file_path):
    with open(file_path) as f:
        dataset_train, dataset_val = pickle.load(f)

    return dataset_train, dataset_val
