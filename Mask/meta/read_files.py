import cv2
import os
import json
import itertools
from tqdm import tqdm
import multiprocessing as mp
from Mask.meta.meta_data import Metadata


def read_file(filename_and_path):
    filename = filename_and_path[0]
    path_data = filename_and_path[1]

    if len(filename) > 12:
        filename = filename[:12]

    data = json.load(open(path_data + "/Descriptions/" + filename))
    img = cv2.imread(path_data + "Images/" + filename + ".jpeg")
    if img is None:
        return None
    img = cv2.resize(img, (128, 128))
    mask = cv2.imread(path_data + "Segmentation/" + filename + "_expert.png")
    if mask is None:
        return None
    mask = cv2.resize(mask, (128, 128))
    info = Metadata(data["meta"], data["dataset"], img, mask)
    return info


def read_all_files_async(path_data, pool_size=6):
    with mp.Pool(pool_size) as pool:
        files = os.listdir(path_data + "Descriptions/")
        x = zip(files, itertools.cycle([path_data]))
        all_info = list(tqdm(pool.imap(read_file, x), total=len(files)))
        all_info = [info for info in all_info if info is not None]

    pool.join()
    return all_info

def read_all_files(path_data):
    all_info = []
    for filename in tqdm(os.listdir(path_data + "Descriptions/")):
        all_info.append(read_file([filename, path_data]))

    return all_info