# Skin-Cancer-Segmentation
Classification and Segmentation with Mask-RCNN of Skin Cancer by ISIC dataset 

## Setup

1) Download the dataset from https://isic-archive.com/ 
    - You can download it from https://github.com/GalAvineri/ISIC-Archive-Downloader 
      To download the whole archive: `python3 download_archive.py -s`
      
   At the end, the directory of the data should be like this:
   
    ```
    Data/
    ├── Images/  (containing the .jpg file)
    ├── Descriptions/  (containing the json file)
    └── Segmentation/  (containing the .png file)
    ```

2) Download the dependency of the project: `pip3 install -r requirements.txt`

3) Create the model: `python3 main.py` 
    - You also have to download the Coco Model, that you can find here: 
    https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

4) Test the model: `python3 test.py`


## Results

Original image

<img src="https://github.com/0x5eba/Skin-Cancer-Segmentation/blob/master/Nei/git.png" width="200" height="200">

Classify and Segment image

<img src="https://github.com/0x5eba/Skin-Cancer-Segmentation/blob/master/Nei/gitres.png" width="400" height="400">
