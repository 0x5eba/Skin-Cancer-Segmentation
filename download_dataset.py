'''
A script to download the ISIC Archive of lesion images
https://github.com/GalAvineri/ISIC-Archive-Downloader
'''

import argparse
import os
import requests
from os.path import join
from multiprocessing.pool import Pool, ThreadPool
from itertools import repeat
from tqdm import tqdm


from os.path import join
import shutil
import time
from urllib3.exceptions import ReadTimeoutError
from requests.exceptions import RequestException
import json
from PIL import Image

# The url template for the image is <base url prefix><image id><base url suffix>
# The url template for the description of the image is: <base url prefix><image id>
base_url_prefix = 'https://isic-archive.com/api/v1/image/'
base_url_suffix = '/download?contentDisposition=inline'


def main(num_images_requested, offset, filter, images_dir, descs_dir, num_processes):
    # If any of the images dir and descs dir don't exist, create them
    create_if_none(images_dir)
    create_if_none(descs_dir)

    if filter is None:
        print('Collecting the images ids')
        ids = get_images_ids(num_images=num_images_requested, offset=offset)

        num_images_found = len(ids)
        if num_images_requested is None or num_images_found == num_images_requested:
            print('Found {0} images'.format(num_images_requested))
        else:
            print('Found {0} images and not the requested {1}'.format(
                num_images_found, num_images_requested))

        print('Downloading descriptions')
        descriptions = download_descriptions(
            ids=ids, descs_dir=descs_dir, num_processes=num_processes)

    else:
        print('Collecting ids of all the images')
        ids = get_images_ids(num_images=None, offset=offset)

        print(
            'Downloading images descriptions, while filtering only {0} images'.format(filter))
        descriptions = download_descriptions_and_filter(ids=ids, num_images_requested=num_images_requested, filter=filter,
                                                        descs_dir=descs_dir)

        num_descs_filtered = len(descriptions)
        if num_images_requested is None or num_descs_filtered == num_images_requested:
            print('Found {0} {1} images'.format(num_images_requested, filter))
        else:
            print('Found {0} {1} images and not the requested {2}'.format(num_descs_filtered, filter,
                                                                          num_images_requested))

    print('Downloading images')
    download_images(descriptions=descriptions,
                    images_dir=images_dir, num_processes=num_processes)

    print('Finished downloading')


def create_if_none(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_images_ids(num_images, offset):
    """
    :param num_images: The number of requested images to download from the archive
    If None, will download the ids all the images in the archive
    :param offset: The offset from which to start downloading the ids of the images
    """
    # Specify the url that lists the meta data about the images (id, name, etc..)
    if num_images is None:
        num_images = 0
    url = 'https://isic-archive.com/api/v1/image?limit={0}&offset={1}&sort=name&sortdir=1'.format(
        num_images, offset)
    # Get the images metadata
    response = requests.get(url, stream=True)
    # Parse the metadata
    meta_data = response.json()
    # Extract the ids of the images
    ids = [meta_data[index]['_id'] for index in range(len(meta_data))]
    return ids


def download_descriptions(ids: list, descs_dir: str, num_processes: int) -> list:
    """
    :param ids:
    :param descs_dir:
    :param num_processes:
    :return: List of jsons
    """
    # Split the download among multiple processes
    pool = ThreadPool(processes=num_processes)
    descriptions = list(tqdm(pool.imap(download_and_save_description_wrapper, zip(
        ids, repeat(descs_dir))), total=len(ids), desc='Descriptions Downloaded'))
    return descriptions


def download_descriptions_and_filter(ids: list, num_images_requested: int, filter: str, descs_dir: str) -> list:
    """
    :param ids:
    :param num_images_requested:
    :param filter:
    :param descs_dir:
    :return: List of jsons
    """
    descriptions = []

    if num_images_requested is None:
        max_num_images = len(ids)
        pbar_desc = 'Descriptions Scanned'
    else:
        max_num_images = num_images_requested
        pbar_desc = '{0}s Found'.format(filter.title())

    pbar = tqdm(total=max_num_images, desc=pbar_desc)

    for id in ids:
        description = download_description(id)
        try:
            diagnosis = description['meta']['clinical']['benign_malignant']
        except KeyError:
            # The description doesn't have the a diagnosis. Skip it.
            continue

        if diagnosis == filter:
            # Save the description
            descriptions.append(description)
            save_description(description, descs_dir)

            if num_images_requested is not None:
                pbar.update(1)

            if num_images_requested is not None and len(descriptions) == num_images_requested:
                break

        if num_images_requested is None:
            pbar.update(1)

    pbar.close()

    return descriptions


def download_images(descriptions: list, images_dir: str, num_processes: int):
    # Split the download among multiple processes
    pool = Pool(processes=num_processes)
    tqdm(pool.map(download_and_save_image_wrapper, zip(descriptions, repeat(
        images_dir))), total=len(descriptions), desc='Images Downloaded')




def download_and_save_description_wrapper(args):
    return download_and_save_description(*args)


def download_and_save_description(id, descriptions_dir) -> list:
    """
    :param id: Id of the image whose description will be downloaded
    :param descriptions_dir:
    :return: Json
    """
    description = download_description(id)
    save_description(description, descriptions_dir)
    return description


def download_description(id) -> list:
    """
    :param id: Id of the image whose description will be downloaded
    :return: Json
    """
    # Build the description url
    url_desc = base_url_prefix + id

    # Download the image and description using the url
    # Sometimes their site isn't responding well, and than an error occurs,
    # So we will retry 10 seconds later and repeat until it succeeds
    while True:
        try:
            # Download the description
            response_desc = requests.get(url_desc, stream=True, timeout=20)
            # Validate the download status is ok
            response_desc.raise_for_status()
            # Parse the description
            parsed_description = response_desc.json()
            return parsed_description
        except RequestException:
            time.sleep(10)
        except ReadTimeoutError:
            time.sleep(10)
        except IOError:
            time.sleep(10)


def save_description(description, descriptions_dir):
    """
    :param description: Json
    :param descriptions_dir:
    :return:
    """
    desc_path = join(descriptions_dir, description['name'])
    with open(desc_path, 'w') as descFile:
        json.dump(description, descFile, indent=2)


def download_and_save_image_wrapper(args):
    download_and_save_image(*args)


def download_and_save_image(description, images_dir):
    """
    :param description: Json describing the image
    :param images_dir: Directory in which to save the image
    """
    # Build the image url
    url_image = base_url_prefix + description['_id'] + base_url_suffix

    # Download the image and description using the url
    # Sometimes their site isn't responding well, and than an error occurs,
    # So we will retry 10 seconds later and repeat until it succeeds
    while True:
        try:
            response_image = requests.get(url_image, stream=True, timeout=20)
            # Validate the download status is ok
            response_image.raise_for_status()

            # Write the image into a file
            img_path = join(images_dir, '{0}.jpg'.format(description['name']))
            with open(img_path, 'wb') as imageFile:
                shutil.copyfileobj(response_image.raw, imageFile)

            # Validate the image was downloaded correctly
            validate_image(img_path)
            return
        except RequestException:
            time.sleep(10)
        except ReadTimeoutError:
            time.sleep(10)
        except IOError:
            time.sleep(10)


def validate_image(image_path):
    # We would like to validate the image was fully downloaded and wasn't truncated.
    # To do so, we can open the image file using PIL.Image and try to resize it to the size
    # the file declares it has.
    # If the image wasn't fully downloaded and was truncated - an error will be raised.
    img = Image.open(image_path)
    img.resize(img.size)



parser = argparse.ArgumentParser()
parser.add_argument('--num_images', type=int, help='The number of images you would like to download from the ISIC Archive. '
                    'Leave empty to download all the available images', default=None)
parser.add_argument(
    '--offset', type=int, help='The offset of the image index from which to start downloading', default=0)
parser.add_argument('--filter', help='Indicates whether to download only benign or malignant images',
                    choices=['benign', 'malignant'], default=None)
parser.add_argument('--images-dir', help='The directory in which the images will be downloaded to',
                    default=join('Data', 'Images'))
parser.add_argument('--descs-dir', help='The directory in which the descriptions of '
                                        'the images will be downloaded to',
                    default=join('Data', 'Descriptions'))
parser.add_argument(
    '--p', type=int, help='The number of processes to use in parallel', default=16)
args = parser.parse_args()

main(num_images_requested=args.num_images, offset=args.offset, filter=args.filter, images_dir=args.images_dir, descs_dir=args.descs_dir,
        num_processes=args.p)
