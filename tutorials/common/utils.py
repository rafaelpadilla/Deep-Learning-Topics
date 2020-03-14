import os
import pickle
import tarfile
import zipfile

import numpy as np
from six.moves import urllib
from sklearn.model_selection import train_test_split


def fetch_dataset(url, destination_folder, decompress=False, del_file_afterwards=False, quiet=False):
    if not os.path.isdir(destination_folder):
        os.makedirs(destination_folder)
    file_name = url[url.rfind('/')+1:]
    destination_file = os.path.join(destination_folder, file_name)
    if not quiet:
        print(f'Downloading file from {url}')
    try:
        urllib.request.urlretrieve(url, destination_file)
        if not quiet:
            print(f'File successfully downloaded from {url}')
    except:
        print(f'Error downloading file from {url}')
        return False, ''
    if not decompress:
        return True, ''
    ending_folder = ''
    if file_name.endswith('.tar') or file_name.endswith('.tar.gz') or file_name.endswith('.tgz'):
        ending_folder = destination_file.replace('.tar.gz','/').replace('.tar','/').replace('.tgz','/')
        if not quiet:
            print(f'Decompressing {file_name} with tarfile into {ending_folder}.')
        tgz_file = tarfile.open(destination_file)
        tgz_file.extractall(path=destination_folder)
        tgz_file.close()
    elif file_name.endswith('.zip'):
        ending_folder = destination_file.replace('.zip','/')
        if not quiet:
            print(f'Decompressing {file_name} with zipfile into {ending_folder}')
        zip_file = zipfile.ZipFile(destination_file)
        zip_file.extractall(destination_folder)
        zip_file.close()
    else:
        if not quiet:
            print('Correct decompression tool not found.')
        if del_file_afterwards:
            a = os.remove(destination_file)
        if not quiet and a:
            print(f'File {file_name} deleted sucessfully.')
        return False, ending_folder

    if del_file_afterwards:
        try:
            os.remove(destination_file)
            if not quiet:
                print(f'File {file_name} deleted sucessfully.')
        except:
            print(f'Error deleting {file_name}.')
    return True, ending_folder



def unpickle(file_path, quiet=False):
    """
    Unpickle the given file and return the data.
    """
    if not quiet:
        print("Loading data: " + file_path)
    with open(file_path, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
    return data

def train_val_split(X, y, proportion_train=0.75, random_state=None, shuffle=True):
    """
    Split dataset into training and validation.
    """
    return train_test_split(X, y, train_size=float(proportion_train), random_state=random_state, shuffle=shuffle)

def get_mean_std(dataset, size=(32,32), channels=3):
    px_per_image = size[0]*size[1]
    sum_channels = np.zeros(channels)
    total_images = 0
    for d in dataset:
        total_images += len(dataset[d]['data'])
    all_values = np.zeros((channels, px_per_image*total_images))
    img_id_channel = np.zeros(channels, dtype=int)
    for d in dataset:
        reshaped = reshape_dataset(dataset[d]['data'], size, channels)
        for channel in range(channels):
            for img in reshaped:
                _from = img_id_channel[channel]*px_per_image
                _to = ((img_id_channel[channel]+1)*px_per_image)
                all_values[channel][_from:_to] = img[channel]
                img_id_channel[channel] += 1

    mean = [all_values[channel].mean() for channel in range(channels)]
    std = [all_values[channel].std() for channel in range(channels)]
    return mean, std

def reshape_dataset(data, size, channels):
    assert data.shape[1] == size[0]*size[1]*channels
    reshaped_data = []
    for im in data:
        # Split each image into channels
        reshaped_data.append(im.reshape((channels, size[0]*size[1])))
    return reshaped_data
# def preprocess(data):
