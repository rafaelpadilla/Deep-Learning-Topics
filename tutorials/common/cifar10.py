
import glob
import os
import pickle

import numpy as np

from utils import get_mean_std, train_val_split, unpickle


class Cifar10:
    def __init__(self, data_dir, random_seed=None):
        self.data_dir = data_dir
        self.random_seed = random_seed
        self.is_valid_path = self._validate_path()
        if not self.is_valid_path:
            print("ALERT: Path is not valid or does not contain all cifar 10 files.")
            return
        # Initializing variables
        self.mean_std_file = 'mean_std.npy' # Path with mean and std
        self.mean, self.std = None, None
        self.label_names = []

    def _validate_path(self):
        # Check if directory exists
        if not os.path.isdir(self.data_dir):
            return False
        # Check if there are 8 files
        databatch_files = glob.glob(os.path.join(self.data_dir, 'data_batch_*'))
        if len(databatch_files) != 5:
            return False
        # Check if  there is a batches.meta and a test_batch file
        if os.path.isfile(os.path.join(self.data_dir, 'batches.meta')) and os.path.isfile(os.path.join(self.data_dir, 'test_batch')):
           return True

    def get_labels(self):
        if not self.is_valid_path:
            print("ALERT: Path is not valid or does not contain all cifar 10 files.")
            return
        if self.label_names == []:
            # Unpickle batches.meta and get label names
            f = unpickle(os.path.join(self.data_dir, 'batches.meta'), quiet=True)
            self.label_names = [lb.decode() for lb in f[b'label_names']]
        return self.label_names

    def load(self, shuffle=False):
        if not self.is_valid_path:
            return
        # Load train/val files ('data_batch_X' -> X={1,2,3,4,5})
        files = [os.path.join(self.data_dir, 'data_batch_%d' %(i+1)) for i in range(5)]
        labels_train_val, raw_images_train_val = self._load_images_labels(files)
        # Load test file (test_batch)
        files = [os.path.join(self.data_dir, 'test_batch')]
        labels_test, raw_images_test = self._load_images_labels(files)
        # Split training and validation sets
        X_train, X_val, y_train, y_val = train_val_split(raw_images_train_val, labels_train_val, proportion_train=0.75, random_state=self.random_seed, shuffle=shuffle)
        train_set = {'labels': y_train, 'data': X_train}
        val_set = {'labels': y_val, 'data': X_val}
        # Create test set
        test_set = {'labels': labels_test, 'data': raw_images_test}
        # Now the dataset is lodaded, lets load the mean and std of the train/val data
        self._load_mean_std({"train": train_set, "validation": val_set})
        return {"train": train_set, "validation": val_set, "test": test_set}

    def _load_images_labels(self, files):
        raw_images = None
        labels = None
        for f in files:
            data = unpickle(f, quiet=True)
            # Get the raw images
            if raw_images is None:
                raw_images = data[b'data'].astype(dtype='float32')
            else:
                raw_images = np.vstack((raw_images, data[b'data'].astype(dtype='float32')))
            # Get the labels
            if labels is None:
                labels = data[b'labels']
            else:
                # Stack labels together
                labels = np.concatenate((labels, data[b'labels']))
        return labels, raw_images

    def _load_mean_std(self, dataset):
        full_path = os.path.join(self.data_dir, self.mean_std_file)
        # If mean and std have not been generated yet
        if os.path.isfile(full_path):
            mean_std = np.load(full_path)
            self.mean = mean_std.item().get('mean')
            self.std = mean_std.item().get('std')
        else:
            self.mean, self.std = get_mean_std(dataset, size=(32,32), channels=3)
            np.save(os.path.join(self.data_dir, self.mean_std_file), {'mean': self.mean, 'std': self.std})

    # def load(self, valid_ratio=0.0, one_hot=True, shuffle=False, dtype='float32'):
    # """
    # Loads CIFAR-10 pickled batch files, given the files' directory.
    # Optionally shuffles samples before dividing training and validation sets.
    # Can also apply global contrast normalization and ZCA whitening.

    # Arguments:
    #     data_dir: pickled batch files directory.
    #     valid_ratio: how much of the training data to hold for validation.
    #     Default: 0.
    #     one_hot: if True returns one-hot encoded labels, otherwise, returns
    #     integers. Default: True.
    #     shuffle: if True shuffles the data before splitting validation data.
    #     Default: False.
    #     gcn: if True applies global constrast normalization. Default: False.
    #     zca: if True applies ZCA whitening. Default: False.
    #     dtype: data type of image ndarray. Default: `float32`.

    # Returns:
    #     train_set: dict containing training data with keys `data` and `labels`.
    #     valid_set: dict containing validation data with keys `data` and `labels`.
    #     test_set: dict containing test data with keys `data` and `labels`.
    #     If zca == True, also returns
    #     mean: the computed mean values for each input dimension.
    #     whitening: the computed ZCA whitening matrix.
    #     For more information please see datasets.utils.zca_whitening.
    # """
    # assert valid_ratio < 1 and valid_ratio >= 0, 'valid_ratio must be in [0, 1)'
    # files = glob.glob(os.path.join(data_dir, 'data_batch_*'))
    # assert len(files) == 5, 'Could not find files!'
    # files = [os.path.join(data_dir, 'data_batch_%d' %(i+1)) for i in range(5)]
    # data_set = None
    # labels = None
    # # Iterate over the batches
    # for f_name in files:
    #     with open(f_name, 'rb') as f:
    #     # Get batch data
    #     batch_dict = pickle.load(f)
    #     if data_set is None:
    #     # Initialize the dataset
    #     data_set = batch_dict['data'].astype(dtype)
    #     else:
    #     # Stack all batches together
    #     data_set = np.vstack((data_set, batch_dict['data'].astype(dtype)))

    #     # Get the labels
    #     # If one_hot, transform all integer labels to one hot vectors
    #     if one_hot:
    #     batch_labels = one_hotify(batch_dict['labels'])
    #     else:
    #     # If not, just return the labels as integers
    #     batch_labels = np.array(batch_dict['labels'])
    #     if labels is None:
    #     # Initalize labels
    #     labels = batch_labels
    #     else:
    #     # Stack labels together
    #     labels = np.concatenate((labels, batch_labels), axis=0)

    # N = data_set.shape[0]
    # if shuffle:
    #     # Shuffle and separate between training and validation set
    #     new_order = np.random.permutation(np.arange(N))
    #     data_set = data_set[new_order]
    #     labels = labels[new_order]

    # # Get the number of samples on the training set
    # M = int((1 - valid_ratio)*N)
    # # Divide the samples
    # train_set, valid_set = {}, {}
    # # Reassing the data and reshape it as images
    # train_set['data'] = data_set[:M].reshape(
    #                     (-1, 3, 32, 32)).transpose((0, 2, 3, 1))
    # #train_set['data'] = data_set[:M]
    # train_set['labels'] = labels[:M]
    # valid_set['data'] = data_set[M:].reshape(
    #                     (-1, 3, 32, 32)).transpose((0, 2, 3, 1))
    # valid_set['labels'] = labels[M:]

    # test_set = {}
    # # Get the test set
    # f_name = os.path.join(data_dir, 'test_batch')
    # with open(f_name, 'rb') as f:
    #     batch_dict = pickle.load(f)
    #     test_set['data'] = batch_dict['data'].astype(dtype).reshape(
    #                     (-1, 3, 32, 32)).transpose((0, 2, 3, 1))
    #     if one_hot:
    #     test_set['labels'] = one_hotify(batch_dict['labels'])
    #     else:
    #     test_set['labels'] = np.array(batch_dict['labels'])

    # return train_set, valid_set, test_set

    # def get_labels(self):
    #     unpickle(self.file_path)


    # def preprocess(dataset):
    #     mean = np.array([125.3, 123.0, 113.9])
    #     std = np.array([63.0, 62.1, 66.7])

    #     dataset -= mean
    #     dataset /= std

    #     return dataset
