import os
from collections import OrderedDict, namedtuple
from datetime import datetime
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


class RunBuilder():
    @staticmethod
    def get_runs(params):
        run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(run(*v))
        return runs


class Logging():
    def __init__(self, file_path):
        self.file_path = file_path
        self.append_line(100 * '=', show_datetime=False)

    def append_line(self, text, show_datetime=True):
        f = open(self.file_path, 'a')
        line = f'{text}\n'
        if show_datetime:
            line = ('(%s) ' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'))) + line
        else:
            line = ' ' * 22 + line
        f.write(line)
        print(line[0:len(line) - 2])
        f.close()


def prepare_tensorboard_dir(experiment_details):
    dir_save_experiment = os.path.join(experiment_details['main_dir_results'],
                                       experiment_details['name_experiment'])
    # assert not os.path.isdir(
    #     dir_save_experiment
    # ), 'There is already one experiment saved with the name %s in the folder %s' % (
    #     experiment_details['name_experiment'], dir_save_experiment)
    if os.path.isdir(dir_save_experiment):
        return dir_save_experiment
    os.makedirs(dir_save_experiment)
    print(f'Directory with results created: {dir_save_experiment}')
    f = open(os.path.join(dir_save_experiment, 'comments.txt'), 'w')
    for k, v in experiment_details.items():
        if type(v) == OrderedDict:
            f.write(f'\n{k}:')
            for k1, v1 in v.items():
                f.write(f'\n\t{k1}: {v1}')
        else:
            f.write(f'\n{k}: {v}')
    f.close()
    return dir_save_experiment


def matplotlib_imshow(img, transf_std, transf_mean, one_channel=False):
    if one_channel:
        transf_std = [transf_std]
        transf_mean = [transf_mean]
    # unnormalize
    img = (img * torch.FloatTensor(transf_std).unsqueeze(1).unsqueeze(1)
           ) + torch.FloatTensor(transf_mean).unsqueeze(1).unsqueeze(1)
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg[0], cmap='Greys')
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


# Before training the model, lets define some helper functions
def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels, transf_std, transf_mean, classes):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    total_images = len(images)
    for idx in np.arange(total_images):
        ax = fig.add_subplot(1, total_images, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], transf_std, transf_mean, one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(classes[preds[idx]], probs[idx] * 100.0,
                                                          classes[labels[idx]]),
                     color=("green" if preds[idx] == labels[idx].item() else "red"))
    return fig


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]
