####################################################################################
import os
import socket
from collections import OrderedDict
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import utils_tb
from net import Net
from utils_tb import Logging, RunBuilder

####################################################################################
# Definitions for training                                                         #
####################################################################################
hyperparameters = OrderedDict(batch_size=[10, 25, 50, 75, 100],
                              lr=[0.01, 0.001],
                              momentum=[0.75, 0.9],
                              total_epochs=[10])
# Descreve experimento
name_experiment = 'experimento_1'
description_experiment = 'Objetivo: testar o funcionamento do tensorboard.'

# Cria pasta e arquivo com descrição dos experimentos
experiment_details = {
    'main_dir_results': './runs',
    'name_experiment': name_experiment,
    'created_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'host': socket.gethostname(),
    'comments': description_experiment,
    'parameters': hyperparameters,
}
dir_tb = utils_tb.prepare_tensorboard_dir(experiment_details)

# Obtém todas combinações dos parametros para rodar experimentos
runs = RunBuilder.get_runs(hyperparameters)

for run in runs:
    dir_save_run = str(run).replace('Run', '').replace('(', '').replace(')', '')
    dir_save_run = os.path.join(dir_tb, dir_save_run)
    log_path = os.path.join(dir_save_run, 'log_history.txt')

    if os.path.isdir(dir_save_run):
        logging = Logging(log_path)
        logging.append_line(
            f'Attempt to run experiment {name_experiment} with parameters \n\t\t\t{run}:')
        logging.append_line('Skipping experiment.', show_datetime=False)
        logging.append_line('Reason: experiment has already been created with such parameters.',
                            show_datetime=False)
        continue

    # Tensorboard writer
    tb = SummaryWriter(dir_save_run)
    # Register log
    logging = Logging(log_path)
    logging.append_line(
        f'Attempt to run experiment {name_experiment} with parameters: \n\t\t\t{run}')
    logging.append_line('Experiment started:')

    # Hyperparameters defined for this run
    batch_size = run.batch_size
    lr = run.lr
    momentum = run.momentum
    total_epochs = run.total_epochs

    # Transforms
    transf_std = 0.5
    transf_mean = 0.5
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((transf_mean, ), (transf_std, ))])

    # Datasets
    trainset = torchvision.datasets.FashionMNIST('./data',
                                                 download=True,
                                                 train=True,
                                                 transform=transform)
    testset = torchvision.datasets.FashionMNIST('./data',
                                                download=True,
                                                train=False,
                                                transform=transform)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Define constants for classes
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker',
               'Bag', 'Ankle boot')

    net = Net()

    # Get random training batch and write it on tensorboard
    data_iter = iter(train_loader)
    images, labels = data_iter.next()
    # Create grid of images
    img_grid = torchvision.utils.make_grid(images, nrow=5)
    utils_tb.matplotlib_imshow(img_grid, transf_std, transf_mean, one_channel=True)

    # Show images
    # plt.show()
    # write to tensorboard
    tb.add_image('four_fashion_mnist_images', img_grid)
    # Inspect the model with tensorboard
    tb.add_graph(net, images)
    tb.close()

    # 4) Adding a projector to TensorBoard
    # select random images and their target indices
    images, labels = utils_tb.select_n_random(trainset.data, trainset.targets)
    # get the class labels for each image
    class_labels = [classes[lab] for lab in labels]
    # log embeddings
    features = images.view(-1, 28 * 28)
    tb.add_embedding(features, metadata=class_labels, label_img=images.unsqueeze(1))
    tb.close()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    # Evaluate metrics every 100 batches
    batches_evaluate = 1000
    running_loss = 0

    for epoch in range(total_epochs):  # loop over the dataset multiple times
        epoch_loss = 0
        epoch_correct_predictions = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # calculate metrics
            running_loss += loss.item()
            epoch_loss += loss.item()
            epoch_correct_predictions += utils_tb.get_num_correct(outputs, labels)
            # every 'batches_evaluate' mini-batches
            if i % batches_evaluate == batches_evaluate - 1:
                # store the running loss
                tb.add_scalar('training loss', running_loss / batches_evaluate,
                              epoch * len(train_loader) + i)
                logging.append_line(
                    f'mini-batch: {epoch * len(train_loader) + i} training_loss: {running_loss / batches_evaluate}'
                )
                # store a Matplotlib Figure showing the model's predictions on a random mini-batch
                tb.add_figure('predictions vs. actuals',
                              utils_tb.plot_classes_preds(net, inputs, labels, transf_std,
                                                          transf_mean, classes),
                              global_step=epoch * len(train_loader) + i)
                running_loss = 0.0

        # Metrics for the epoch
        accuracy = epoch_correct_predictions / len(trainset)
        accuracy_str = '%.2f%%' % (100 * accuracy)
        logging.append_line(
            f'epoch: {epoch} \t total_corect: {epoch_correct_predictions} ({accuracy_str}) \t loss: {epoch_loss}'
        )
        # Add scalars
        tb.add_scalar('epoch loss', epoch_loss, epoch)
        tb.add_scalar('epoch correct predictions', epoch_correct_predictions, epoch)
        tb.add_scalar('epoch accuracy', accuracy, epoch)
        # Add histograms
        tb.add_histogram('conv1.bias', net.conv1.bias, epoch)
        tb.add_histogram('conv1.weight', net.conv1.weight, epoch)
        tb.add_histogram('conv1.weight.grad', net.conv1.weight.grad, epoch)

    logging.append_line('Finished training with given parameters.')
