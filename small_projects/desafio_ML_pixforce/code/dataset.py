####################################################################################################
# Pixforce                                                                                         #
# Desenvolvido por: Rafael Padilla (eng.rafaelpadilla@gmail.com)                                   #
# Última atualização: 11 julho 2020                                                                #
####################################################################################################

import os
import pickle

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose


class PixForceDataset(Dataset):
    def __init__(self, ds_file_path, transformations=None):
        # Verifica se é um arquivo válido
        assert os.path.isfile(ds_file_path), 'O arquivo {ds_file_path} não é um arquivo válido.'
        # Abre o arquivo e obtém amostras e seus rótulos
        self.ds_file_path = ds_file_path
        data = pickle.load(open(ds_file_path, 'rb'))
        self.images = [im.astype(np.uint8) for im in data['samples']]
        self.labels = data['labels']
        assert len(self.images) == len(
            self.labels
        ), 'A quantidade de amostras ({len(self.images)}) é diferente da quantidade de rótulos ({len(self.labels)})'
        # Transformações para serem aplicadas na imagem
        self.transformations = transformations

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Obtem amostra com rótulo
        imgs = self.images[idx]
        labels = self.labels[idx]
        # Aplica transformações
        if self.transformations:
            if isinstance(self.transformations, list):
                for transformation in self.transformations:
                    imgs = transformation(imgs)
            elif isinstance(self.transformations, Compose):
                imgs = self.transformations(imgs)
        return {'images': imgs, 'labels': labels}

    def add_transformation(self, new_transformation, begin=True):
        new_transformations = []
        # Inclui nova transformacao no começo
        if begin:
            new_transformations.append(new_transformation)
        if isinstance(self.transformations, list):
            for transformation in self.transformations:
                new_transformations.append(transformation)
            # Inclui nova transformacao no final
            if not begin:
                new_transformations.append(new_transformation)
        elif isinstance(self.transformations, Compose):
            for transformation in self.transformations.transforms:
                new_transformations.append(transformation)
            # Inclui nova transformacao no final
            if not begin:
                new_transformations.append(new_transformation)
            new_transformations = transforms.Compose(new_transformations)
        self.transformations = new_transformations
