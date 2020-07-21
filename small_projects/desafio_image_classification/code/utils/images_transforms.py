####################################################################################################
# Pixforce                                                                                         #
# Desenvolvido por: Rafael Padilla (eng.rafaelpadilla@gmail.com)                                   #
# Última atualização: 11 julho 2020                                                                #
####################################################################################################


import random

import numpy as np
import torch as th


class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def add_noise(self, sample):
        if isinstance(sample, np.ndarray):
            noise = np.random.normal(self.mean, self.std, sample.shape)
            # Mantém o formato do dado inicial
            return np.clip((sample + noise).astype(sample.dtype), 0, 255)

    def __call__(self, samples):
        if isinstance(samples, list):
            ret = []
            for sample in samples:
                sample = self.add_noise(sample)
                ret.append(sample)
            return ret


class SaltAndPepper(object):
    def __init__(self, prob_noise=.5):
        self.prob_noise = prob_noise

    def add_noise(self, sample):
        if isinstance(sample, np.ndarray):
            # Criando matriz de ruídos (0: pimenta, 255: sal, 1: sem mudanças )
            noise = np.random.choice([0, 255, 1], size=sample.shape[0:2], p=[self.prob_noise/2,
                                     self.prob_noise/2, 1-self.prob_noise])
            # Replica o mesmo ruído para todos os canais
            noise = np.stack((noise,)*3, axis=-1)
            # Mantém o formato do dado inicial
            return np.clip(noise*sample, 0, 255).astype(sample.dtype)

    def __call__(self, samples):
        if isinstance(samples, list):
            ret = []
            for sample in samples:
                sample = self.add_noise(sample)
                ret.append(sample)
            return ret


class Flatten(object):
    def __call__(self, samples):
        if isinstance(samples, list):
            ret = []
            for sample in samples:
                sample = sample.flatten()
                ret.append(sample)
            return ret
