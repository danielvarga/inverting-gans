import sys

import keras
from keras.layers import Dense, Flatten, Activation, Reshape, Input
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np


import clocks


def toroidal_sampler(batch_size, latent_dim):
    assert latent_dim % 2 == 0
    z_sample = np.random.normal(size=(batch_size, latent_dim))
    l2 = np.sqrt(z_sample[:, 0::2] ** 2 + z_sample[:, 1::2] ** 2)
    l22 = np.zeros_like(z_sample)
    # must be a nicer way but who cares
    l22[:,0::2] = l2
    l22[:,1::2] = l2
    z_sample /= l22
    return z_sample


class ClockDataGenerator(keras.utils.Sequence):
    def __init__(self, latent_dim, epoch_size, batch_size):
        assert latent_dim % 2 == 0 # direct product of circles
        self.latent_dim = latent_dim
        self.epoch_size = epoch_size
        self.batch_size = batch_size

    def __len__(self):
        return self.epoch_size // self.batch_size

    def __getitem__(self, index):
        latent_batch = toroidal_sampler(self.batch_size, self.latent_dim)
        imgs = []
        for latent_vec in latent_batch:
            params = np.arctan2(latent_vec[::2], latent_vec[1::2])
            imgs.append(clocks.clock(params))
        imgs = np.array(imgs)
        return imgs, latent_batch

    def on_epoch_end(self):
        pass


class DataGenerator(keras.utils.Sequence):
    def __init__(self, sampler, transform, latent_dim, epoch_size, batch_size):
        self.latent_dim = latent_dim
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.sampler = sampler
        self.transform = transform

    def __len__(self):
        return self.epoch_size // self.batch_size

    def __getitem__(self, index):
        z = self.sampler(size=(self.batch_size, self.latent_dim))
        return z, self.transform(z)

    def on_epoch_end(self):
        pass


def clock_test():
    cdg = ClockDataGenerator(4, 1, 10)
    imgs, latent_batch = cdg[0]
    print(imgs.shape, latent_batch.shape)
    # params_batch = np.arctan2(latent_batch[:, ::2], latent_batch[:, 1::2])
    latent_vec = latent_batch[0]
    params = np.arctan2(latent_vec[::2], latent_vec[1::2])
    print(params)
    plt.imshow(imgs[0])
    plt.show()


def main():
    d = 2
    x = Input(shape=(d, ))
    net = x
    net = Dense(100, activation="relu")(net)
    net = Dense(100, activation="relu")(net)
    y = Dense(2, activation="linear")(net)
    model = Model(x, y)
    model.compile(optimizer=Adam(lr=0.0001), loss='mse')

    epoch_size = 20000
    batch_size = 32
    latent_generator = DataGenerator(sampler=np.random.normal, transform=lambda x: x*x,
            latent_dim=d, epoch_size=epoch_size, batch_size=batch_size)

    model.fit_generator(generator=latent_generator, epochs=10, steps_per_epoch=epoch_size // batch_size)

    n = 50
    points = []
    for x in np.linspace(-2, 2, n):
        for y in np.linspace(-2, 2, n):
            points.append((x, y))
    points = np.array(points)
    images = model.predict(points, batch_size=batch_size)
    plt.scatter(images[:, 0], images[:, 1])
    plt.show()


main()
