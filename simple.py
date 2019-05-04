import sys

import keras
import keras.backend as K
from keras.layers import Dense, Flatten, Activation, Reshape, Input, Lambda
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np


import clocks
import vis


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


# unlike clocks.clock, this outputs in (0, 1)
def clocks_latent_to_pixel(latent_batch):
    imgs = []
    for latent_vec in latent_batch:
        params = np.arctan2(latent_vec[::2], latent_vec[1::2])
        imgs.append(clocks.clock(params))
    imgs = np.array(imgs).astype(np.float32) / 255
    return imgs


class ClockDataGenerator(keras.utils.Sequence):
    def __init__(self, latent_dim, epoch_size, batch_size, mode=None):
        assert latent_dim % 2 == 0 # direct product of circles
        self.latent_dim = latent_dim
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        assert mode in ("xz", "zx", "xx", "zz")
        self.mode = mode

    def __len__(self):
        return self.epoch_size // self.batch_size

    def __getitem__(self, index):
        latent_batch = toroidal_sampler(self.batch_size, self.latent_dim)
        if self.mode == "zz":
            return latent_batch, latent_batch
        imgs = clocks_latent_to_pixel(latent_batch)
        if self.mode == "zx":
            return latent_batch, imgs
        elif self.mode == "xz":
            return imgs, latent_batch
        elif self.mode == "xx":
            return imgs, imgs
        else:
            assert False

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
    print(imgs.shape, latent_batch.shape, imgs.min(), imgs.max())
    # params_batch = np.arctan2(latent_batch[:, ::2], latent_batch[:, 1::2])
    latent_vec = latent_batch[0]
    params = np.arctan2(latent_vec[::2], latent_vec[1::2])
    print(params)
    plt.imshow(imgs[0])
    plt.show()


def toroid_loss(z):
    z_odd = z[:, 1::2]
    z_even = z[:, 0::2]
    loss = K.mean((z_odd ** 2 + z_even ** 2 - 1) ** 2, axis=1)
    return loss


def main():
    param_count = 2
    latent_dim = 2 * param_count
    epoch_size = 20000
    batch_size = 32
    img_size = 28

    x = Input(shape=(img_size, img_size, 3))
    net = Flatten()(x)
    net = Dense(100, activation="relu")(net)
    net = Dense(100, activation="relu")(net)
    net = Dense(100, activation="relu")(net)
    z_prime = Dense(latent_dim, activation="linear")(net)
    encoder = Model(x, z_prime)

    z = Input(shape=(latent_dim, ))
    net = z
    net = Dense(100, activation="relu")(net)
    net = Dense(100, activation="relu")(net)
    net = Dense(100, activation="relu")(net)
    net = Dense(100, activation="relu")(net)
    pixel_dim = img_size * img_size * 3
    net = Dense(pixel_dim, activation="sigmoid")(net)
    decoder_output = Reshape((img_size, img_size, 3))(net)
    decoder = Model(z, decoder_output)
    y = decoder(z_prime)

    tor_loss = toroid_loss(z_prime)

    def custom_loss(x, x_prime):
        return K.mean(K.square(x - x_prime)) + tor_loss

    def tor_metric(x, x_prime):
        return tor_loss

    autoencoder = Model(x, y)

    autoencoder.compile(optimizer=Adam(lr=0.0001), loss=custom_loss, metrics=['mse', tor_metric])
    clock_generator = ClockDataGenerator(latent_dim, epoch_size=epoch_size, batch_size=batch_size, mode="xx")
    autoencoder.fit_generator(generator=clock_generator, epochs=100, steps_per_epoch=epoch_size // batch_size)
    latent_points = toroidal_sampler(100, latent_dim)
    imgs = decoder.predict(latent_points)
    print(imgs.shape)
    # right now this is dumb, no constraint on latent space
    vis.plot_images(imgs, 10, 10, "generated")
    imgs, imgs = clock_generator[0]
    vis.display_reconstructed(autoencoder, imgs, "reconstructed")
    vis.flattorus_visualization(decoder, "flattorus")


def main_a_bit_less_old():
    param_count = 2
    latent_dim = 2 * param_count
    epoch_size = 20000
    batch_size = 32
    img_size = 28

    x = Input(shape=(latent_dim, ))
    net = x
    net = Dense(100, activation="relu")(net)
    net = Dense(100, activation="relu")(net)
    net = Dense(100, activation="relu")(net)
    net = Dense(100, activation="relu")(net)
    pixel_dim = img_size * img_size * 3
    y = Dense(pixel_dim, activation="sigmoid")(net)
    y = Reshape((img_size, img_size, 3))(y)
    model = Model(x, y)
    model.compile(optimizer=Adam(lr=0.0001), loss='mse')
    clock_generator = ClockDataGenerator(latent_dim, epoch_size=epoch_size, batch_size=batch_size, mode="zx")
    model.fit_generator(generator=clock_generator, epochs=5, steps_per_epoch=epoch_size // batch_size)
    latent_points = toroidal_sampler(100, latent_dim)
    imgs = model.predict(latent_points)
    vis.plot_images(imgs, 10, 10, "generated-hinted")

    fig=plt.figure(figsize=(8, 8))
    columns = 10
    rows = 10
    for i in range(columns * rows):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(imgs[i])
    plt.show()

    x = Input(shape=(img_size, img_size, 3))
    net = Flatten()(x)
    net = Dense(100, activation="relu")(net)
    net = Dense(100, activation="relu")(net)
    net = Dense(100, activation="relu")(net)
    y = Dense(latent_dim, activation="linear")(net)
    model = Model(x, y)
    model.compile(optimizer=Adam(lr=0.0001), loss='mse')
    clock_generator = ClockDataGenerator(latent_dim, epoch_size=epoch_size, batch_size=batch_size, mode="xz")
    model.fit_generator(generator=clock_generator, epochs=5, steps_per_epoch=epoch_size // batch_size)


def main_old():
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
# main_a_bit_less_old()
