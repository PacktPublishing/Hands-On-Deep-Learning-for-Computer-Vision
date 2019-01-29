"""
MNIST GAN that works well on CPU only and produce
consistent results (see conf.py for details).

Working GAN arch and parameters based code from
https://www.datacamp.com/community/tutorials/generative-adversarial-networks
"""
import conf
import os
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import initializers

class MNISTGan():
    def __init__(self):
        # The dimension of our random noise vector.
        self.random_dim = 100
        # Optimizer for all three networks.
        self.optimizer=Adam(lr=0.0002, beta_1=0.5)
        # Loading mnist data, we're using only training data.
        (self.x_train, self.y_train), _ = mnist.load_data()
        # Image vector size used as output size for Generator
        # and input size for Discriminator.
        # In case of MNIST data set we create we have a vector of size 28*28.
        self.ivs=self.x_train.shape[1]*self.x_train.shape[2]
        # Normalize our inputs to be in the range[-1, 1]
        self.x_train = (self.x_train.astype(np.float32) - 127.5)/127.5
        # Reshape to match our Discriminator input size.
        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1]*self.x_train.shape[2])
        # Create input and output networks, mix them togther
        # as one.
        self.generator=self.get_generator()
        self.discriminator=self.get_discriminator()
        self.gan=self.get_gan()

    def get_generator(self):
        generator = Sequential()
        generator.add(Dense(256, input_dim=self.random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        generator.add(LeakyReLU(0.2))
        generator.add(Dense(512))
        generator.add(LeakyReLU(0.2))
        generator.add(Dense(1024))
        generator.add(LeakyReLU(0.2))
        generator.add(Dense(self.ivs, activation='tanh'))
        generator.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        return generator

    def get_discriminator(self):
        discriminator = Sequential()
        discriminator.add(Dense(1024, input_dim=self.ivs, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))
        discriminator.add(Dense(512))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))
        discriminator.add(Dense(256))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))
        discriminator.add(Dense(1, activation='sigmoid'))
        discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        return discriminator

    def get_gan(self):
        # We initially set trainable to False since we only want to train either the
        # generator or discriminator at a time
        self.discriminator.trainable = False
        # gan input (noise) will be 100-dimensional vectors
        gan_input = Input(shape=(self.random_dim,))
        # the output of the generator (an image)
        x = self.generator(gan_input)
        # get the output of the discriminator (probability if the image is real or not)
        gan_output = self.discriminator(x)
        gan = Model(inputs=gan_input, outputs=gan_output)
        gan.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        return gan

    def get_noise(self, n):
        return np.random.normal(0, 1, size=[n, self.random_dim])

    def train_discriminator(self, batch_size):
        """
        Train discriminator with both fake and real data,
        in both cases we provide correct informations about
        the data.
        """
        # Get a random set of input noise.
        noise = self.get_noise(batch_size)

        # Generate fake MNIST images.
        generated_images = self.generator.predict(noise)
        # Get a random set of images from the actual MNIST dataset.
        image_batch = self.x_train[np.random.randint(0, self.x_train.shape[0], size=batch_size)]
        # Put them together in a single vector(list).
        X = np.concatenate([image_batch, generated_images])

        # Generate 0.0 (fake) for the whole vector.
        Y = np.zeros(2*batch_size)
        # Label real images corretly as 1.0.
        Y[:batch_size] = 1.0

        self.discriminator.trainable = True
        self.discriminator.train_on_batch(X, Y)

    def train_generator(self, batch_size):
        """
        Train generator with noise and label
        it as real data.
        """
        X = self.get_noise(batch_size)
        # Label noise as real data meaning as 1.0.
        Y = np.ones(batch_size)
        # Freeze discriminator to train generator only.
        self.discriminator.trainable = False
        self.gan.train_on_batch(X, Y)

    def save_gan_images(self, epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
        """
        Generate a sample of examples.
        """
        noise = self.get_noise(examples)
        generated_images = self.generator.predict(noise)
        generated_images = generated_images.reshape(examples, 28, 28)

        plt.figure(figsize=figsize)
        for i in range(generated_images.shape[0]):
            plt.subplot(dim[0], dim[1], i+1)
            plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
            plt.axis('off')
        plt.tight_layout()
        imgn='results/gan_mnist_%d.png' % epoch
        plt.savefig(imgn)
        return imgn

    def train(self, epochs, batch_size):
        """
        Train our GAN for a number of epochs (training sessions)
        and using batch_size samples in each epoch.

        Save a snapshot of learned mnist images at even epochs.
        """
        batch_count = self.x_train.shape[0] / batch_size

        for e in range(1, epochs+1):
            for i in range(int(batch_count)):
                print('Epoch %d of %d [ batch %d of %d ]' % (e, epochs, i+1, batch_count), end='\r', flush=True)
                self.train_discriminator(batch_size)
                self.train_generator(batch_size)

            if e == 1 or e % 2 == 0:
                imgn=self.save_gan_images(e)

if __name__ == '__main__':
    mg=MNISTGan()
    mg.train(epochs=50, batch_size=256)
