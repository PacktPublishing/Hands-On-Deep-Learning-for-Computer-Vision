# To run this script and see
# both original data set and the one
# with added noise install matplotlib with
#
# conda install matplotlib=2.2.3
#
# (never version of matplotlib don'
# currently work)
# Inspired by
# https://blog.keras.io/building-autoencoders-in-keras.html

from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

def get_noisy_mnist(noise_factor=0.5, show_samples=False):
    (x_train, x_labels), (x_test, x_labels) = mnist.load_data()
    if show_samples:
        print('A sample of original MNIST dataset:')
        print(x_train[0][0], x_labels[0])

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)

    if show_samples:
        print('A sample with addded noise:')
        print(x_train_noisy[0][0], x_labels[0])
    return x_train_noisy, x_train, x_test_noisy, x_test

def plot_mnist(x):
    n = 10
    plt.figure(figsize=(20, 2))
    for i in range(0, n):
        ax = plt.subplot(1, n, i+1)
        plt.imshow(x[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

if __name__ == '__main__':
    xn, x, xntest, xt = get_noisy_mnist(show_samples=True)
    plot_mnist(x)
    plot_mnist(xn)
