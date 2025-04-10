# An MNIST loader.

import numpy as np
import gzip
import struct

def load_images(filename):
    with open(filename, 'rb') as f:
        _ignored, n_images, columns, rows = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(n_images, rows, columns)
    return images

# def load_images(filename):
#     # Open and unzip the file of images:
#     with gzip.open(filename, 'rb') as f:
#         # Read the header information into a bunch of variables:
#         _ignored, n_images, columns, rows = struct.unpack('>IIII', f.read(16))
#         # Read all the pixels into a NumPy array of bytes:
#         all_pixels = np.frombuffer(f.read(), dtype=np.uint8)
#         # Reshape the pixels into a matrix where each line is an image:
#         return all_pixels.reshape(n_images, columns * rows)


# 60000 images, each 784 elements (28 * 28 pixels)
X_train = load_images("train-images-idx3-ubyte")

# 10000 images, each 784 elements, with the same structure as X_train
X_test = load_images("t10k-images-idx3-ubyte")


def load_labels(filename):
    # Open and unzip the file of images:
    with gzip.open(filename, 'rb') as f:
        # Skip the header bytes:
        f.read(8)
        # Read all the labels into a list:
        all_labels = f.read()
        # Reshape the list of labels into a one-column matrix:
        return np.frombuffer(all_labels, dtype=np.uint8).reshape(-1, 1)


def one_hot_encode(Y):
    n_labels = Y.shape[0]
    n_classes = 10
    encoded_Y = np.zeros((n_labels, n_classes))
    for i in range(n_labels):
        label = Y[i]
        encoded_Y[i][label] = 1
    return encoded_Y


# 60K labels, each a single digit from 0 to 9
Y_train_unencoded = load_labels("train-labels-idx1-ubyte")

# 60K labels, each consisting of 10 one-hot encoded elements
Y_train = one_hot_encode(Y_train_unencoded)

# 10000 labels, each a single digit from 0 to 9
Y_test = load_labels("t10k-labels-idx1-ubyte")
