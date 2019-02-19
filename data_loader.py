"""
data_loader for the smnist_dataset

TODO
"""

import random
random.seed(619) # to make it random 
from deep_teaching_commons.data.fundamentals.mnist import Mnist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def load_smnist(n=10, one_hot_enc=False, flatten=False, normalized=True):
    """
    TODO
    """
    _, labels, _, _ = Mnist().get_all_data(one_hot_enc=False)
    indices = np.arange(0, 60000)
    random.shuffle(indices)
    values = [(j, labels[j]) for j in indices]
    indices_subset = [[v[0] for v in values if v[1] == j][:n]
                          for j in range(10)]
    flattened_indices = [i for sub in indices_subset for i in sub]
    random.shuffle(flattened_indices)
    
    x_train, y_train, test_images, test_labels = Mnist().get_all_data(one_hot_enc=one_hot_enc, flatten=flatten, normalized=normalized)
    train_images = np.array([x_train[j] for j in flattened_indices])
    train_labels = np.array([y_train[j] for j in flattened_indices])
    return train_images, train_labels, test_images, test_labels

def plot_mnist(elts, m, n):
    """
    Code from: https://github.com/mnielsen/rmnist/blob/master/plot_mnist.
    by Michael Nielson
    
    Plot MNIST images in an m by n table. Note that we crop the images
    so that they appear reasonably close together.  Note that we are
    passed raw MNIST data and it is reshaped.

    """
    fig = plt.figure()
    images = [elt.reshape(28, 28) for elt in elts]
    img = np.concatenate([np.concatenate([images[m*y+x] for x in range(m)], axis=1)
                          for y in range(n)], axis=0)
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(img, cmap = matplotlib.cm.binary)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()