# coding: utf-8
"""
Plot samples according different need.
"""

import matplotlib.pyplot as plt

def plot_zero_to_nine(images, labels):

    fig, ax = plt.subplots(
        nrows=2,
        ncols=5,
        sharex=True,
        sharey=True,
    )

    ax = ax.flatten()
    for i in range(10):
        img = images[labels == 1][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

def plot_nums_by_class(images, labels, label,nums):
    fig, ax = plt.subplots(
        nrows=2,
        ncols=5,
        sharex=True,
        sharey=True,
    )

    ax = ax.flatten()
    for i in range(nums):
        img = images[labels == label][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()