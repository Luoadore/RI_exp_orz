import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import scipy

# read data
mnist = input_data.read_data_sets('/media/luo/result/RI/MNIST_data', one_hot=False)
print('read data down.')

# extract and save
"""
label_one = [0,0,0,0,0,0,1,0,0,0]
index = np.where((mnist.train.labels == label_one).all(1))[0]
train_data = mnist.train.images[index]
train_label = mnist.train.labels[index]
"""
if not os.path.exists('/media/luo/result/RI/MNIST_data/train/'):
    os.makedirs('/media/luo/result/RI/MNIST_data/train/')
if not os.path.exists('/media/luo/result/RI/MNIST_data/test/'):
    os.makedirs('/media/luo/result/RI/MNIST_data/test/')

train_data = mnist.train.images
train_label = mnist.train.labels
test_data = mnist.test.images
test_label = mnist.test.labels

train_dir = []
test_dir = []


for k, temp in enumerate(train_data):
    out = temp.reshape(28, 28)
    filename = "/media/luo/result/RI/MNIST_data/train/{}.png".format(str(k).zfill(3))
    scipy.misc.toimage(out).save(filename)
    train_dir.append([filename, str(train_label[k])])
    if k % 500 == 0:
        print('Save 500 samples.')
    #plt.imsave("real_image/{}.png".format(str(k).zfill(3)), temp.reshape(28, 28), format="png",cmap=plt.cm.gray)

for k, temp in enumerate(test_data):
    out = temp.reshape(28, 28)
    filename = "/media/luo/result/RI/MNIST_data/test/{}.png".format(str(k).zfill(3))
    scipy.misc.toimage(out).save(filename)
    test_dir.append([filename, str(test_label[k])])
    if k % 500 == 0:
        print('Save 500 samples.')

def write_txt(file_dir, data):
    with open(file_dir, 'a') as f:
        for each in data:
            f.write(' '.join(each) + '\n')
    print('write done.')

write_txt('/media/luo/result/RI/MNIST_data/train.txt', train_dir)
write_txt('/media/luo/result/RI/MNIST_data/test.txt', test_dir)