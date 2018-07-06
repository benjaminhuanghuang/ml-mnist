'''
    https://blog.csdn.net/briblue/article/details/80398369


    Format of training images file
        0000 位置也是一个魔数  2051
        0004 代表了本文件中的图片数量
        0008 文件这个位置存放的是一张图片的高
        0012 文件这个位置存放的是一张图片的宽
        0016 从这里起，代表的是图像中的每一个像素点
    
    Read first image
        从文件起始位置偏移16个byte，然后读取后面的２８＊２８也就是 784　个字节

    Read Nth image
        从文件起始位置偏移 16＋(n-1)*784 个byte，然后读取后面的２８＊２８个字节．
    
    Format of label file
        0000 起始位置是一个魔数　数值为　2049
        0004 文件这个地方存放的数值是　6000 代表　6000 个标签
        0008 文件这个地方开始按顺序存放与训练图片对应的数字标签　数值　０～９
'''
from tensorflow.examples.tutorials.mnist import input_data

# 如果当前文件所在目录中，不存在 MNIST_data 这个目录的话，程序会自动下载 MNIST 数据到这个位置，如果已经存在了的话，就直接读取数据文件。
# mnist 是一个 dataset 类实例，里面有许多 numpy 数组，存放图片和标签．
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# There are 3 dataset in mnist
# MNIST database has 50000 traning image and 10000 testing image
# input_data function add 5000 validation into mnist when loading
mnist.train.images
mnist.train.labels

mnist.test.images
mnist.test.labels

mnist.validation.images
mnist.validation.labels

print(mnist.train.images.shape)  # (55000, 784) means 55000 rows and 784 cols
print(mnist.train.labels.shape)

# Get second image
image = mnist.train.images[1,:]
image = image.reshape(28,28)
print("Lable of second image")
print(mnist.train.labels[1]) # [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.] one-hot style, means value 3

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.figure()
plt.imshow(image)
plt.show()