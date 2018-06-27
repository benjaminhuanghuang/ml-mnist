'''
    Deep Learning with Python, Ch2

'''
from keras.datasets import mnist

# Need fix certificate verify failed:
# browse to Applications/Python 3.6 and double-click Install Certificates.command
# download data to ~/.keras/datasets/
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape)
print(len(train_labels))
print(test_labels)

'''
1. Build neural network and give training data to the neural network
'''
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

'''
2. Training
'''
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

'''
ask the network to produce predictions for
test_images, and we will verify if these predictions match the labels from test_labels.
'''

