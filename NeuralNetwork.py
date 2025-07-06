import numpy as np
import json
from keras.api.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def tanh(x):
    return np.tanh(x)

def tanhderiv(output):
    return 1 - (output ** 2)

def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)

images, labels = (x_train[:1000].reshape(1000, 28*28) / 255, y_train[:1000])
labelsVector = np.zeros((len(labels), 10))
    
for i in range(len(labels)):
    l = labels[i]
    labelsVector[i][l] = 1
labels = labelsVector

lr = 0.1
iterations = 100
hiddenSize = 512
packageSize = 128

weights1 = 0.02 * np.random.random((784, hiddenSize)) - 0.01
weights2 = 0.2 * np.random.random((hiddenSize, 10)) - 0.1

for i in range(iterations):
    for k in range(int(len(images) / packageSize)):
        packageStart = (k * packageSize)
        packageEnd = (k + 1) * packageSize

        layer0 = images[packageStart:packageEnd]
        layer1 = tanh(np.dot(layer0, weights1))
        dropout_mask = np.random.randint(2, size=layer1.shape)
        layer1 *= dropout_mask * 2
        layer2 = softmax(np.dot(layer1, weights2))

        layerDelta2 = (labels[packageStart:packageEnd] - layer2) / packageSize
        layerDelta1 = layerDelta2.dot(weights2.T) * tanhderiv(layer1)

        layerDelta1 *= dropout_mask

        weights2 += lr * layer1.T.dot(layerDelta2)
        weights1 += lr * layer0.T.dot(layerDelta1)

weights2 = weights2.T
weights1 = weights1.T

weights_data = {
    "weights_0_1": weights1.tolist(),
    "weights_1_2": weights2.tolist()
}

with open("weights.json", "w") as json_file:
    json.dump(weights_data, json_file)

