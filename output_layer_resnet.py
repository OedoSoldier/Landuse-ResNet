import h5py
import tensorflow as tf
import numpy as np
import sys
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img
from tensorflow.python.keras import backend as K
from scipy.misc import imresize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import sqrt, floor, ceil


def plot_layer(model):
    layer_num = len(model.layers)
    print(layer_num)
    x = floor(sqrt(layer_num))
    y = ceil(layer_num / x)
    for i in range(layer_num):
        extract_layer = K.function([model.layers[0].input], [model.layers[i].output])
        f = extract_layer([img])[0]
        print(f.shape)
        show_img = f[:, :, :, 1]
        show_img.shape = (f.shape[1], f.shape[2])
        plt.subplot(x, y, i + 1)
        plt.imshow(show_img, cmap='gray')
        plt.axis('off')
    plt.show()

def plot_feature(model, layer):
    extract_layer = K.function([model.layers[0].input], [model.layers[layer].output])
    f = extract_layer([img])[0]
    print(f.shape)
    x = floor(sqrt(f.shape[-1]))
    y = ceil(f.shape[-1] / x)
    print(x, y)
    for i in range(f.shape[-1]):
        show_img = f[:, :, :, i]
        show_img.shape = (f.shape[1], f.shape[2])
        plt.subplot(x, y, i + 1)
        plt.imshow(show_img, cmap='gray')
        plt.axis('off')
    plt.show()

def plot_filter(model, input_img):
    extract_layer = K.function([model.layers[0].input], [model.layers[1].output])
    f = extract_layer([input_img])[0]
    print(f.shape)

    for i in range(64):
        show_img = f[:, :, :, i]
        show_img.shape = (f.shape[1], f.shape[2])
        plt.subplot(8, 8, i + 1)
        plt.imshow(show_img, cmap='gray')
        plt.axis('off')
    plt.show()

def decode_predictions(preds, labels, top=5):
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [(labels[i], pred[i]) for i in top_indices]
        results.append(result)
    return results

def show_labes(image, pred, name):
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    ax1 = plt.subplot(gs[0])
    x = [pred[i][0] for i in range(len(pred))][::-1]
    y = [pred[i][1] * 100 for i in range(len(pred))][::-1]
    print(x, y)
    colors=['#edf8fb','#b2e2e2','#66c2a4','#2ca25f','#006d2c']
    width = 0.4
    ind = np.arange(len(y))
    ax1.barh(ind, y, width, align='center', color=colors)
    ax1.set_yticks(ind+width/2)
    ax1.set_yticklabels(x, minor=False, fontsize=10)
    for i, v in enumerate(y):
        ax1.text(v + 1, i, '%5.2f%%' % v)
    plt.title('Probability Output')
    ax2 = plt.subplot(gs[1])
    ax2.axis('off')
    ax2.imshow(image)
    plt.title(name)
    plt.show()

def get_weights(epoch,logs):
    wsAndBs = model.layers[indexOfTheConvLayer].get_weights()
    #or model.get_layer("layerName").get_weights()

    weights = wsAndBs[0]
    biases = wsAndBs[1]
    #do what you need to do with them
    #you can see the epoch and the logs too: 
    print("end of epoch: " + str(epoch)) # for instance

image_name = sys.argv[1]
label_name = sys.argv[2]

labels = [line.strip() for line in open(label_name, 'r')]

image = np.array(load_img(image_name))
img = preprocess_input(np.array([imresize(image, (200, 200, 3))]).astype('float32'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
# model.summary()
input_img = model.input
print(input_img.shape)

for i in range(len(model.layers[:10])):
    print(model.layers[i].name)
    weights = model.layers[i].get_weights()
    print([i.shape for i in weights])
    # plot_feature(model, i)
    plot_filter(model, input_img)

bottleneck_feature = model.predict(img)

print(bottleneck_feature.shape)

model = load_model('resnet_model.h5')
# model.summary()

pred = model.predict(bottleneck_feature)
result = decode_predictions(pred, labels, 5)
print(result[0][0])

# show_labes(image, result[0], image_name)