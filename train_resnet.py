import tensorflow as tf
import numpy as np
import os
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.python.keras.backend import clear_session
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.python.keras.resnet50 import ResNet50, preprocess_input
from scipy.misc import imresize

data_dir = 'images/'
contents = [item for item in os.listdir(data_dir) if not item.startswith('.')]
classes = [each for each in contents if os.path.isdir(data_dir + each)]

batch = []
labels = []

for each in classes:
    print("Starting {} images".format(each))
    class_path = data_dir + each
    files = os.listdir(class_path)
    for ii, file in enumerate(files, 1):
        img = tf.keras.preprocessing.image.load_img(os.path.join(class_path, file)).resize((224, 224), Image.ANTIALIAS)
        img = np.array(img)
        batch.append(img.reshape((1, 224, 224, 3)))
        labels.append(each)

codes = np.concatenate(batch)

lb = LabelBinarizer()
lb.fit(labels)
labels_vecs = lb.transform(labels)
labels_vecs = np.where(labels_vecs == 1)[1].reshape((-1, 1))

ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
 
train_idx, test_idx = next(ss.split(codes, labels))

X_train, y_train = codes[train_idx], labels_vecs[train_idx]
X_test, y_test = codes[test_idx], labels_vecs[test_idx]

print("There are {} train images and {} test images.".format(X_train.shape[0], X_test.shape[0]))
print('There are {} unique classes to predict.'.format(np.unique(y_train).shape[0]))

#One-hot encoding the labels
num_classes = 21
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

#Creating a checkpointer 
checkpointer = ModelCheckpoint(filepath='scratchmodel.best.hdf5', 
                               verbose=1,save_best_only=True)

#Loading the ResNet50 model with pre-trained ImageNet weights
model = ResNet50(weights='imagenet', include_top=False, input_shape=(200, 200, 3))

#Reshaping the training data
X_train_new = np.array([imresize(X_train[i], (200, 200, 3)) for i in range(0, len(X_train))]).astype('float32')

#Preprocessing the data, so that it can be fed to the pre-trained ResNet50 model. 
resnet_train_input = preprocess_input(X_train_new)

#Creating bottleneck features for the training data
train_features = model.predict(resnet_train_input)

#Saving the bottleneck features
np.savez('resnet_features_train', features=train_features)

#Reshaping the testing data
X_test_new = np.array([imresize(X_test[i], (200, 200, 3)) for i in range(0, len(X_test))]).astype('float32')

#Preprocessing the data, so that it can be fed to the pre-trained ResNet50 model.
resnet_test_input = preprocess_input(X_test_new)

#Creating bottleneck features for the testing data
test_features = model.predict(resnet_test_input)

#Saving the bottleneck features
np.savez('resnet_features_test', features=test_features)

model = Sequential()
model.add(GlobalAveragePooling2D(input_shape=train_features.shape[1:]))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])

model.fit(train_features, y_train, batch_size=32, epochs=10,
          validation_split=0.2, callbacks=[checkpointer], verbose=1, shuffle=True)

#Evaluate the model on the test data
score  = model.evaluate(test_features, y_test)

#Accuracy on test data
print('Accuracy on the Test Images: ', score[1])

clear_session()
