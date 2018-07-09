import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def decode_predictions(preds, labels, top=5):
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [(labels[i], pred[i]) for i in top_indices]
        results.append(result)
    return results

labels = [line.strip() for line in open('label.txt', 'r')]

X_test = np.concatenate([np.load('resnet_features_train.npy'),
                    np.load('resnet_features_test.npy')])
y_test = np.concatenate([np.load('resnet_labels_train.npy'),
                    np.load('resnet_labels_test.npy')])
y_test = y_test.tolist()
y_test = np.array([np.argmax(np.array(i)) for i in y_test])

model = load_model('resnet_model.h5')
model.summary()

# score  = model.evaluate(X_test, y_test)
# print('Accuracy on the Test Images: ', score[1])


y_pred = model.predict_classes(X_test)
print(classification_report(y_test, y_pred, target_names=labels))
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index = labels, columns = labels)
print(df_cm)
df_cm.to_csv('cm')
plt.figure()
sn.heatmap(df_cm, annot=True)
plt.show()

# pred = model.predict(X_test)
# result = decode_predictions(pred, labels, 1)
# print(result)