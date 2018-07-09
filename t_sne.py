from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
palette = np.array(sns.color_palette("hls", 21))
from sklearn.utils import shuffle
X = np.concatenate([np.load('resnet_features_train.npy'),
                    np.load('resnet_features_test.npy')])
y = np.concatenate([np.load('resnet_labels_train.npy'),
                    np.load('resnet_labels_test.npy')])
y = y.tolist()
y = np.array([np.argmax(np.array(i)) for i in y])
print(y)

style_labels = list(np.loadtxt('label.txt', str, delimiter='\n'))
print(style_labels)
X = X.reshape((-1, 2048))

X, y = shuffle(X, y, random_state=0)

# X=X[0:1000]
# y=y[0:1000]
print(X.shape, y.shape)
X_tsne = TSNE(n_components=2, early_exaggeration=10.0,
              random_state=20180705).fit_transform(X)
#X_tsne = PCA().fit_transform(X)
print(X_tsne.shape)
import itertools

print('end')
# fig=plt.figure()
# ax=Axes3D(fig)
import matplotlib

markers = matplotlib.markers.MarkerStyle.filled_markers

markers = marker = itertools.cycle(markers)

f = plt.figure(figsize=(15, 5))
ax = plt.subplot(aspect='equal')
print(X_tsne)
for i in range(21):
    ax.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1],
               marker=next(markers), c=palette[i], label=style_labels[i])
plt.legend(loc=2, numpoints=1, ncol=2, fontsize=12, bbox_to_anchor=(1.05, 0.8))
ax.axis('off')
plt.savefig('t_sne.png')
plt.show()
