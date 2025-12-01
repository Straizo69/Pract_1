import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def visualize_classifier(classifier, X, y):

    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    mesh_x, mesh_y = np.meshgrid(
        np.arange(min_x, max_x, 0.01),
        np.arange(min_y, max_y, 0.01)
    )

    mesh_predictions = classifier.predict(
        np.c_[mesh_x.ravel(), mesh_y.ravel()])
    mesh_predictions = mesh_predictions.reshape(mesh_x.shape)

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFD580'])
    cmap_bold = ListedColormap(['#FF0000', '#00CC00', '#0000FF', '#FFA500'])

    plt.figure()
    plt.pcolormesh(mesh_x, mesh_y, mesh_predictions, cmap=cmap_light)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=80)
    plt.xlim(mesh_x.min(), mesh_x.max())
    plt.ylim(mesh_y.min(), mesh_y.max())
    plt.title("Decision boundaries")
    plt.show()
