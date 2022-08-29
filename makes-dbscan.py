from cProfile import label
from sre_constants import NOT_LITERAL_UNI_IGNORE
from turtle import color, title
from sklearn.datasets import make_blobs, make_moons, make_circles
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from matplotlib.widgets import Slider, Button


X, y = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=0)

# Plotting data
fig, axs = plt.subplots(2, 2, figsize=(20, 10))
fig.suptitle(f"Plotting DBSCAN")

axs[0, 0].scatter(X[:, 0], X[:, 1], c=y, marker=".")
axs[0, 0].set_title(f"Grund Truth : make_blobs data", color='C0')


def dbscan_clustering(eps, min_samples):
    # Erstelle DBSCAN
    dbscan_clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    cluster_labels = dbscan_clustering.fit_predict(X)
    axs[0, 1].scatter(X[:, 0], X[:, 1], c=cluster_labels, marker='.')
    axs[0, 1].set_title(
        f"DBSCAN Clustering, eps: {eps}, min samples: {min_samples}", color='C15')


def xaligned_axes(ax, y_distance, width, **kwargs):
    return plt.axes([ax.get_position().x0,
                     ax.get_position().y0-y_distance,
                     ax.get_position().width, width],
                    **kwargs)


# set axes color
axcolor = 'lightgoldenrodyellow'

# get best position for slider in subplots
ax_eps = xaligned_axes(ax=axs[0, 1], y_distance=0.05,
                       width=0.03, facecolor=axcolor)
ax_min_samples = xaligned_axes(
    ax=axs[0, 1], y_distance=0.07, width=0.03, facecolor=axcolor)


# Slider

allowed_eps = np.arange(0.1, 1.1, 0.1)
eps_slider = Slider(ax=ax_eps, label='eps', valmin=0.1,
                    valmax=1, valinit=0.5, valstep=allowed_eps, orientation='horizontal')

allowed_min_samples = range(1, 11)
min_samples_slider = Slider(ax=ax_min_samples, label='min_samples', valmin=1, valmax=10,
                            valinit=4, valstep=allowed_min_samples, orientation='horizontal')

# The function to be called anytime a slider's value changes


def update(val):
    dbscan_clustering(eps_slider.val, min_samples_slider.val)
    fig.canvas.draw_idle()


# register the update function with each slider
eps_slider.on_changed(update)
min_samples_slider.on_changed(update)

plt.show()
