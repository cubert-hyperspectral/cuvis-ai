import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA as sk_pca


def visualize_image(preds: np.ndarray, title="Empty"):
    plt.imshow(preds, cmap="viridis")  # You can choose any colormap you like
    plt.colorbar()
    plt.title(title)
    plt.axis("off")
    plt.show()


def visualize_features(data: np.ndarray, labels=None, title="Empty"):
    # compress down to 3 max dimensions

    n_comps = max(3, data.shape[2])

    n_pixels = data.shape[0] * data.shape[1]
    image_2d = data.reshape(n_pixels, -1)
    pca = sk_pca(n_components=n_comps)
    pca.fit(image_2d)

    compressed = pca.transform(image_2d)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(compressed[:, 0], compressed[:, 1], compressed[:, 2], c=labels)
    plt.title(title)
    plt.axis("off")
    plt.show()
