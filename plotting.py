import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.special import softmax
import numpy as np


CIFAR_LABELS = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}


def plot_points(points_list, r):
    axis_lim = r * 1.2
    plt.xlim(-axis_lim, axis_lim)
    plt.ylim(-axis_lim, axis_lim)
    plt.gca().set_aspect("equal", adjustable="box")
    for points, color in zip(points_list, ["blue", "red"]):
        plt.scatter(points[:, 0], points[:, 1], color=color, s=6)


def _plot_synthetic_predictions(model, X_train, y_train, title="Model predictions"):
    x, y = np.meshgrid(np.arange(-10, 10.4, 0.4), np.arange(-10, 10.4, 0.4))
    coords = np.array((x, y)).T.reshape((-1, 2))
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(title)
    plt.contourf(
        x,
        y,
        softmax(model.predict(coords), axis=1)[:, 1].reshape(x.shape),
        cmap=plt.cm.coolwarm,
        norm=Normalize(vmin=0, vmax=1, clip=True),
    )
    for clazz, color in zip([0, 1], ["blue", "red"]):
        points_idx = np.argmax(y_train, axis=1) == clazz
        plt.scatter(X_train[points_idx, 0],
                    X_train[points_idx, 1], color=color, s=3)


def plot_synthetic_results(X_train, y_train, results, title=None):
    colors = plt.cm.get_cmap('tab10').colors
    plt.figure(figsize=(15, 5))
    st = plt.suptitle(title, fontsize=18)
    temp_vals = results.keys()
    for i, temps in enumerate(temp_vals):
        plt.subplot(1, len(temps)+1, i+1)
        _plot_synthetic_predictions(
            results[temps]['model'], X_train, y_train, title=f'Model predicitons\n{temps}')
    plt.subplot(1, len(temps)+1, len(temps)+1)
    plt.title("Accuracy")
    for i, temps in enumerate(temp_vals):
        plt.plot(results[temps]['history']['val_accuracy'],
                 color=colors[i], label=f'Valid {temps}', )

    for i, temps in enumerate(temp_vals):
        plt.plot(results[temps]['history']['accuracy'],
                 '--', color=colors[i], label=f'Train {temps}')
    plt.xlabel('Epochs')
    plt.legend()
    plt.gca().set_aspect(1 / plt.gca().get_data_ratio())
    st.set_y(1.06)
    st.set_x(0.51)
    plt.show()


def plot_cifar(X, y, size=8):
    plt.figure(figsize=(15, 5))
    y = y.flatten()
    idx = np.random.permutation(len(X))[:size]
    X_sample = X[idx]
    y_sample = y[idx]
    for i, (im, label) in enumerate(zip(X_sample, y_sample)):
        plt.subplot(1, size, i + 1)
        plt.axis("off")
        plt.title(CIFAR_LABELS[label])
        plt.imshow(im)
    plt.show()
