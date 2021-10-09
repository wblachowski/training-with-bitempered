import math
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.special import softmax
from keras.utils import np_utils


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


def generate_points(n, r1, r2):
    points = []
    for _ in range(n):
        d = random.uniform(r1, r2)
        a = math.radians(random.randint(0, 360))
        points.append([d * math.sin(a), d * math.cos(a)])
    return np.array(points)


def points_to_data(blue, red):
    X = np.concatenate((blue, red))
    y = np.concatenate((np.full(len(blue), 0), np.full(len(red), 1)))
    y = np_utils.to_categorical(y)
    return X, y


def plot_points(points_list, r):
    axis_lim = r * 1.2
    plt.xlim(-axis_lim, axis_lim)
    plt.ylim(-axis_lim, axis_lim)
    plt.gca().set_aspect("equal", adjustable="box")
    for points, color in zip(points_list, ["blue", "red"]):
        plt.scatter(points[:, 0], points[:, 1], color=color, s=6)


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


def mix_points(a, b, r1, r2, fraction):
    a_indx = [i for i, x in enumerate(a) if r1 <= np.linalg.norm(x) <= r2]
    b_indx = [i for i, x in enumerate(b) if r1 <= np.linalg.norm(x) <= r2]

    for k in range(int(fraction * min(len(a_indx), len(b_indx)))):
        i = a_indx[k]
        j = b_indx[k]
        temp = np.copy(a[i])
        a[i] = b[j]
        b[j] = temp


def mix_cifar(y, percentage):
    y_mixed = np.copy(y)
    idx = np.random.permutation(len(y))[: int(percentage * len(y))]
    y_mixed[idx] = (y_mixed[idx] + np.random.randint(1, 10)) % 10
    return y_mixed, idx


def plot_predictions(model, X_train, Y_train, title="Model predictions"):
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
        points_idx = np.argmax(Y_train, axis=1) == clazz
        plt.scatter(X_train[points_idx, 0], X_train[points_idx, 1], color=color, s=3)