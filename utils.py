import math
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.special import softmax
from keras.utils import np_utils


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


def mix(a, b, r1, r2, fraction):
    a_indx = [i for i, x in enumerate(a) if r1 <= np.linalg.norm(x) <= r2]
    b_indx = [i for i, x in enumerate(b) if r1 <= np.linalg.norm(x) <= r2]

    for k in range(int(fraction * min(len(a_indx), len(b_indx)))):
        i = a_indx[k]
        j = b_indx[k]
        temp = np.copy(a[i])
        a[i] = b[j]
        b[j] = temp


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