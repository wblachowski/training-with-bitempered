import math
import random

import numpy as np
from tensorflow.keras.utils import to_categorical


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
    y = to_categorical(y)
    return X, y


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


def get_best_lr(finder):
    def valley(lrs: list, losses: list, num_it: int):
        "Suggests a learning rate from the longest valley and returns its index"
        n = len(losses)
        max_start, max_end = 0, 0

        # find the longest valley
        lds = [1]*n
        for i in range(1, n):
            for j in range(0, i):
                if (losses[i] < losses[j]) and (lds[i] < lds[j] + 1):
                    lds[i] = lds[j] + 1
                if lds[max_end] < lds[i]:
                    max_end = i
                    max_start = max_end - lds[max_end]

        sections = (max_end - max_start) / 3
        idx = max_start + int(sections) + int(sections/2)

        return float(lrs[idx]), (float(lrs[idx]), losses[idx])
    return valley(finder.lrs, finder.losses, len(finder.lrs))[1]
