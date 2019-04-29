import numpy as np
import math

from numba import jit, float64, int64


@jit(float64(float64), nopython=True)
def tricubic(x):
    if x <= -1.0 or x >= 1.0:
        return 0.0
    else:
        return math.pow(1.0 - math.pow(abs(x), 3), 3)


class Loess(object):

    @staticmethod
    @jit(float64[:](float64[:]), nopython=True)
    def normalize_array(array):
        min_val = np.min(array)
        max_val = np.max(array)
        return (array - min_val) / (max_val - min_val)

    def __init__(self, xx, yy):
        self.n_xx = self.normalize_array(xx)
        self.n_yy = self.normalize_array(yy)
        self.max_xx = np.max(xx)
        self.min_xx = np.min(xx)
        self.max_yy = np.max(yy)
        self.min_yy = np.min(yy)

    @staticmethod
    @jit(float64(float64[:], int64[:]), nopython=True)
    def get_max_distance(distances, rng):
        return np.max(distances[rng[0]:rng[1] + 1])

    @staticmethod
    @jit(int64[:](float64[:], int64), nopython=True)
    def get_min_range(distances, window):
        i_min = np.argmin(distances)
        n = distances.shape[0]
        i0 = max(i_min - 1, 0)
        i1 = min(i_min + 1, n - 1)
        for i in range(window):
            if distances[i0] < distances[i1] and i0 > 0:
                i0 = i0 - 1
            elif distances[i0] > distances[i1] and i1 < n - 1:
                i1 = i1 + 1
            else:
                break
        return np.array([i0, i1])

    @staticmethod
    @jit(float64[:](float64[:], int64[:]), nopython=True)
    def get_weights(distances, min_range):
        n = min_range[1] - min_range[0] + 1
        max_distance = np.max(distances[min_range[0]:min_range[1] + 1])
        weights = np.zeros(n)

        for i in range(n):
            weights[i] = tricubic(distances[min_range[0] + i] / max_distance)
        return weights

    def normalize_x(self, value):
        return (value - self.min_xx) / (self.max_xx - self.min_xx)

    def denormalize_y(self, value):
        return value * (self.max_yy - self.min_yy) + self.min_yy

    @jit
    def estimate(self, x, window):
        n_x = self.normalize_x(x)
        distances = np.abs(self.n_xx - n_x)
        min_range = self.get_min_range(distances, window)
        weights = self.get_weights(distances, min_range)

        sum_weight = 0.0
        sum_weight_x = 0.0
        sum_weight_y = 0.0
        sum_weight_x2 = 0.0
        sum_weight_xy = 0.0

        for i in range(min_range[0], min_range[1] + 1):
            w = weights[i - min_range[0]]
            sum_weight += w
            sum_weight_x += self.n_xx[i] * w
            sum_weight_y += self.n_yy[i] * w
            sum_weight_x2 += self.n_xx[i] * self.n_xx[i] * w
            sum_weight_xy += self.n_xx[i] * self.n_yy[i] * w

        mean_x = sum_weight_x / sum_weight
        mean_y = sum_weight_y / sum_weight

        b = (sum_weight_xy - mean_x * mean_y * sum_weight) / \
            (sum_weight_x2 - mean_x * mean_x * sum_weight)
        a = mean_y - b * mean_x

        y = a + b * n_x
        return self.denormalize_y(y)


def main():
    xx = np.array([1544326239000.0, 1544326247000.0, 1544326257766.0,
                   1544326261000.0, 1544326275000.0, 1544326283000.0,
                   1544326281000.0])
    yy = np.array([1600.0, 1750.0, 1700.0, 1600.0, 1800.0, 1550.0, 1650.0])

    loess = Loess(xx, yy)

    y = loess.estimate(1544326247000.0, window=5)
    print(y)


if __name__ == "__main__":
    main()
