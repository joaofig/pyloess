import numpy as np
import math

from numba import jit, float64, int64


@jit(float64(float64), nopython=True)
def tricubic(x):
    if x <= -1.0 or x >= 1.0:
        return 0.0
    else:
        return 70.0 * math.pow(1.0 - math.pow(abs(x), 3), 3) / 81.0


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
        min_idx = np.argsort(distances)[:window]
        return np.sort(min_idx)

        # i_min = np.argmin(distances)
        # n = distances.shape[0]
        # i0 = max(i_min - 1, 0)
        # i1 = min(i_min + 1, n - 1)
        # while i1 - i0 + 1 < window:
        #     if distances[i0] < distances[i1]:
        #         if i0 > 0:
        #             i0 -= 1
        #         else:
        #             i1 += 1
        #     else:
        #         if i1 < n - 1:
        #             i1 += 1
        #         else:
        #             i1 -= 1
        # return np.array([i0, i1])

    @staticmethod
    # @jit(float64[:](float64[:], int64[:]), nopython=True)
    def get_weights(distances, min_range):
        n = min_range.shape[0]
        max_distance = np.max(distances[min_range])
        weights = np.zeros(n)

        for i in range(n):
            weights[i] = tricubic(distances[min_range[0] + i] / max_distance)
        return weights

    def normalize_x(self, value):
        return (value - self.min_xx) / (self.max_xx - self.min_xx)

    def denormalize_y(self, value):
        return value * (self.max_yy - self.min_yy) + self.min_yy

    # @jit
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

        for i in min_range:
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
    xx = np.array([0.5578196, 2.0217271, 2.5773252, 3.4140288, 4.3014084, 4.7448394, 5.1073781,
                   6.5411662, 6.7216176, 7.2600583, 8.1335874, 9.1224379, 11.9296663, 12.3797674,
                   13.2728619, 14.2767453, 15.3731026, 15.6476637, 18.5605355, 18.5866354, 18.7572812])
    yy = np.array([18.63654, 103.49646, 150.35391, 190.51031, 208.70115, 213.71135, 228.49353,
                   233.55387, 234.55054, 223.89225, 227.68339, 223.91982, 168.01999, 164.95750,
                   152.61107, 160.78742, 168.55567, 152.42658, 221.70702, 222.69040, 243.18828])

    loess = Loess(xx, yy)

    y = loess.estimate(1.0, window=5)
    print(y)


if __name__ == "__main__":
    main()
