import torch
import time


def tricubic(x):
    y = torch.zeros_like(x)
    idx = (x >= -1) & (x <= 1)
    y[idx] = torch.pow(1.0 - torch.pow(torch.abs(x[idx]), 3), 3)
    return y


class Loess(object):

    @staticmethod
    def normalize_tensor(tensor):
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        return (tensor - min_val) / (max_val - min_val)

    def __init__(self, xx, yy):
        if torch.is_tensor(xx):
            self.n_xx = self.normalize_tensor(xx)
        else:
            self.n_xx = self.normalize_tensor(torch.tensor(xx))

        if torch.is_tensor(yy):
            self.n_yy = self.normalize_tensor(yy)
        else:
            self.n_yy = self.normalize_tensor(torch.tensor(yy))
        self.max_xx = torch.max(xx).item()
        self.min_xx = torch.min(xx).item()
        self.max_yy = torch.max(yy).item()
        self.min_yy = torch.min(yy).item()

    @staticmethod
    def get_min_range(distances, window):
        min_idx = torch.argmin(distances).item()
        n = len(distances)
        if min_idx == 0:
            return torch.arange(0, window)
        if min_idx == n-1:
            return torch.arange(n - window, n)

        min_range = [min_idx]
        while len(min_range) < window:
            i0 = min_range[0]
            i1 = min_range[-1]
            if i0 == 0:
                min_range.append(i1 + 1)
            elif i1 == n-1:
                min_range.append(i0 - 1)
            elif distances[i0-1] < distances[i1+1]:
                min_range.append(i0 - 1)
            else:
                min_range.append(i1 + 1)
            min_range.sort()
        return torch.tensor(min_range)

    @staticmethod
    def get_weights(distances, min_range):
        max_distance = torch.max(distances[min_range])
        weights = tricubic(distances[min_range] / max_distance)
        return weights

    def normalize_x(self, value):
        return (value - self.min_xx) / (self.max_xx - self.min_xx)

    def denormalize_y(self, value):
        return value * (self.max_yy - self.min_yy) + self.min_yy

    def estimate(self, x, window, use_matrix=False):
        n_x = self.normalize_x(x)
        distances = torch.abs(self.n_xx - n_x)
        min_range = self.get_min_range(distances, window)
        weights = self.get_weights(distances, min_range)

        if use_matrix:
            wm = torch.eye(window).mul(weights)
            xm = torch.ones((window, 2))
            xp = torch.tensor([[1.0], [n_x]])
            xm[:, 1] = self.n_xx[min_range]
            ym = self.n_yy[min_range]
            xmt_wm = torch.transpose(xm, 0, 1) @ wm
            beta = torch.pinverse(xmt_wm @ xm) @ xmt_wm @ ym
            y = (beta @ xp)[0]
        else:
            xx = self.n_xx[min_range]
            yy = self.n_yy[min_range]
            sum_weight = torch.sum(weights).item()
            sum_weight_x = torch.dot(xx, weights).item()
            sum_weight_y = torch.dot(yy, weights).item()
            sum_weight_x2 = torch.sum(torch.mul(torch.mul(xx, xx), weights))\
                .item()
            sum_weight_xy = torch.sum(torch.mul(torch.mul(xx, yy), weights))\
                .item()

            mean_x = sum_weight_x / sum_weight
            mean_y = sum_weight_y / sum_weight

            b = (sum_weight_xy - mean_x * mean_y * sum_weight) / \
                (sum_weight_x2 - mean_x * mean_x * sum_weight)
            a = mean_y - b * mean_x

            y = a + b * n_x
        return self.denormalize_y(y)


def main():
    xx = torch.tensor([0.5578196, 2.0217271, 2.5773252, 3.4140288, 4.3014084,
                       4.7448394, 5.1073781, 6.5411662, 6.7216176, 7.2600583,
                       8.1335874, 9.1224379, 11.9296663, 12.3797674, 13.2728619,
                       4.2767453, 15.3731026, 15.6476637, 18.5605355,
                       18.5866354, 18.7572812])
    yy = torch.tensor([18.63654, 103.49646, 150.35391, 190.51031, 208.70115,
                       213.71135, 228.49353, 233.55387, 234.55054, 223.89225,
                       227.68339, 223.91982, 168.01999, 164.95750, 152.61107,
                       160.78742, 168.55567, 152.42658, 221.70702, 222.69040,
                       243.18828])

    loess = Loess(xx, yy)

    for x in xx:
        xi = x.item()
        y = loess.estimate(xi, window=7, use_matrix=False)
        print(xi, y)


if __name__ == "__main__":
    start = time.time()

    main()

    end = time. time()
    print(end - start)
