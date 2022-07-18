import numpy as np
from scipy.special import kl_div


def ks_test(x: np.ndarray, y: np.ndarray) -> float:
    output = 0
    sum_x = 0
    sum_y = 0
    for i in range(x.size):
        sum_x += x[i]
        sum_y += y[i]
        output = max(output, abs(sum_x - sum_y))

    return output / x.size


def chi_square_distance(x: np.ndarray, y: np.ndarray) -> float:
    return 0.5 * np.sum(((x - y) ** 2) / (x + y))


def symmetric_kl_div(x: np.ndarray, y: np.ndarray) -> float:
    x += np.ones_like(x) * 0.001
    x /= np.sum(x)
    y += np.ones_like(x) * 0.001
    y /= np.sum(x)

    a = kl_div(x, y)
    b = kl_div(y, x)

    s = np.sum((a + b) / 2)
    return float(s) / x.size
