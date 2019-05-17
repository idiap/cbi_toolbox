import numpy as np


def fft_trikernel(nt, a, n1, n2, n3, h1, h2, h3, pad_fact):
    T = a / (nt - 1)
    dnu = 1 / (T * (pad_fact * nt - 1))
    nu = -1 / (2 * T) + np.arange(pad_fact * nt) * dnu

    trikernel_hat = np.power(np.sinc(np.outer(h1, nu)), (n1 + 1)) * np.power(
        np.sinc(np.outer(h2, nu)), (n2 + 1)) * np.power(np.sinc(np.outer(h3, nu)), (n3 + 1))

    kernel = np.abs(np.fft.fft(trikernel_hat, axis=1))

    return kernel[:, 0:nt] / (T * nt * pad_fact)


def get_kernel_table(nt, n1, n2, h, s, angles, degree=True):
    pad_fact = 4
    angles = np.atleast_1d(angles)

    if degree:
        angles = np.deg2rad(angles)

    h1 = np.abs(np.sin(angles) * h)
    h2 = np.abs(np.cos(angles) * h)

    a = np.max(h1 * (n1 + 1) / 2 + h2 * (n1 + 1) / 2 + s * (n2 + 1) / 2)

    table = fft_trikernel(nt, a, n1, n1, n2, h1, h2, s, pad_fact)

    return table, a
