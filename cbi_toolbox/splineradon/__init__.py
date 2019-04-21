import numpy as np

import cbi_toolbox.csplineradon as cradon


def fft_trikernel(Nt, a, n1, n2, n3, h1, h2, h3, pad_fact):
    T = a / (Nt - 1)
    # TODO check this with Michael
    dnu = 1 / (T * (pad_fact * Nt - 1))
    nu = -1 / (2 * T) + np.arange(pad_fact * Nt) * dnu

    trikernel_hat = np.power(np.sinc(np.outer(nu, h1)), (n1 + 1)) * np.power(
        np.sinc(np.outer(nu, h2)), (n2 + 1)) * np.power(np.sinc(np.outer(nu, h3)), (n3 + 1))

    kernel = np.abs(np.fft.fft(trikernel_hat, axis=0))

    return kernel[0:Nt, :] / (T * Nt * pad_fact)


def get_kernel_table(Nt, n1, n2, h, s, angles, degree=True):
    pad_fact = 4
    angles = np.atleast_1d(angles)

    if degree:
        angles = np.deg2rad(angles)

    table = np.zeros((Nt, angles.size))

    h1 = np.abs(np.sin(angles) * h)
    h2 = np.abs(np.cos(angles) * h)

    a = np.max(h1 * (n1 + 1) / 2 + h2 * (n1 + 1) / 2 + s * (n2 + 1) / 2)

    table = fft_trikernel(Nt, a, n1, n1, n2, h1, h2, s, pad_fact)

    return table, a
