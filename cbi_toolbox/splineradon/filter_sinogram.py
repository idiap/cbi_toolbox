import numpy as np


def get_filter(n, filter_name: str, degree):
    nu = np.concatenate((np.arange((n + 1) / 2), np.arange(-(n / 2 - 1), 0)))
    n0 = 3

    k = 50
    k_vector = np.arange(-k, k + 0.5)
    k_vector = k_vector[..., np.newaxis]

    filter_name = filter_name.upper()
    pre_filter = False

    if filter_name == 'NONE':
        filter_f = np.ones(nu.shape)
        pre_filter = True

    elif filter_name == 'RAM-LAK':
        filter_f = np.abs(nu)
        pre_filter = True

    elif filter_name == 'SHEPP-LOGAN':
        filter_f = np.abs(nu) * np.sinc(nu)
        pre_filter = True

    elif filter_name == 'COSINE':
        filter_f = np.abs(nu) * np.cos(np.pi * nu)
        pre_filter = True

    elif filter_name == 'B-SPLINE':
        filter_f = np.abs(nu) / np.sum(np.power(np.sinc(nu + k_vector), degree + 1), 0)

    elif filter_name == 'OBLIQUE':
        filter_f = np.abs(nu) / np.power(np.sinc(nu), degree + 1)

    elif filter_name == 'FRACTIONAL':
        filter_f = np.abs(np.sin(np.pi * nu) / np.pi) / np.sum(np.power(np.abs(np.sinc(nu + k_vector)), degree + 2), 0)

    elif filter_name == 'FRACTIONALOBLIQUE':
        filter_f = np.abs(np.sin(np.pi * nu) / 2) / np.power(np.abs(np.sinc(nu)), degree + 2)

    elif filter_name == 'BSPLINEPROJ':
        nu_k = nu + k_vector
        filter_f = np.sum(np.abs(nu_k) * np.power(np.sinc(nu_k), degree + 2 + n0), 0) / (
                np.sum(np.power(np.sinc(nu_k), n0 + 1), 0) *
                np.sum(np.power(np.abs(np.power(np.sinc(nu_k), degree + 1)), 2), 0))

    else:
        raise ValueError('Illegal filter name: {}'.format(filter_name))

    return filter_f, pre_filter


def filter_sinogram(sinogram, filter_name, degree):
    length = sinogram.shape[0]
    n = np.power(2, np.ceil(np.log2(4 * length)))

    filter_f, pre_filter = get_filter(n, filter_name, degree)
    filtered = np.fft.fft(sinogram, n, axis=0)
    filtered = np.real(np.fft.ifft(filtered * filter_f))
    filtered = filtered[:length, :]

    return filtered, pre_filter
