import numpy as np
from scipy import fft

from cbi_toolbox.utils import make_broadcastable, fft_size


def get_filter(n, filter_name: str, degree=None):

    filter_name = filter_name.upper()
    pre_filter = True

    freq = np.concatenate((np.arange(1, n / 2 + 1, 2, dtype=np.int),
                           np.arange(n / 2 - 1, 0, -2, dtype=np.int)))

    filter_f = np.zeros(n)
    filter_f[0] = 0.25
    filter_f[1::2] = -1 / (np.pi * freq) ** 2
    filter_f = 2 * np.real(fft.rfft(filter_f))

    if filter_name == 'NONE':
        filter_f = np.ones(filter_f.shape)

    elif filter_name == 'RAM-LAK':
        pass

    elif filter_name == 'SHEPP-LOGAN':
        omega = np.pi * fft.rfftfreq(n)[1:]
        filter_f[1:] *= np.sin(omega) / omega

    elif filter_name == 'COSINE':
        freq = np.linspace(np.pi/2, np.pi, filter_f.size, endpoint=False)
        cosine_filter = np.sin(freq)
        filter_f *= cosine_filter

    else:
        nu = np.arange((n + 1) / 2)
        n0 = 3

        k = 50
        k_vector = np.arange(-k, k + 0.5)
        k_vector = k_vector[..., np.newaxis]
        pre_filter = False

        if filter_name == 'B-SPLINE':
            filter_f = np.abs(
                nu) / np.sum(np.power(np.sinc(nu + k_vector), degree + 1), 0)

        elif filter_name == 'OBLIQUE':
            filter_f = np.abs(nu) / np.power(np.sinc(nu), degree + 1)

        elif filter_name == 'FRACTIONAL':
            filter_f = np.abs(np.sin(np.pi * nu) / np.pi) / \
                np.sum(np.power(np.abs(np.sinc(nu + k_vector)), degree + 2), 0)

        elif filter_name == 'FRACTIONALOBLIQUE':
            filter_f = np.abs(np.sin(np.pi * nu) / 2) / \
                np.power(np.abs(np.sinc(nu)), degree + 2)

        elif filter_name == 'BSPLINEPROJ':
            nu_k = nu + k_vector
            filter_f = np.sum(np.abs(nu_k) * np.power(np.sinc(nu_k), degree + 2 + n0), 0) / (
                np.sum(np.power(np.sinc(nu_k), n0 + 1), 0) *
                np.sum(np.power(np.abs(np.power(np.sinc(nu_k), degree + 1)), 2), 0))

        else:
            raise ValueError('Illegal filter name: {}'.format(filter_name))

    return filter_f, pre_filter


def filter_sinogram(sinogram, filter_name, degree=None):
    length = sinogram.shape[1]
    n = fft_size(length)

    filter_f, pre_filter = get_filter(n, filter_name, degree)
    filtered = fft.rfft(sinogram, n, axis=1)
    filter_f = make_broadcastable(filter_f[np.newaxis, ...], filtered)
    filtered *= filter_f
    filtered = fft.irfft(filtered, axis=1, overwrite_x=True)

    return filtered[:, :length], pre_filter


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    n = 256

    filters = [
        'None',
        'Ram-Lak',
        'Shepp-Logan',
        'Cosine'
    ]

    freq = fft.rfftfreq(n)
    for filt in filters:
        f, _ = get_filter(n, filt)
        plt.plot(freq, f, label=filt)

    plt.legend()
    plt.show()
