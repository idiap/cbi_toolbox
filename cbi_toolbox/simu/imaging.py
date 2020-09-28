import scipy.signal as sig
import numpy.random as random


def widefield(obj, psf):
    image = sig.fftconvolve(obj, psf)
    image.clip(0, None, out=image)
    return image


def noise(image, seed=None):
    rng = random.default_rng(seed)
    poisson = rng.poisson(image)
    gauss = rng.normal()

    out = poisson + gauss

    return out


if __name__ == "__main__":
    from cbi_toolbox.simu import primitives, optics
    import napari

    TEST_SIZE = 256

    sample = primitives.boccia(TEST_SIZE)
    psf = optics.gaussian_psf(pixelscale=25e-6/TEST_SIZE, npix_axial=TEST_SIZE)

    image = widefield(sample, psf)
    noisy = noise(image)

    with napari.gui_qt():
        viewer = napari.view_image(sample)
        viewer.add_image(image)
        viewer.add_image(noisy)
