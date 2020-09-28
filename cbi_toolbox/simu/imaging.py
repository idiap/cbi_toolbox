import scipy.signal as sig


def widefield(obj, psf):
    return sig.fftconvolve(obj, psf)


if __name__ == "__main__":
    from cbi_toolbox.simu import primitives, optics
    import napari

    obj = primitives.boccia(256)
    psf = optics.gaussian_psf(pixelscale=25e-6/256, npix_axial=256)
    image = widefield(obj, psf)

    with napari.gui_qt():
        napari.view_image(image)
