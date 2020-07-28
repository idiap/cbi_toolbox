import numpy as np
from scipy.stats import truncnorm
from cbi_toolbox import simu


def texture(size):
    dtype = np.float32
    n_spheres = 500
    flatten = 2
    norm_bound = 4 / flatten
    normal = truncnorm(-norm_bound, norm_bound)

    max_radius = 0.1 * size
    min_radius = 0.02 * size

    max_in_radius = 0.5

    min_intens = 0.05
    max_intens = 0.2

    volume = np.ones((size, size, size), dtype=dtype)

    for _ in range(n_spheres):
        center = np.ceil(normal.rvs(3) * (size / 2 - max_radius) / norm_bound + size / 2).astype(int)

        radius = int(np.random.uniform(min_radius, max_radius))
        in_radius = np.random.uniform(0, max_in_radius)
        intens = np.random.uniform(min_intens, max_intens)

        object = simu.ball(radius * 2, in_radius=in_radius, dtype=dtype)

        volume[center[0] - radius:center[0] + radius, center[1] - radius:center[1] + radius,
        center[2] - radius:center[2] + radius] *= (1 - object * intens)

    return 1 - volume


if __name__ == '__main__':
    volume = texture(512)

    from scipy.ndimage import gaussian_filter
    # gaussian_filter(volume, sigma=2, output=volume, mode='constant', truncate=3)


    import napari

    with napari.gui_qt():
        napari.view_image(volume)
