import numpy as np
from scipy.stats import truncnorm
from cbi_toolbox import simu
import noise


def spheres(size, density=1):
    dtype = np.float32
    n_spheres = int(density * size**3 / 5000)

    max_radius = int(0.1 * size)
    min_radius = int(0.02 * size)

    max_in_radius = 0.5

    min_intens = 0.05
    max_intens = 0.2

    pad_size = size + 4 * max_radius

    volume = np.ones((pad_size, pad_size, pad_size), dtype=dtype)

    for _ in range(n_spheres):
        center = (np.random.rand(3) * (size + 2 * max_radius)).astype(int) + max_radius

        radius = int(np.random.uniform(min_radius, max_radius))
        in_radius = np.random.uniform(0, max_in_radius)
        intens = np.random.uniform(min_intens, max_intens)

        object = simu.ball(radius * 2, in_radius=in_radius, dtype=dtype)

        volume[center[0] - radius:center[0] + radius, center[1] - radius:center[1] + radius,
        center[2] - radius:center[2] + radius] *= (1 - object * intens)

    return 1 - volume[volume.ndim * [slice(2*max_radius, -2*max_radius)]]


def simplex(size, scale=1, octaves=3, persistence=0.7, lacunarity=3.5, seed=None):
    if seed is None:
        seed = int(np.random.randint(2**10) * scale)
    else:
        seed = seed

    volume = np.empty((size, size, size), dtype=np.float32)
    scale /= size

    # TODO optimize loops
    for x in range(size):
        for y in range(size):
            for z in range(size):
                sample = noise.snoise3(seed + x*scale, seed + y*scale, seed + z*scale,
                                       octaves=octaves, persistence=persistence, lacunarity=lacunarity)
                volume[x, y, z] = sample
    return volume


if __name__ == '__main__':
    import napari

    # volume = spheres(256)
    #
    # with napari.gui_qt():
    #     napari.view_image(volume)

    volume = simplex(256)

    with napari.gui_qt():
        napari.view_image(volume)
