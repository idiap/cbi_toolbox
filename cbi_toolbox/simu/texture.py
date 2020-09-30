"""
The texture module allows to generate 3D textures for synthetic samples
"""

import numpy as np
import noise
from cbi_toolbox.simu import primitives


def spheres(size, density=1, seed=None):
    """
    Generates a texture full of hollow spheres

    Parameters
    ----------
    size : int
        size of the texture
    density : int, optional
        spheres density in the texture, by default
    seed : int, optional
        seed of the rng, default is None

    Returns
    -------
    array [size, size, size]
        the texture
    """

    dtype = np.float32
    n_spheres = int(density * 10000)

    max_radius = int(0.1 * size)
    min_radius = int(0.02 * size)

    max_in_radius = 0.5

    min_intens = 0.05
    max_intens = 0.2

    pad_size = size + 4 * max_radius

    volume = np.ones((pad_size, pad_size, pad_size), dtype=dtype)

    rng = np.random.default_rng(seed)

    for _ in range(n_spheres):
        center = (rng.random(3) * (size + 2 * max_radius)
                  ).astype(int) + max_radius

        radius = int(rng.uniform(min_radius, max_radius))
        in_radius = rng.uniform(0, max_in_radius)
        intens = rng.uniform(min_intens, max_intens)

        obj = primitives.ball(radius * 2, in_radius=in_radius, dtype=dtype)

        volume[center[0] - radius:center[0] + radius, center[1] - radius:center[1] + radius,
               center[2] - radius:center[2] + radius] *= (1 - obj * intens)

    return 1 - volume[volume.ndim * [slice(2*max_radius, -2*max_radius)]]


def simplex(size, scale=1, octaves=3, persistence=0.7, lacunarity=3.5, seed=None):
    """
    Generates 3D simplex noise

    Parameters
    ----------
    size : int
        size of the texture
    scale : int, optional
        scale of the noise, by default 1
    octaves : int, optional
        number of octaves used, by default 3
    persistence : float, optional
        relative amplitude of octaves, by default 0.7
    lacunarity : float, optional
        relative frequency of octaves, by default 3.5
    seed : int, optional
        seed for the noise, by default None

    Returns
    -------
    array [size, size, size]
        the texture
    """

    if seed is None:
        seed = int(np.random.default_rng().integers(2**10) * scale)

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

    TEST_SIZE = 128
    s_spheres = spheres(TEST_SIZE)
    s_simplex = simplex(TEST_SIZE)

    phantom = primitives.phantom(TEST_SIZE) * (s_simplex * 0.5 + 0.75)

    with napari.gui_qt():
        viewer = napari.view_image(s_spheres)
        viewer.add_image(s_simplex)
        viewer.add_image(phantom)
