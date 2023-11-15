"""
The textures module allows to generate 3D textures for synthetic samples
"""

# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by François Marelli <francois.marelli@idiap.ch>
#
# This file is part of CBI Toolbox.
#
# CBI Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the 3-Clause BSD License.
#
# CBI Toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# 3-Clause BSD License for more details.
#
# You should have received a copy of the 3-Clause BSD License along
# with CBI Toolbox. If not, see https://opensource.org/licenses/BSD-3-Clause.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import opensimplex
from opensimplex.internals import _noise2, _noise3
from numba import njit, prange

from cbi_toolbox.simu import primitives


def spheres(shape, density=1, seed=None):
    """
    Generates a texture full of hollow spheres

    Parameters
    ----------
    shape : tuple(int)
        shape of the texture array
    density : int, optional
        spheres density in the texture, by default
    seed : int, optional
        seed of the rng, default is None

    Returns
    -------
    array [shape]
        the texture
    """

    dtype = np.float32
    n_spheres = int(density * 10000)

    size = max(shape)

    max_radius = int(0.1 * size)
    min_radius = int(0.02 * size)

    max_in_radius = 0.5

    min_intens = 0.05
    max_intens = 0.2

    pad_shape = [s + 4 * max_radius for s in shape]

    volume = np.ones(pad_shape, dtype=dtype)

    rng = np.random.default_rng(seed)

    for _ in range(n_spheres):
        center = (rng.random(3) * (np.array(shape) + 2 * max_radius)).astype(
            int
        ) + max_radius

        radius = int(rng.uniform(min_radius, max_radius))
        in_radius = rng.uniform(0, max_in_radius)
        intens = rng.uniform(min_intens, max_intens)

        obj = primitives.ball(radius * 2, in_radius=in_radius, dtype=dtype)

        volume[
            center[0] - radius : center[0] + radius,
            center[1] - radius : center[1] + radius,
            center[2] - radius : center[2] + radius,
        ] *= (
            1 - obj * intens
        )

    return 1 - volume[volume.ndim * (slice(2 * max_radius, -2 * max_radius),)]


def simplex(shape, scale=1, seed=None):
    """
    Generates 2D/3D simplex noise
    Noise values are in [-1, 1]

    Parameters
    ----------
    shape : tuple (int)
        shape of the texture array
    scale : int, optional
        scale of the noise, by default 1
    ndim : int, optional
        number of dimensions of the array to generate (2, 3), by default 3
    seed : int, optional
        seed for the noise, by default None

    Returns
    -------
    array [shape]
        the simplex noise
    """

    if seed is None:
        seed = np.random.default_rng().integers(2**10)

    if len(shape) not in (2, 3):
        raise ValueError(
            f"Only 2D and 3D textures can be generated, got ndim={len(shape)}"
        )

    opensimplex.seed(seed)
    x = np.arange(shape[0]) / max(shape) * scale
    y = np.arange(shape[1]) / max(shape) * scale

    if len(shape) == 3:
        z = np.arange(shape[2]) / max(shape) * scale
        volume = opensimplex.noise3array(x, y, z)

    else:
        volume = opensimplex.noise2array(x, y)

    return volume


def forward_simplex(coordinates, scale=1, time=False, out=None, seed=None):
    """
    Computes simplex noise over given coordinates
    Noise values are in [-1, 1]
    Uses Numba acceleration

    Parameters
    ----------
    coordinates : np.ndarray [D, W, H, <Z>], or [T, D, W, H, <Z>] fore time series
        coordinates where the noise must be computed (meshgrid)
    scale : int, optional
        scale of the noise, by default 1
    time : bool, optional
        if the input corresponds to a time series of coordinates, by default False
    out: array, optional
        output array, by default None
    seed : int, optional
        seed for the noise, by default None

    Returns
    -------
    np.ndarray [W, H, <Z>], or [T, W, H, <Z>] for time series
        the simplex noise computed at the given coordinates
    """

    if seed is None:
        seed = np.random.default_rng().integers(2**10)

    if time:
        ndim = coordinates.shape[1]
    else:
        ndim = coordinates.shape[0]

    if coordinates.ndim != ndim + 1 + time:
        raise ValueError(
            "Coordinates should be in a meshgrid, but the size "
            "of the first dimension plus one does not match the dimensions of the whole array. "
        )

    if not ndim in (2, 3):
        raise NotImplementedError(
            "Only 2D and 3D coordinate arrays are implemented (3D and 4D meshgrids)"
        )

    generator = opensimplex.OpenSimplex(seed)

    if time:
        shape = (coordinates.shape[0], *coordinates.shape[2:])

    else:
        shape = coordinates.shape[1:]

    if out is None:
        out = np.empty(shape)

    elif out.shape != shape:
        raise ValueError("Output shape does not match coordinates.")

    if time:
        if ndim == 2:
            return _forward_simplex_inner2t(coordinates, scale, out, generator._perm)
        elif ndim == 3:
            return _forward_simplex_inner3t(
                coordinates, scale, out, generator._perm, generator._perm_grad_index3
            )
    else:
        if ndim == 2:
            return _forward_simplex_inner2(coordinates, scale, out, generator._perm)
        elif ndim == 3:
            return _forward_simplex_inner3(
                coordinates, scale, out, generator._perm, generator._perm_grad_index3
            )


@njit(parallel=True)
def _forward_simplex_inner3t(coordinates, scale, out, perm, perm_grad_index3):
    """
    Inner implementation of the simplex noise computation over given coordinates
    For 3D + time arrays
    Uses Numba acceleration
    """
    for kt in prange(coordinates.shape[0]):
        _forward_simplex_inner3(
            coordinates[kt, ...], scale, out[kt, ...], perm, perm_grad_index3
        )
    # for kt, coord in enumerate(coordinates):
    #     _forward_simplex_inner3(coord, scale, out[kt, ...], perm, perm_grad_index3)
    return out


@njit
def _forward_simplex_inner3(coordinates, scale, out, perm, perm_grad_index3):
    """
    Inner implementation of the simplex noise computation over given coordinates
    For 3D arrays
    Uses Numba acceleration
    """
    for kx, row in enumerate(coordinates.T):
        for ky, col in enumerate(row):
            for kz, dep in enumerate(col):
                out[kz, ky, kx] = _noise3(
                    dep[0] * scale,
                    dep[1] * scale,
                    dep[2] * scale,
                    perm,
                    perm_grad_index3,
                )
    return out


@njit(parallel=True)
def _forward_simplex_inner2t(coordinates, scale, out, perm):
    """
    Inner implementation of the simplex noise computation over given coordinates
    For 2D + time arrays
    Uses Numba acceleration
    """
    for kt in prange(coordinates.shape[0]):
        _forward_simplex_inner2(coordinates[kt, ...], scale, out[kt, ...], perm)
    return out


@njit
def _forward_simplex_inner2(coordinates, scale, out, perm):
    """
    Inner implementation of the simplex noise computation over given coordinates
    For 2D arrays
    Uses Numba acceleration
    """
    for kx, row in enumerate(coordinates.T):
        for ky, col in enumerate(row):
            out[ky, kx] = _noise2(col[0] * scale, col[1] * scale, perm)
    return out


if __name__ == "__main__":
    import napari

    TEST_SHAPE = (64, 64, 32)
    s_spheres = spheres(TEST_SHAPE)
    s_simplex = simplex(TEST_SHAPE, scale=8)

    viewer = napari.view_image(s_spheres)
    viewer.add_image(s_simplex)

    napari.run()
