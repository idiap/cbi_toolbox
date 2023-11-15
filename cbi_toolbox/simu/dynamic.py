"""
The dynamic module provides simulations of imaging of periodic events such as a
beating heart.
"""

# Copyright (c) 2022 Idiap Research Institute, http://www.idiap.ch/
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


def sigsin(phase, slope, bias):
    """
    Compute the sigsin function.

    Parameters
    ----------
    phase : array_like
        The phase (radians) at which to compute sigsin.
    slope : float
        The slope of the sigsin. Higher = sharper pulse.
    bias : float
        The bias of the sigsin. Higher = shorter pulse.

    Returns
    -------
    array_like
        The computed sigsin values.
    """

    return (1 + np.exp((bias - 1) * slope)) / (
        1 + np.exp(-(np.sin(phase) - bias) * slope)
    )


def sigsin_beat(
    phase,
    size,
    sigsin_slopes=(5, 9, 9),
    sigsin_saturations=(0.95, 0.8, 0.7),
    sigsin_init_phases=(np.pi / 18, 0, np.pi / 20),
    sigsin_amplitudes=(0.4, 0.9, np.pi / 10),
    wavelength=2,
    phase_0=np.pi * 4 / 5,
    beat_center=(0.48, 0.49),
    rotate_first=False,
    dtype=np.float64,
):
    """
    Generate coordinate grids transformed using sigsin functions.
    If used to generate an ellipse, this resembles a beating heart.
    The sigsins control respectively the following transforms: scaling in X,
    scaling in Y, and rotation.

    X is defined as the direction of propagation of the sigsins.

    Parameters
    ----------
    phase : float or np.ndarray(float) [N]
        Unitless phase(s) at which to compute the coordinates.
        The unitless phase varies in [0, 1[.
    size : int
        Size of the coordinate grids.
    sigsin_slopes : tuple, optional
        Slopes of the sigsins, by default (5, 9, 9)
    sigsin_saturations : tuple, optional
        Saturations of the sigsins, by default (0.95, 0.8, 0.7)
    sigsin_init_phases : tuple, optional
        Initial phase shifts of the sigsins in radians,
        by default (np.pi / 18, 0, np.pi / 20)
    sigsin_amplitudes : tuple, optional
        Amplitude of the sigsins, by default (0.4, 0.9, np.pi / 10)
    wavelength : float, optional
        Wavelength of the contraction pulse in the X dimension, by default 2.
    phase_0 : float, optional
        Initial phase in radians that corresponds to the unitless phase of 0,
        by default np.pi*4/5. This is used to delay all the sigsins simultaneously.
    beat_center : tuple, optional
        Center of the X and Y scaling, by default (0.48, 0.49)
    rotate_first : bool, optional
        Whether to compute the rotation before the scalings, by default False
    dtype : numpy type
        The array type to use for coordinates, by default numpy.float64.

    Returns
    -------
    np.ndarray [N, 2, size, size]
        The array of coordinates in the transformed space.
        For each of the N phases, a [2, size, size] coordinates meshgrid is given.
    """

    beat_center = np.array(beat_center)
    phase = np.atleast_1d(phase)
    slopes = np.atleast_1d(sigsin_slopes)
    sats = np.atleast_1d(sigsin_saturations)
    phases = np.atleast_1d(sigsin_init_phases) + phase_0
    amps = np.atleast_1d(sigsin_amplitudes)

    coords = np.mgrid[:size, :size] / (size - 1)
    coords = np.vstack((coords, np.ones_like(coords[0:1])))

    x_coords = np.arange(size) / size

    phases_ = (
        x_coords[None, None, :] / wavelength + phase[None, :, None]
    ) * 2 * np.pi + phases[:, None, None]

    sigsins = amps[:, None, None] * sigsin(
        phases_, slopes[:, None, None], sats[:, None, None]
    )

    zeros = np.zeros_like(sigsins[0])
    ones = np.ones_like(sigsins[0])
    sin = np.sin(-sigsins[2])
    cos = np.cos(-sigsins[2])
    scale_x = 1 + sigsins[0]
    scale_y = 1 + sigsins[1]
    center_x = beat_center[0]
    center_y = beat_center[1]

    if not rotate_first:
        fullmat = np.array(
            [
                [
                    scale_x * cos,
                    -scale_x * sin,
                    center_x + scale_x * (center_y * sin - center_x * cos),
                ],
                [
                    scale_y * sin,
                    scale_y * cos,
                    center_y - scale_y * (center_x * sin + center_y * cos),
                ],
                [zeros, zeros, ones],
            ]
        )
    else:
        fullmat = np.array(
            [
                [
                    scale_x * cos,
                    -scale_y * sin,
                    center_x - center_x * scale_x * cos + center_y * scale_y * sin,
                ],
                [
                    scale_x * sin,
                    scale_y * cos,
                    center_y - center_x * scale_x * sin - center_y * scale_y * cos,
                ],
                [zeros, zeros, ones],
            ]
        )

    transformed = np.einsum(
        "nmtw,mwh->tnwh", fullmat.astype(dtype), coords.astype(dtype), dtype=dtype
    )

    return transformed[:, :2, ...]


def sigsin_beat_3(
    phase,
    size,
    sigsin_slopes=(5, 9, 9),
    sigsin_saturations=(0.95, 0.8, 0.7),
    sigsin_init_phases=(np.pi / 18, np.pi / 20, 0),
    sigsin_amplitudes=(0.7, 0.9, np.pi / 8),
    wavelength=2,
    phase_0=np.pi * 4 / 5,
    beat_center=(0.48, 0.49),
    rotate_first=False,
    dtype=np.float64,
):
    """
    Generate 3D coordinate grids transformed using sigsin functions.
    If used to generate a tube, this resembles a beating heart.
    The sigsins control respectively the following transforms: scaling in Z,
    scaling in X, and rotation.

    Y is defined as the direction of propagation of the sigsins.

    Parameters
    ----------
    phase : float or np.ndarray(float) [N]
        Unitless phase(s) at which to compute the coordinates.
        The unitless phase varies in [0, 1[.
    size : int
        Size of the coordinate grids.
    sigsin_slopes : tuple, optional
        Slopes of the sigsins, by default (5, 9, 9)
    sigsin_saturations : tuple, optional
        Saturations of the sigsins, by default (0.95, 0.8, 0.7)
    sigsin_init_phases : tuple, optional
        Initial phase shifts of the sigsins in radians,
        by default (np.pi / 18, 0, np.pi / 20)
    sigsin_amplitudes : tuple, optional
        Amplitude of the sigsins, by default (0.4, 0.9, np.pi / 10)
    wavelength : float, optional
        Wavelength of the contraction pulse in the Y dimension, by default 2.
    phase_0 : float, optional
        Initial phase in radians that corresponds to the unitless phase of 0,
        by default np.pi*4/5. This is used to delay all the sigsins simultaneously.
    beat_center : tuple, optional
        Center of the Z and Y scaling, by default (0.48, 0.49)
    rotate_first : bool, optional
        Whether to compute the rotation before the scalings, by default False
    dtype : numpy type
        The array type to use for coordinates, by default numpy.float64.

    Returns
    -------
    np.ndarray [N, 3, size, size, size]
        The array of coordinates in the transformed space.
        For each of the N phases, a [3, size, size, size] coordinates meshgrid is given.
    """

    beat_center = np.array(beat_center)
    phase = np.atleast_1d(phase)
    slopes = np.atleast_1d(sigsin_slopes)
    sats = np.atleast_1d(sigsin_saturations)
    phases = np.atleast_1d(sigsin_init_phases) + phase_0
    amps = np.atleast_1d(sigsin_amplitudes)

    coords = np.mgrid[:size, :size, :size] / (size - 1)
    coords = np.vstack((coords, np.ones_like(coords[0:1])))

    y_coords = np.arange(size) / size

    phases_ = (
        y_coords[None, None, :] / wavelength + phase[None, :, None]
    ) * 2 * np.pi + phases[:, None, None]

    sigsins = amps[:, None, None] * sigsin(
        phases_, slopes[:, None, None], sats[:, None, None]
    )

    zeros = np.zeros_like(sigsins[0])
    ones = np.ones_like(sigsins[0])
    sin = np.sin(-sigsins[2])
    cos = np.cos(-sigsins[2])
    scale_z = 1 + sigsins[0]
    scale_x = 1 + sigsins[1]
    center_z = beat_center[0]
    center_x = beat_center[1]

    if not rotate_first:
        fullmat = np.array(
            [
                [
                    scale_z * cos,
                    -scale_z * sin,
                    zeros,
                    center_z + scale_z * (center_x * sin - center_z * cos),
                ],
                [
                    scale_x * sin,
                    scale_x * cos,
                    zeros,
                    center_x - scale_x * (center_z * sin + center_x * cos),
                ],
                [zeros, zeros, ones, zeros],
                [zeros, zeros, zeros, ones],
            ]
        )
    else:
        fullmat = np.array(
            [
                [
                    scale_z * cos,
                    -scale_x * sin,
                    zeros,
                    center_z - center_z * scale_z * cos + center_x * scale_x * sin,
                ],
                [
                    scale_z * sin,
                    scale_x * cos,
                    zeros,
                    center_x - center_z * scale_z * sin - center_x * scale_x * cos,
                ],
                [zeros, zeros, ones, zeros],
                [zeros, zeros, zeros, ones],
            ]
        )

    transformed = np.einsum(
        "nmty,mzxy->tnzxy", fullmat.astype(dtype), coords.astype(dtype), dtype=dtype
    )

    return transformed[:, :-1, ...]


def sample_phases(
    n_phases,
    f_signal=np.pi,
    f_sample=10,
    sigma=1.5e-2,
    sigma_ratio=1.5e-2,
    initial_phase=None,
    pattern=None,
    seed=None,
):
    """
    Simulate the sampling of a periodic signal with local noise and random
    acceleration. This computes the unitless phases at which the signal is
    sampled given the frequency of the acquisition device and the noise.

    Parameters
    ----------
    n_phases : int
        The number of phases to sample.
    f_signal : float, optional
        The frequency of the sampled signal, by default np.pi
    f_sample : int, optional
        The frequency of the acquisition device, by default 10
    sigma : float, optional
        Standard deviation of the phase error after one period, by default 1.5e-2
    sigma_ratio : float, optional
        Strength of the integrated variation (acceleration effect) with respect
        to the instantaneous variation. Must be in [0,1], by default 1.5e-2
    initial_phase : float, optional
        First sampled phase (between 0 and 1), by default None uses an uniformly
        distributed random initial phase.
    pattern : array-like (float), optional
        Sampling pattern inside one frame, expressed as position between 0 the
        start of the acquisition frame and 1 the start of the next frame, by
        default None (corresponds to (0,), a single sampling per frame).
        Must be in the range [0, 1[ and sorted, and start with 0.
    seed : int, optional
        Seed used for the random number generator, by default None

    Returns
    -------
    np.ndarray(float) [n_phases * len(pattern)]
        The unitless phases at which the signal is sampled.
    """

    rng = np.random.default_rng(seed)
    if initial_phase is None:
        initial_phase = rng.uniform()
    phase = initial_phase

    delta_phase = f_signal / f_sample

    acc_sigma = sigma * np.sqrt(sigma_ratio)
    theta_sigma = sigma * np.sqrt(1 - sigma_ratio)

    if pattern is None or len(pattern) == 1:
        loc_sigma_acc = acc_sigma * np.sqrt(
            6 * delta_phase**3 / (delta_phase**2 + 3 * delta_phase + 2)
        )
        loc_sigma_theta = theta_sigma * np.sqrt(delta_phase)

        phases = np.empty(n_phases)

        for index in range(n_phases):
            phases[index] = phase
            delta_phase += rng.normal(scale=loc_sigma_acc)
            phase += delta_phase
            phase += rng.normal(scale=loc_sigma_theta)

    else:
        assert list(pattern) == sorted(pattern), "Pattern must be sorted"
        assert max(pattern) < 1, "Elements of pattern must be < 1"
        assert min(pattern) >= 0, "Elements of pattern must be >= 0"
        assert len(set(pattern)) == len(pattern), "Pattern must not contain duplicates"

        phases = np.empty(n_phases * len(pattern))

        pattern = np.array(pattern)
        pattern -= pattern[0]
        pattern_ = np.concatenate((pattern, [1]))

        d_pattern = pattern_[1:] - pattern_[:-1]

        d_phase = d_pattern * delta_phase

        loc_sigma_acc = acc_sigma * np.sqrt(
            6 * d_phase**3 / (d_phase**2 + 3 * d_phase + 2)
        )
        loc_sigma_acc /= d_pattern

        loc_sigma_theta = theta_sigma * np.sqrt(d_phase)

        for index in range(n_phases):
            for jndex, d_pat in enumerate(d_pattern):
                phases[len(pattern) * index + jndex] = phase

                delta_phase += rng.normal(scale=loc_sigma_acc[jndex])
                phase += delta_phase * d_pat
                phase += rng.normal(scale=loc_sigma_theta[jndex])

    return phases


if __name__ == "__main__":
    import napari
    from cbi_toolbox.simu import primitives

    phases = sample_phases(20, 1, 20)
    beat = sigsin_beat(phases, 32)
    beat = primitives.forward_ellipse(beat, (0.5, 0.5), (0.3, 0.4))

    napari.view_image(beat)
    napari.run()

    beat = sigsin_beat_3(phases, 32)
    beat = primitives.forward_ellipse_3(beat, (0.5, 0.5, 0.5), (0.3, 0.4, 0.4))

    napari.view_image(beat)
    napari.run()
