"""
This modules implements signal conversion to bspline interpolation coefficients
and back.
"""

# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by François Marelli <francois.marelli@idiap.ch>,
# Christian Jaques <francois.marelli@idiap.ch>
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
#
# This code is a translation of Michael Liebling's matlab code,
# which was already largely based on a C-library written by Philippe
# Thevenaz, BIG, EPFL

import math

import numpy as np
from scipy import signal
from scipy import ndimage

from cbi_toolbox.utils import make_broadcastable
from cbi_toolbox import bsplines
from cbi_toolbox import parallel


def initial_causal_coefficient(coeff, z, tolerance, boundary_condition="mirror"):
    """
    Computes the initial causal coefficient from an array of coefficients
    In the input array, signals are considered to be along the first dimension
    (1D computations).
    """

    n = coeff.shape[0]

    # mirror boundaries condition (mirror is on last sample)
    horizon = n
    if tolerance > 0.0:
        horizon = int(math.ceil(math.log(tolerance) / math.log(abs(z))))

    if boundary_condition.upper() == "MIRROR":
        if horizon >= n:
            horizon = n
            # vectorization of exponentials of z
            z_powers = np.arange(horizon - 2) + 1
            z_exp = np.ones(horizon - 2) * z
            z_exp = np.power(z_exp, z_powers)
            # the whole signal is taken into account, boundary conditions
            # have to be taken into account. Mirror conditions -->
            z_exp_mirror = np.ones(horizon - 2) * z
            z_powers = np.arange(2 * n - 3, n - 1, -1)
            z_exp_mirror = np.power(z_exp_mirror, z_powers)
            z_exp += z_exp_mirror

            z_exp = make_broadcastable(z_exp, coeff)

            # compute sum c[k]*z**k
            c = np.sum(coeff[1 : horizon - 1, ...] * z_exp, axis=0)
            c /= 1 - z ** (2 * n - 1)
            c += coeff[0, ...] + coeff[-1, ...] * z ** (n - 1)
        else:
            # vectorization of exponentials of z
            z_powers = np.arange(horizon)
            z_exp = np.ones(horizon) * z
            z_exp = np.power(z_exp, z_powers)

            z_exp = make_broadcastable(z_exp, coeff)

            # compute sum c[k]*z**k
            c = np.sum(coeff[:horizon, ...] * z_exp, axis=0)

    elif boundary_condition.upper() == "PERIODIC":
        temp_shape = list(coeff.shape)
        temp_shape[0] = 1
        temp = np.ones(temp_shape)
        temp[0, ...] = coeff[0, ...]
        c1 = np.concatenate((temp, coeff[-1:0:-1, ...]), axis=0)
        if tolerance > 0.0:
            horizon2 = np.mod(np.arange(horizon), n)
            # vectorization of exponentials of z
            z_powers = np.arange(horizon)
            z_exp = np.ones(horizon) * z
            z_exp = np.power(z_exp, z_powers)
            # the whole signal is taken into account, boundary conditions
            # have to be taken into account. Periodic conditions -->
            z_exp = make_broadcastable(z_exp, c1)

            # compute sum c[k]*z**k
            c = np.sum(c1[horizon2, ...] * z_exp, axis=0)
        else:
            # vectorization of exponentials of z
            z_powers = np.arange(horizon - 1) + 1
            z_exp = np.ones(horizon - 1) * z
            z_exp = np.power(z_exp, z_powers)
            z_exp = make_broadcastable(z_exp, c1)

            # compute sum c[k]*z**k
            c = c1[0, ...] + np.sum(np.multiply(c1[1:, ...], z_exp), axis=0) / (
                1 - z**n
            )

    else:
        raise ValueError(
            "Illegal boundary condition: {}".format(boundary_condition.upper())
        )

    return c


def initial_anticausal_coefficient(c, z, boundary_condition="Mirror"):
    """
    Computes the initial anti-causal coefficient from an array of coefficients
    In the input array, signals are considered to be along the first dimension
    (1D computations).
    """
    if boundary_condition.upper() == "MIRROR":
        c0 = -z * (c[-1, ...] + z * c[-2, ...]) / (1 - z**2)
    elif boundary_condition.upper() == "PERIODIC":
        n = c.shape[0]
        z_exp = np.ones(n) * z
        z_power = np.arange(n, 0, -1)
        z_exp = np.power(z_exp, z_power)

        temp_shape = list(c.shape)
        temp_shape[0] = 1
        temp = np.ones(temp_shape)

        temp[0] = c[-1, ...]
        c1 = np.concatenate((c[-2::-1, ...], temp), axis=0)

        z_exp = make_broadcastable(z_exp, c1)

        c0 = (1.0 / (z**n - 1.0)) * np.sum(np.multiply(c1, z_exp), axis=0)
    else:
        raise ValueError(
            "Illegal boundary condition: {}".format(boundary_condition.upper())
        )
    return c0


def convert_to_interpolation_coefficients(
    c, degree, tolerance=1e-9, boundary_condition="Mirror", in_place=False
):
    """
    Computes the b-spline interpolation coefficients of a signal
    In the input array, the signals are considered along the first
    dimension (1D computations).
    """

    if degree == 0 or degree == 1 or c.shape[0] == 1:
        if not in_place:
            return c.copy()
        return c

    elif 2 <= degree <= 5:
        if boundary_condition.upper() == "MIRROR":
            mode = "mirror"
        elif boundary_condition.upper() == "PERIODIC":
            mode = "grid-wrap"
        elif boundary_condition.upper() == "CONSTANT":
            mode = "grid-constant"
        else:
            raise ValueError(
                "Invalid boundary condition: {}".format(boundary_condition)
            )

        output = None
        if in_place:
            output = c

        if c.ndim < 2:
            output = ndimage.spline_filter1d(
                c, degree, axis=0, mode=mode, output=output
            )

        else:
            if output is None:
                output = np.empty_like(c)

            split_size = c.shape[-1]

            def to_spline(start, width):
                in_array = c[:, ..., start : start + width]
                out_array = output[:, ..., start : start + width]
                ndimage.spline_filter1d(
                    in_array, degree, axis=0, mode=mode, output=out_array
                )

            parallel.parallelize(to_spline, split_size)

        return output

    if not in_place:
        c = np.array(c)

    if degree == 6:
        z = [
            -0.488294589303044755130118038883789062112279161239377608394,
            -0.081679271076237512597937765737059080653379610398148178525368,
            -0.00141415180832581775108724397655859252786416905534669851652709,
        ]
    elif degree == 7:
        z = [
            -0.5352804307964381655424037816816460718339231523426924148812,
            -0.122554615192326690515272264359357343605486549427295558490763,
            -0.0091486948096082769285930216516478534156925639545994482648003,
        ]
    elif degree == 8:
        z = [
            -0.57468690924876543053013930412874542429066157804125211200188,
            -0.163035269297280935240551896860737052234768145508298548489731,
            -0.0236322946948448500234039192963613206126659208546294370651457,
            -0.000153821310641690911739352530184021607629640540700430019629940,
        ]
    elif degree == 9:
        z = [
            -0.60799738916862577900772082395428976943963471853990829550220,
            -0.201750520193153238796064685055970434680898865757470649318867,
            -0.043222608540481752133321142979429688265852380231497069381435,
            -0.00212130690318081842030489655784862342205485609886239466341517,
        ]
    elif degree == 10:
        z = [
            -0.63655066396942385875799205491349773313787959010128860432339,
            -0.238182798377573284887456162200161978666543494059728787251924,
            -0.065727033228308551538201803949684252205121694392255863103034,
            -0.0075281946755486906437698340318148831650567567441314086093636,
            -0.0000169827628232746642307274679399688786114400132341362095006930,
        ]
    elif degree == 11:
        z = [
            -0.66126606890073470691013126292248166961816286716800880802421,
            -0.272180349294785885686295280258287768151235259565335176244192,
            -0.089759599793713309944142676556141542547561966017018544406214,
            -0.0166696273662346560965858360898150837154727205519335156053610,
        ]
    else:
        raise ValueError("Invalid spline degree {0}".format(degree))

    # compute overall gain
    z = np.atleast_1d(z)
    # apply gain to coeffs
    c *= np.prod((1 - z) * (1 - 1 / z))

    # loop over all poles
    for pole in z:
        # causal initialization
        c[0, ...] = initial_causal_coefficient(c, pole, tolerance, boundary_condition)
        # causal filter
        zinit = pole * c[0, ...]
        zinit = zinit[np.newaxis, ...]

        if c.ndim < 2:
            c[1:, ...], _ = signal.lfilter(
                [1], [1, -pole], c[1:, ...], axis=0, zi=zinit
            )

        else:
            split_size = c.shape[-1]

            def lfilt(start, width):
                c[1:, ..., start : start + width], _ = signal.lfilter(
                    [1],
                    [1, -pole],
                    c[1:, ..., start : start + width],
                    axis=0,
                    zi=zinit[:, ..., start : start + width],
                )

            parallel.parallelize(lfilt, split_size)

        # anticausal initialization
        c[-1, ...] = initial_anticausal_coefficient(
            c, pole, boundary_condition=boundary_condition
        )
        # anticausal filter
        zinit = pole * c[-1, ...]
        zinit = zinit[np.newaxis, ...]

        if c.ndim < 2:
            c[:-1], _ = signal.lfilter(
                [-pole], [1, -pole], np.flipud(c[0:-1, ...]), axis=0, zi=zinit
            )

        else:
            split_size = c.shape[-1]
            fc = np.flipud(c[0:-1, ...])

            def lfilt2(start, width):
                c[:-1, ..., start : start + width], _ = signal.lfilter(
                    [-pole],
                    [1, -pole],
                    fc[:, ..., start : start + width],
                    axis=0,
                    zi=zinit[:, ..., start : start + width],
                )

            parallel.parallelize(lfilt2, split_size)

        c[:-1] = np.flipud(c[:-1])

    return c


def convert_to_samples(c, deg, boundary_condition="Mirror", in_place=False):
    """
    Convert interpolation coefficients into samples
    In the input array, the signals are considered along the first dimension
    (1D computations).
    """

    n = c.shape[0]
    if n == 1:
        return c

    kerlen = int(2 * math.floor(deg / 2.0) + 1)

    k = -math.floor(deg / 2.0) + np.arange(kerlen)
    kernel = bsplines.bspline(k, deg)

    # different extensions based on boundary condition
    if boundary_condition.upper() == "MIRROR":
        boundary = "mirror"
    elif boundary_condition.upper() == "PERIODIC":
        boundary = "wrap"
    else:
        raise ValueError(
            "Illegal boundary condition: {}".format(boundary_condition.upper())
        )

    kernel = make_broadcastable(kernel, c)

    output = None
    if in_place:
        output = c

    return ndimage.convolve(c, kernel, mode=boundary, output=output)
