# B-spline interpolation function for degree up to 7
# Christian Jaques, june 2016, Computational Bioimaging Group, Idiap
# This code is a translation of Michael Liebling's matlab code,
# which was already largely based on a C-library written by Philippe
# Thevenaz, BIG, EPFL

from math import sqrt, ceil, log

from numpy import flipud
from scipy.signal import lfilter

from cbi_toolbox.splineradon.convert_to_samples import *


def initial_causal_coefficient(coeff, z, tolerance, boundary_condition='Mirror'):
    n = coeff.shape[0]
    if len(coeff.shape) > 1:
        m = coeff.shape[1]
    else:
        m = 1

    # mirror boundaries condition (mirror is on last sample)
    horizon = n
    if tolerance > 0.0:
        horizon = int(ceil(log(tolerance) / log(abs(z))))

    if boundary_condition.upper() == 'MIRROR':
        if horizon >= n:
            horizon = n
            # vectorization of exponentials of z
            z_powers = np.vstack(np.arange(horizon - 2) + 1)
            z_exp = np.ones((horizon - 2, m)) * z
            z_exp = np.power(z_exp, z_powers)
            # the whole signal is taken into account, boundary conditions
            # have to be taken into account. Mirror conditions -->
            z_exp_mirror = np.ones((horizon - 2, m)) * z
            z_powers = np.vstack(np.arange(2 * n - 3, n - 1, -1))
            z_exp_mirror = np.power(z_exp_mirror, z_powers)
            z_exp = z_exp + z_exp_mirror
            # compute sum c[k]*z**k
            c = np.sum(coeff[1:horizon - 1, :] * z_exp, axis=0)
            c = c / (1 - z ** (2 * n - 1))
            c = c + coeff[0, :] + coeff[-1, :] * z ** (n - 1)
        else:
            # vectorization of exponentials of z
            z_powers = np.vstack(np.arange(horizon))
            z_exp = np.ones((horizon, m)) * z
            z_exp = np.power(z_exp, z_powers)
            # compute sum c[k]*z**k
            c = np.sum(coeff[:horizon, :] * z_exp, axis=0)

    elif boundary_condition.upper() == 'PERIODIC':
        temp = np.ones((1, m))
        temp[0] = coeff[0]
        c1 = np.concatenate((temp, coeff[-1:0:-1, :]), axis=0)
        if tolerance > 0.0:
            horizon2 = np.mod(np.arange(horizon), n)
            # vectorization of exponentials of z
            z_powers = np.vstack(np.arange(horizon))
            z_exp = np.ones((horizon, m)) * z
            z_exp = np.power(z_exp, z_powers)
            # the whole signal is taken into account, boundary conditions
            # have to be taken into account. Periodic conditions -->
            # compute sum c[k]*z**k
            c = np.sum(c1[horizon2, :] * z_exp, axis=0)
        else:
            # vectorization of exponentials of z
            z_powers = np.vstack(np.arange(horizon - 1) + 1)
            z_exp = np.ones((horizon - 1, m)) * z
            z_exp = np.power(z_exp, z_powers)
            # compute sum c[k]*z**k
            c = c1[0, :] + np.sum(np.multiply(c1[1:, :], z_exp), axis=0) / (1 - z ** n)

    return c


def initial_anticausal_coefficient(c, z, boundary_condition='Mirror'):
    if boundary_condition.upper() == 'MIRROR':
        c0 = -z * (c[-1, :] + z * c[-2, :]) / (1 - z ** 2)
    elif boundary_condition.upper() == 'PERIODIC':
        n, m = c.shape
        z_exp = np.ones((n, m)) * z
        z_power = np.vstack(np.arange(n, 0, -1))
        z_exp = np.power(z_exp, z_power)
        temp = np.ones((1, m))
        temp[0] = c[-1, :]
        c1 = np.concatenate((c[-2::-1, :], temp), axis=0)
        c0 = (1. / (z ** n - 1.)) * np.sum(np.multiply(c1, z_exp), axis=0)

    return c0


def convert_to_interpolation_coefficients(c, degree, tolerance, boundary_condition='Mirror'):
    if degree == 0 or degree == 1:
        return c

    n = c.shape[0]
    # this is a bit of a hack, better way to process vectors of shape (n,) ?
    if len(c.shape) == 1:
        c = np.vstack(c)

    if n == 1:
        return c

    if degree == 2:
        z = sqrt(8.) - 3.
    elif degree == 3:
        z = sqrt(3.0) - 2.0
    elif degree == 4:
        z = [sqrt(664.0 - sqrt(438976.0)) + sqrt(304.0) - 19.0,
             sqrt(664.0 + sqrt(438976.0)) - sqrt(304.0) - 19.0]
    elif degree == 5:
        z = [sqrt(135.0 / 2.0 - sqrt(17745.0 / 4.0)) + sqrt(105.0 / 4.0) - 13.0 / 2.0,
             sqrt(135.0 / 2.0 + sqrt(17745.0 / 4.0)) - sqrt(105.0 / 4.0) - 13.0 / 2.0]
    elif degree == 6:
        z = [-0.488294589303044755130118038883789062112279161239377608394,
             -0.081679271076237512597937765737059080653379610398148178525368,
             -0.00141415180832581775108724397655859252786416905534669851652709]
    elif degree == 7:
        z = [-0.5352804307964381655424037816816460718339231523426924148812,
             -0.122554615192326690515272264359357343605486549427295558490763,
             -0.0091486948096082769285930216516478534156925639545994482648003]
    elif degree == 8:
        z = [-0.57468690924876543053013930412874542429066157804125211200188,
             -0.163035269297280935240551896860737052234768145508298548489731,
             -0.0236322946948448500234039192963613206126659208546294370651457,
             -0.000153821310641690911739352530184021607629640540700430019629940]
    elif degree == 9:
        z = [-0.60799738916862577900772082395428976943963471853990829550220,
             -0.201750520193153238796064685055970434680898865757470649318867,
             -0.043222608540481752133321142979429688265852380231497069381435,
             -0.00212130690318081842030489655784862342205485609886239466341517]
    elif degree == 10:
        z = [-0.63655066396942385875799205491349773313787959010128860432339,
             -0.238182798377573284887456162200161978666543494059728787251924,
             -0.065727033228308551538201803949684252205121694392255863103034,
             -0.0075281946755486906437698340318148831650567567441314086093636,
             -0.0000169827628232746642307274679399688786114400132341362095006930]
    elif degree == 11:
        z = [-0.66126606890073470691013126292248166961816286716800880802421,
             -0.272180349294785885686295280258287768151235259565335176244192,
             -0.089759599793713309944142676556141542547561966017018544406214,
             -0.0166696273662346560965858360898150837154727205519335156053610]
    else:
        raise ValueError("Invalid spline degree {0}".format(degree))

    # compute overall gain
    if isinstance(z, float):
        z = np.array([z])
    nb_poles = len(z)
    lmbda = 1.
    for k in range(nb_poles):
        lmbda *= (1.0 - z[k]) * (1.0 - 1. / z[k])
    # apply gain to coeffs
    c = c * lmbda

    # loop over all poles
    for k in range(nb_poles):
        # causal initialization
        c[0, :] = initial_causal_coefficient(c, z[k], tolerance, boundary_condition)
        # causal filter
        zinit = np.array(z[k] * c[0, :])
        zinit = zinit[:, np.newaxis]
        c[1:, :], zf = lfilter([1], [1, -z[k]], c[1:, :], axis=0, zi=np.transpose(zinit))
        # anticausal initialization
        c[-1, :] = initial_anticausal_coefficient(c, z[k], boundary_condition=boundary_condition)
        # anticausal filter
        zinit = z[k] * c[-1, :]
        zinit = zinit[:, np.newaxis]
        c[:-1], zf = lfilter([-z[k]], [1, -z[k]], flipud(c[0:-1, :]), axis=0, zi=np.transpose(zinit))
        c[:-1] = flipud(c[:-1])

    return c
