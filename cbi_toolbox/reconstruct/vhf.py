"""
This module implements virtual high-framerate sequence reconstruction
methods for periodic signals.

[1] Mariani, O., Ernst, A., Mercader, N. and Liebling, M., 2019. Reconstruction
of image sequences from ungated and scanning-aberrated laser scanning
microscopy images of the beating heart. IEEE transactions on computational
imaging, 6, pp.385-395.
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


import tempfile
import pathlib
import shutil
from functools import partial
import numpy as np
import scipy.spatial.distance as scidist
from scipy import sparse
import concorde.tsp as tsp
import scs
import jax
import jax.numpy as jnp
import optax


def phase_diff(phase_a, phase_b):
    """
    Compute the difference between unitless phases (between 0 and 1).
    The result of the difference lies in the interval [-0.5, 0.5[.

    Parameters
    ----------
    phase_a : array-like of floats (or float)
        Unitless phases, first operand of the difference.
    phase_b : array-like of floats (or float)
        Unitless phases, second operand of the difference.

    Returns
    -------
    type(phase_a)
        The phase difference.
    """

    diff = (phase_a - phase_b) % 1
    diff -= diff >= 0.5
    return diff


def phase_error(phase_a, phase_b, floating_origin=True):
    """
    Compute the absolute phase error between unitless phases (between 0 and 1).
    If the starting phases of the two operands are not fixed with respect to
    one another, the operands are first shifted by a scalar bias to minimize
    this error (i.e. the zero phase of the second sighal is set so that the
    mean phase differnce between the two operands is zero).

    Parameters
    ----------
    phase_a : array-like of floats
        Reference unitless phases.
    phase_b : array-like of floats
        Target unitless phases.
    floating_origin : bool, optional
        If the zero phase of the target phase can be shifted with respect to the
        zero of the reference phases to minimize the error, by default True.

    Returns
    -------
    numpy.ndarray(float)
        The absolute phase error between the two.
    """

    if floating_origin:
        phase_a = phase_a - phase_a[0] + phase_b[0]

    diffs = phase_diff(phase_a, phase_b)

    if floating_origin:
        mean = np.mean(diffs)
        nit = 0
        while np.abs(mean) > 1e-5:
            if nit > 10:
                raise RuntimeError("Floating origin did not converge")

            diffs -= mean
            diffs %= 1
            diffs -= diffs > 0.5
            mean = np.mean(diffs)
            nit += 1

    err = np.abs(diffs)

    return err


def argsort_tsp(images):
    """
    Returns the indices that sort a sequence of images using TSP to obtain a
    virtual high-framerate signal.
    The direction of the produced sequence is undefined.

    Parameters
    ----------
    images : numpy.ndarray [N, ...]
        List of images to sort.

    Returns
    -------
    numpy.ndarray(int)
        Array of indices that sort the sequence according to the TSP slution.

    Raises
    ------
    RuntimeWarning
        When the TSP solver does not converge.
    """

    distances = scidist.pdist(
        images.reshape(images.shape[0], -1), metric="minkowski", p=1
    )

    # TSP files expect integer numbers, this allows higher precision
    # It is unknown what the maximum allowed value is, this is good enough
    distances /= distances.max()
    distances *= 1e6

    try:
        ccdir = pathlib.Path(tempfile.mkdtemp())
        ccfile = ccdir / "data.tsp"

        with ccfile.open("w") as file_pointer:
            file_pointer.writelines(
                [
                    "NAME: PHASESORT\n",
                    "TYPE: TSP\n",
                    "DIMENSION: {}\n".format(images.shape[0]),
                    "EDGE_WEIGHT_TYPE: EXPLICIT\n",
                    "EDGE_WEIGHT_FORMAT: UPPER_ROW\n",
                    "EDGE_WEIGHT_SECTION\n",
                ]
            )
            np.savetxt(file_pointer, distances, fmt="%d")
            file_pointer.write("EOF\n")

        solver = tsp.TSPSolver.from_tspfile(str(ccfile))
        solution = solver.solve(verbose=False)

        if not solution.success:
            raise RuntimeWarning("Solver did not converge.")

    finally:
        shutil.rmtree(ccdir)

    return solution.tour


def unfold_phases(phases, threshold=-0.25, in_place=False):
    """
    Unfold consecutive unitless phases from the range [0, 1] to R based on a
    negative delta threshold.

    Parameters
    ----------
    phases : numpy.ndarray [N]
        The unitless phases to unfold, in [0, 1].
    threshold : float, optional
        The negative threshold to go to the next period, by default -0.25
    in_place : bool, optional
        Compute operations in-place, by default False

    Returns
    -------
    numpy.ndarray [N]
        The unfolded phases over multiple periods.
    """

    if not in_place:
        phases = np.array(phases)

    for idx, phase in enumerate(phases[1:]):
        if phase - phases[idx] < threshold:
            phases[idx + 1 :] += 1

    return phases


def orient_phases(phases, seq_lengths=None, in_place=False):
    """
    Detect the orientation of a sequence of phases sorted using TSP and flip it
    if necessary. Assumes that the phases have been acquired in continuous
    sequences under Shannon sampling assumption.

    Parameters
    ----------
    phases : numpy.ndarray (float) [N]
        The phases to orient, given in the order of acquisition.
    seq_lengths : tuple (int)
        The lengths of sequences of images acquired consecutively, if multiple
        sequences have been acquired. The total sum of sequence lengths must be
        equal to the total number of images N. By default, None: considers the
        entire sequence to be consecutive.

    Returns
    -------
    c_phases : numpy.ndarray (float) [N]
        The estimated unitless phases corresponding to the input images.

    Raises
    ------
    ValueError
    """

    if not in_place:
        phases = np.array(phases)

    if seq_lengths is None:
        seq_lengths = (phases.size,)
    if sum(seq_lengths) != phases.size:
        raise ValueError("The sum of sequence lengths must be equal to N.")

    # Detect sequence direction under Shannon sampling hypothese
    seq_start = 0
    phase_delta = 0
    for seq in seq_lengths:
        seq_end = seq_start + seq
        seq_deltas = (
            phases[seq_start + 1 : seq_end] - phases[seq_start : seq_end - 1]
        ) % 1
        phase_delta += seq_deltas.sum()
        seq_start = seq_end
    phase_delta /= phases.size - len(seq_lengths)

    if phase_delta > 0.5:
        phases *= -1
        phases %= 1

    return phases


def average_frequency(phases, acq_frequency, n_periods=2, unfold=True):
    """
    Computes the initial average frequency of a periodic signal from uniformly
    sampled phases by fitting a line using least squares.

    Parameters
    ----------
    phases : numpy.ndarray (float) [N]
        The unitless phases sampled in the signal.
    acq_frequency : float
        The frequency at which the signal was sampled.
    n_periods : float, optional
        The number of periods over which to compute the average, by default 1.
    unfold : bool, optional
        If the input phases are folded over a single period, by default True.

    Returns
    -------
    float
        The average frequency of the sampled signal over the given region. The
        utits are the same as the acquisition frequency.
    """

    n_anchors = phases.size
    if unfold:
        phases = unfold_phases(phases)

    if phases[-1] < n_periods:
        n_anchors = phases.size
    else:
        n_anchors = np.argmax(phases > n_periods)

    if n_anchors < 2:
        raise ValueError(
            "Insufficient number of points, increase n_periods or provide more points."
        )

    # Use linear regression to find the slope of the curve
    x_anchors = np.stack((np.arange(n_anchors), np.ones(n_anchors)), axis=-1)
    sol = np.linalg.lstsq(x_anchors, phases[:n_anchors], rcond=None)[0][0]

    return sol * acq_frequency


def sample_video(images, phases, n_frames):
    """
    Re-sample a non-uniformly sampled image sequence to generate a video.

    Parameters
    ----------
    images : np.ndarray [N, ...]
        Sequence of images of size N.
    phases : np.ndarray [N]
        Sequence of phases corresponding to the images respectively.
    n_frames : int
        Number of frames in the generated video

    Returns
    -------
    video: np.ndarray [n_frames, W, H]
        The resampled video.
    """

    order = np.argsort(phases)
    phases = phases[order]
    images = images[order]

    video = np.empty(shape=(n_frames, *images.shape[1:]), dtype=images.dtype)
    phase_idx = 0

    for frame_idx in range(n_frames):
        frame_phase = frame_idx / n_frames
        phase_dist = np.abs(phases[phase_idx] - frame_phase)

        for _ in range(phases.size - phase_idx - 1):
            _dist = np.abs(phases[phase_idx + 1] - frame_phase)
            if _dist < phase_dist:
                phase_dist = _dist
                phase_idx += 1
            else:
                break

        video[frame_idx] = images[phase_idx]

    return video


def vhf_phase_naive(images):
    """
    Use naive vhf method to estimate the phases corresponding to a randomly
    sampled periodic signal by solving TSP, as exposed in [1].

    Parameters
    ----------
    images : numpy.ndarray [N, ...]
        The measured images.

    Returns
    -------
    numpy.ndarray (float) [N]
        The estimated unitless phases corresponding to each input image.
    """

    # Sort the images using TSP, assume uniform sampling
    argsort = argsort_tsp(images)
    uni_phases = np.argsort(argsort) / argsort.size

    uni_phases -= uni_phases[0]
    uni_phases %= 1

    return uni_phases


def estimate_di(phases, distances, k_size, w_width, d_i_k=None):
    """
    Estimate the image function distance on phases based on phase neighbours.

    Parameters
    ----------
    phases : numpy.ndarray (float) [N]
        The phase array.
    distances : numpy.ndarray (float) [N, N]
        The pairwise image distances.
    k_size : int
        The number of points of di to estimate.
    w_width : float
        The size of the weighting window.
    d_i_k : numpy.ndarray (float) [N, K+1], optional
        The array to store the d_i_kput, by default None

    Returns
    -------
    numpy.ndarray [N, K+1]
        The estimated distance functions for each point.
    float
        The residual error after optimization.

    Raises
    ------
    ValueError
    RuntimeWarning
    """
    size = phases.size

    # This is a weird bug, sometimes %1 still contains a 1
    delta_c = ((phases[None, :] - phases[:, None]) % 1) % 1

    # Create the base condition matrix for increasing monotonic
    a_cone_template = sparse.diags(
        (-1.0, 1.0), offsets=(0, -1), shape=(k_size, k_size - 1), format="csc"
    )

    if d_i_k is None:
        d_i_k = np.empty((size, k_size + 1))
    else:
        if not np.array_equal(d_i_k.shape, (size, k_size + 1)):
            raise ValueError(
                f"Incorrect d_i_k array size, expected {(size, k_size+1)} and got {d_i_k.shape}"
            )

    d_i_k[:, 0] = 0
    d_i_k[:, -1] = 0
    # pivots = np.empty(size, dtype=int)

    # Assign each point to an interval, and compute its position in it
    modfs = np.modf((delta_c * k_size).ravel())
    modfs = zip(modfs[0], modfs[1].astype(int))

    # Create the X matrix for least squares
    rho = sparse.lil_array((size**2, k_size + 1), dtype=float)
    for index, (ratio, position) in enumerate(modfs):
        rho[index, position : position + 2] = (1 - ratio, ratio)
    rho = rho[:, 1:-1]
    rho = rho.tocsc()

    def gaussian(x_val, sigma):
        return np.exp(-0.5 * (x_val / sigma) ** 2)

    delta_c -= delta_c >= 0.5

    for index, c_delta in enumerate(delta_c):
        # Use gaussian weighted errors
        weights = np.sqrt(gaussian(c_delta, w_width))
        weights = np.repeat(weights, size)

        w_rho = rho * weights[:, None]
        w_dist = distances.ravel() * weights

        # Solve the linear least squares unconstrained
        try:
            d_k = (sparse.linalg.inv(w_rho.T.tocsc() @ w_rho) @ w_rho.T) @ w_dist
        except RuntimeError:
            d_k = np.linalg.lstsq(w_rho.toarray(), w_dist, rcond=None)[0]

        # Find the position of the maximum
        k_max = np.argmax(d_k)

        # Create the matrices for quadratic solver
        q_cone = 2 * w_rho.T.tocsc() @ w_rho
        c_cone = -2 * w_dist.T @ w_rho

        a_cone = a_cone_template.copy()
        a_cone[k_max + 1 :, :] *= -1

        b_cone = np.zeros(a_cone.shape[0])

        data = dict(P=q_cone, A=a_cone, b=b_cone, c=c_cone)
        cone = dict(l=a_cone.shape[0])

        # Solve the constrained problem
        solver = scs.SCS(data, cone, eps_abs=1e-7, eps_rel=1e-7, verbose=False)
        sol = solver.solve(x=d_k)

        if not sol["info"]["status"] == "solved":
            raise RuntimeWarning("Conic solver did not converge.")

        # The first distance is imposed to be 0
        d_i_k[index, 1:-1] = sol["x"]
        # pivots[index] = k_max + 1

    return d_i_k, sol["info"]["res_pri"]


@partial(jax.jit, static_argnames=["sequences"])
def _regu_loss(phases, sequences):
    """
    Compute the regularization loss based on phase delta deviation.

    Parameters
    ----------
    phases : jax.ndarray (float) [N]
        The input phases.
    sequences : tuple(int)
        The acquisition sequence lengths.

    Returns
    -------
    float
        The loss value.
    """
    regu_loss = 0
    seq_start = 0

    n_measures = phases.size - 2 * len(sequences)

    for seq_len in sequences:
        seq_phases = phases[seq_start : seq_start + seq_len]
        deltas = (seq_phases[1:] - seq_phases[:-1]) % 1
        deltas = (deltas[1:] - deltas[:-1]) ** 2

        regu_loss += deltas.sum()

        seq_start += seq_len

    regu_loss /= n_measures

    return regu_loss


@partial(jax.jit, static_argnames=["w_width"])
# @jax.jit
def _distance_loss(phases, distances, d_i_k, w_width):
    """
    Compute the loss based on the deviation between measured image distances and
    average image distance functions.

    Parameters
    ----------
    phases : jax.ndarray (float) [N]
        The input phases.
    distances : jax.jndarray (float) [N, N]
        The pairwise image distances.
    d_i_k : jax.jndarray (float) [N, K]
        The average image distance functions.
    w_width : float
        The gaussian window width.
    """

    def gaussian(x_val, sigma):
        return jnp.exp(-0.5 * (x_val / sigma) ** 2)

    k_range = jnp.arange(d_i_k.shape[1]) / (d_i_k.shape[1] - 1)
    delta_phases = (phases[None, :] - phases[:, None]) % 1
    diff_phases = delta_phases.copy()
    diff_phases -= diff_phases >= 0.5
    w_i_j = gaussian(diff_phases, w_width)

    def body_func(index, loss_value):
        d_i_j = jnp.interp(delta_phases[index], k_range, d_i_k[index])
        diffs = (d_i_j - distances[index]) ** 2
        loss_value += jnp.sum(diffs * w_i_j[index])
        return loss_value

    dist_loss = jax.lax.fori_loop(0, delta_phases.shape[0], body_func, 0.0)

    dist_loss /= jnp.sum(w_i_j)

    return dist_loss


def _gradient_descent_phase(
    phases,
    regu_lambda,
    learning_rate,
    a_tol,
    grad_iter,
    distances,
    d_i_k,
    w_width,
    sequences,
    optim=None,
):
    """
    Minimize the loss using gradient descent.

    Parameters
    ----------
    phases : jax.ndarray (float) [N]
        The input phases.
    regu_lambda : float
        The regularization strength, in [0,1].
    learning_rate : float
        The learning rate for gradient descent.
    a_tol : float
        The absolute tolerance for convergence.
    grad_iter : int
        The maximum number of iterations.
    distances : jax.ndarray (float) [N, N]
        The pairwise image distances.
    d_i_k : jax.ndarray (float) [N, K]
        The estimated distance functions.
    w_width : float
        The window width.
    sequences : tuple(float)
        The aqcuisition sequences length.
    optim : tuple (optax.GradientTransformation, optax state), optional
        The optimizer and its state. If left empty, adam is used (fresh init).

    Returns
    -------
    phases : jax.ndarray (float) [N]
        The estimated phases.
    losses : tuple
        The final losses, in order: (distance, regularization)
    optim : tuple (optax.GradientTransformation, optax state)
        The optimizer and its state. Provide it to the next iteration for
        smoother convergence.
    """

    d_r_loss = jax.value_and_grad(_regu_loss)
    d_d_loss = jax.value_and_grad(_distance_loss)

    if optim is None:
        optim = optax.adam(learning_rate)
        opt_state = optim.init(phases)
    else:
        optim, opt_state = optim

    for _ in range(grad_iter):
        r_loss, r_grad = d_r_loss(phases, sequences)
        d_loss, d_grad = d_d_loss(phases, distances, d_i_k, w_width)

        # total_loss = (1 - regu_lambda) * d_loss + regu_lambda * r_loss
        total_grad = (1 - regu_lambda) * d_grad + regu_lambda * r_grad

        updates, opt_state = optim.update(total_grad, opt_state, phases)
        phases = optax.apply_updates(phases, updates)

        if jnp.abs(updates).mean() < a_tol:
            break

    return phases, (float(d_loss), float(r_loss)), (optim, opt_state)


def _correction_pass(
    c_phases,
    distances,
    seq_lengths,
    k_size,
    w1_width,
    w2_width,
    regu_lambda,
    max_iter=100,
    a_tol=1e-4,
    grad_iter=100,
    learning_rate=1e-3,
):
    """
    Inner implementation of the phase correction algorithm.

    Parameters
    ----------
    c_phases : jax.ndarray (float) [N]
        The initial estimate for the phases, given in acquisition order.
    distances : jax.ndarray (float) [N, N]
        The image-to-image distance pairs.
    seq_lengths : tuple (int)
        The lengths of sequences of images acquired consecutively, if multiple
        sequences have been acquired. The total sum of sequence lengths must be
        equal to the total number of images N.
    k_size : int
        Number of poins used to sample distance functions.
    w1_width : float
        Gaussian window width (standard deviation) used to weight contributions
        of neighbours for distance function estimation.
    w2_width : float
        Gaussian window width (standard deviation) used to weight contributions
        of neighbours for gradient descent.
    regu_lambda : float
        The regularization strength.
    max_iter : int, optional
        Maximum iterations of the algorithm, by default 100
    a_tol : float, optional
        Absolute tolerance used to define convergence, by default 1e-4
    grad_iter : int, optional
        Max number of iterations of the gradient descent, by default 100
    learning_rate : float, optional
        The learning rate of the gradient descent, by default 1e-3

    Returns
    -------
    phases : numpy.ndarray (float) [N]
        The corrected phases
    status : tuple
        The convergence status, containing:
        converged : bool
            If the algorithm converged.
        n_iter : int
            The number of iterations ran.
        losses : tuple(float)
            The final distance and regularization losses, in that order.

    Raises
    ------
    ValueError
    """

    size = c_phases.size
    d_i_k = np.empty((size, k_size + 1))
    optim = None

    converged = False
    for out_iter in range(max_iter):
        d_i_k, _ = estimate_di(c_phases, distances, k_size, w1_width, d_i_k=d_i_k)
        jn_d_i_k = jnp.asarray(d_i_k)

        c_phases_old = c_phases

        c_phases, losses, optim = _gradient_descent_phase(
            c_phases,
            regu_lambda,
            learning_rate,
            a_tol,
            grad_iter,
            distances,
            jn_d_i_k,
            w2_width,
            seq_lengths,
            optim=optim,
        )

        if jnp.abs(c_phases_old - c_phases).mean() < a_tol:
            converged = True
            break

    status = (converged, out_iter, losses)
    return c_phases, status


def phase_correction(
    phases,
    images,
    seq_lengths=None,
    w_width=3e-2,
    k_size=None,
    regu_lambda=0.5,
    precision_passes=0,
    **kwargs,
):
    """
    Correct a sequence of phases corresponding to a periodic signal using both
    image-to-image and phase distances to find non uniform samplings. This
    method assumes that the images have been acquired at regular time
    intervals in one or multiple consecutive sequences.

    Parameters
    ----------
    phases : numpy.ndarray (float) [N]
        The initial estimate for the phases, given in acquisition order.
    images : numpy.ndarray (float) [N, ...]
        The acquired images, in order of acquisition.
        phases[i] should correspond to images[i].
    seq_lengths : tuple (int)
        The lengths of sequences of images acquired consecutively, if multiple
        sequences have been acquired. The total sum of sequence lengths must be
        equal to the total number of images N. By default, None: considers the
        entire sequence to be consecutive.
    w_width : float, optional
        Gaussian window width (standard deviation) used to weight contributions
        of neighbours, by default 3e-2
    k_size : int, optional
        Number of poins used to sample distance functions, by default None
        (uses N/2)
    regu_lambda : float, optional
        The regularization strength, by default 0.5
    precision_passes : int, optional
        The number of precision passes to perform. At each pass, window width is
        halved and k_size is doubled. By default 0.
    ** kwargs : named arguments passed to the `.vhf._correction_pass` function.
        max_iter : int, optional
            Maximum iterations of the algorithm, by default 100
        a_tol : float, optional
            Absolute tolerance used to define convergence, by default 1e-4
        grad_iter : int, optional
            Max number of iterations of the gradient descent, by default 100
        learning_rate : float, optional
            The learning rate of the gradient descent, by default 1e-3

    Returns
    -------
    phases : numpy.ndarray (float) [N]
        The corrected phases
    status : tuple
        The convergence status, containing:
        converged : bool. If the algorithm converged.
        n_iter : int. The number of iterations ran.
        losses : tuple(float). The final distance and regularization losses, in that order.

    Raises
    ------
    ValueError
    """

    size = phases.size

    if seq_lengths is None:
        seq_lengths = (size,)

    if sum(seq_lengths) != size:
        raise ValueError(
            "The sum of sequence lengths must be equal to the total number of images."
        )

    # Compute all pairwise distances between images
    distances = scidist.pdist(
        images.reshape(images.shape[0], -1), metric="minkowski", p=1
    )
    distances /= distances.max()
    distances = scidist.squareform(distances)
    distances = jnp.asarray(distances)

    if k_size is None:
        k_size = size // 2

    c_phases = jnp.asarray(phases)

    for p_pass in range(precision_passes + 1):
        if p_pass > 0:
            w_width /= 2
            k_size *= 2

        c_phases, status = _correction_pass(
            c_phases,
            distances,
            seq_lengths,
            k_size=k_size,
            w1_width=w_width,
            w2_width=w_width,
            regu_lambda=regu_lambda,
            **kwargs,
        )

    return np.asarray(c_phases), status


def l_curve(
    phases,
    images,
    seq_lengths=None,
    w_width=3e-2,
    k_size=None,
    n_points=21,
    iterations=50,
    **kwargs,
):
    """
    Correct a sequence of phases corresponding to a periodic signal using both
    image-to-image and phase distances to find non uniform samplings. This
    method assumes that the images have been acquired at regular time
    intervals in one or multiple consecutive sequences.

    Parameters
    ----------
    phases : numpy.ndarray (float) [N]
        The initial estimate for the phases, given in acquisition order.
    images : numpy.ndarray (float) [N, ...]
        The acquired images, in order of acquisition.
        phases[i] should correspond to images[i].
    seq_lengths : tuple (int)
        The lengths of sequences of images acquired consecutively, if multiple
        sequences have been acquired. The total sum of sequence lengths must be
        equal to the total number of images N. By default, None: considers the
        entire sequence to be consecutive.
    w_width : float, optional
        Gaussian window width (standard deviation) used to weight contributions
        of neighbours, by default 3e-2
    k_size : int, optional
        Number of poins used to sample distance functions, by default None
        (uses N/2)
    n_points : int, optional
        Number of values of lambda to compute for the L-curve. By default 21.
    iterations : int, optional
        Maximum number of iterations for the phase correction algorithm. By default 50.
    ** kwargs : named arguments passed to the `.vhf._correction_pass` function.
        a_tol : float, optional. Absolute tolerance used to define convergence, by default 1e-4
        learning_rate : float, optional. The learning rate of the gradient descent, by default 1e-3

    Returns
    -------
    losses: numpy.ndarray [2, n_points]
        The values of the losses for each lambda, given in the order
        (distance cost, regularization cost)
    lambdas: numpy.ndarray [n_points]
        The lambda values
    """

    size = phases.size

    if seq_lengths is None:
        seq_lengths = (size,)

    if sum(seq_lengths) != size:
        raise ValueError(
            "The sum of sequence lengths must be equal to the total number of images."
        )

    # Compute all pairwise distances between images
    distances = scidist.pdist(
        images.reshape(images.shape[0], -1), metric="minkowski", p=1
    )
    distances /= distances.max()
    distances = scidist.squareform(distances)
    distances = jnp.asarray(distances)

    if k_size is None:
        k_size = size // 2

    c_phases = jnp.asarray(phases)

    lambdas = np.linspace(0, 1, n_points)
    losses = []

    for l_val in lambdas:
        c_phases, status = _correction_pass(
            c_phases,
            distances,
            seq_lengths,
            k_size=k_size,
            w1_width=w_width,
            w2_width=w_width,
            regu_lambda=l_val,
            max_iter=iterations,
            grad_iter=iterations,
            **kwargs,
        )
        losses.append(status[-1])

    losses = np.array(losses).T
    return losses, lambdas


def vhf_phase_uncorrected(images, seq_lengths=None):
    """
    Use naive vhf to estimate the phases corresponding to a periodic
    signal sampled at regular time intervals, and its starting average
    frequency. The phases are estimated to be uniformly sampled, which can be
    improved by running the `.vhf.phase_correction` algorithm.

    Parameters
    ----------
    images : numpy.ndarray (float) [N, ...]
        The acquired images, in acquisition order.
    seq_lengths : tuple (int)
        The lengths of sequences of images acquired consecutively, if multiple
        sequences have been acquired. The total sum of sequence lengths must be
        equal to the total number of images N. By default, None: considers the
        entire sequence to be consecutive.

    Returns
    -------
    uni_phases : numpy.ndarray (float) [N]
        The estimated unitless phases corresponding to the input images.
    """

    size = images.shape[0]

    if seq_lengths is None:
        seq_lengths = (size,)

    if sum(seq_lengths) != size:
        raise ValueError(
            "The sum of sequence lengths must be equal to the total number of images."
        )

    uni_phases = vhf_phase_naive(images)

    uni_phases = orient_phases(uni_phases, seq_lengths, in_place=True)

    return uni_phases
