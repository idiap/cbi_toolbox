"""
The parallel package provides tools to split parallel computations.
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

import os
import concurrent.futures


def distribute_bin(dimension, rank, workers):
    """
    Computes the start index and bin size to evenly split array-like data into
    multiple bins.

    Parameters
    ----------
    dimension : int
        The size of the array to distribute.
    rank : int, optional
        The rank of the worker.
    workers : int, optional
        The total number of workers.

    Returns
    -------
    (int, int)
        The start index of this bin, and its size.
        The distributed data should be array[start:start + bin_size].
    """

    if workers > dimension:
        workers = dimension

    if rank >= workers:
        return 0, 0

    bin_size = dimension // workers
    large_bin_number = dimension - bin_size * workers

    bin_index = 0

    if rank < large_bin_number:
        bin_size += 1
    else:
        bin_index += large_bin_number

    bin_index += rank * bin_size

    return bin_index, bin_size


def distribute_bin_all(dimension, workers):
    """
    Computes the start indexes and bin sizes of all splits to distribute
    computations across multiple workers.

    Parameters
    ----------
    dimension : int
        the size of the array to be distributed
    workers : int, optional
        the amount of workers

    Returns
    -------
    ([int], [int])
        The list of start indexes and the list of bin sizes to distribute data.
    """

    original_size = workers
    if workers > dimension:
        workers = dimension

    bin_size = dimension // workers
    large_bin_number = dimension - bin_size * workers

    bin_index = 0
    bin_indexes = []
    bin_sizes = []

    for j_index in range(original_size):
        if j_index >= workers:
            bin_indexes.append(0)
            bin_sizes.append(0)
            continue

        l_bin_size = bin_size
        if j_index < large_bin_number:
            l_bin_size += 1

        bin_indexes.append(bin_index)
        bin_sizes.append(l_bin_size)

        bin_index += l_bin_size

    return bin_indexes, bin_sizes


def parallelize(func, size, workers=None):
    """
    Launches a function multiple times in parallel using multithreading.
    Useful only if the GIL is released in the parallelized function (this is
    the case for many ``numpy`` and ``scipy`` routines).

    Parameters
    ----------
    func : function (callable)
        The function that will be run in parallel. It must take 2 arguments,
        which are the returns of ``distribute_bin_all`` corresponding to the
        thread pool (the list of starting indexes of data bins, and the list of
        bin sizes).
    size : int
        The size of the array that will be split between workers.
    workers : int, optional
        The maximum number of workers, by default None (will be maximized for
        the system).

    Returns
    -------
    iterator
        An iterator containing the results of the function calls, in a random
        order (see concurrent.futures.ThreadPoolExecutor.map).
    """

    try:
        omp_threads = int(os.environ['OMP_NUM_THREADS'])
    except KeyError:
        omp_threads = os.cpu_count()

    uworkers = os.cpu_count() if workers is None else workers

    workers = min(omp_threads, len(
        os.sched_getaffinity(0)), size, uworkers)

    if workers == 1:
        out = func(0, size)
        outputs = [out]

    else:
        bins, bin_dims = distribute_bin_all(size, workers)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            outputs = executor.map(func, bins, bin_dims)

    return outputs
