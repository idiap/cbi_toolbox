import numpy as np


def roll_array(array, shifts, axis=None):
    axis -= 1
    rolled = np.empty_like(array)

    for index, array_2D in enumerate(array):
        rolled[index, ...] = np.roll(array_2D, shifts[index], axis=axis)

    return rolled


def center_of_mass(array, axis=1):
    pos = np.arange(1, array.shape[axis] + 1)

    dims = np.arange(array.ndim)

    for index in range(axis):
        dims[axis - index] = dims[axis - index - 1]

    dims[0] = axis

    transposed = np.transpose(array, dims)

    return pos.dot(transposed) / array.sum(axis)
