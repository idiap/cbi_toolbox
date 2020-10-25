import numpy as np


def roll_array(array, shifts, axis):
    axis -= 1
    rolled = np.empty_like(array)

    for index, array_2D in enumerate(array):
        rolled[index, ...] = np.roll(array_2D, shifts[index], axis=axis)

    return rolled


def center_of_mass(array, axis):
    pos = np.arange(array.shape[axis])

    axes = list(range(array.ndim))
    axes.remove(axis)
    pos = np.expand_dims(pos, axes)

    return np.tensordot(pos, array, (axis, axis)).squeeze() / array.sum(axis)


def make_broadcastable(array, target):
    """Append the required amount of dimensions to an array to make it broadcastable with respect to another"""

    assert array.ndim <= target.ndim

    broadcast_shape = np.ones(target.ndim, dtype=int)
    for ax, dim in enumerate(array.shape):
        broadcast_shape[ax] = dim
    return array.reshape(broadcast_shape)


def transpose_dim_to(array, src_dim, target_dim):
    """Transpose the array so that the given dim goes to the target and the order of the rest is unchanged"""
    dims = list(range(array.ndim))
    dims.remove(src_dim)
    dims.insert(target_dim, src_dim)
    return np.transpose(array, dims)


def threshold_crop(array, threshold, dim, summed=False):
    """Crop array to relative threshold on given axis"""
    dims = list(range(array.ndim))
    dims.remove(dim)
    dims = tuple(dims)

    if summed:
        amplitude = array.sum(dims)
    else:
        amplitude = array.max(dims)

    amplitude = amplitude / amplitude.max()
    thresh = np.nonzero(amplitude > threshold)[0]

    slices = [slice(None)] * array.ndim
    slices[dim] = slice(thresh[0], thresh[-1]+1)
    slices = tuple(slices)
    return array[slices]


def positive_index(index, size):
    """
    Return a positive index from any index.

    :param index:
    :param size:
    :return:
    """
    if not (-size <= index <= size):
        raise ValueError(
            "Invalid index {} for size {}".format(index, size))

    if index < 0:
        index = size + index

    return index
