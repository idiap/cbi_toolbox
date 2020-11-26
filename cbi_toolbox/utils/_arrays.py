"""
This module contains utility functions to work on numpy arrays.
"""

import numpy as np


def roll_array(array, shifts, axis):
    """
    Roll an array over a given axis, with a different shift for each element
    along the first axis.
    This applies numpy.roll to each sub-array with the corresponding shift.

    Parameters
    ----------
    array : numpy.ndarray
        The original array.
    shifts : list(int)
        The shifts to apply, should be the same size as array.shape[0].
    axis : int
        The axis on which to apply the roll (>0).

    Returns
    -------
    numpy.ndarray
        The rolled array.
    """

    axis -= 1
    rolled = np.empty_like(array)

    for index, array_2D in enumerate(array):
        rolled[index, ...] = np.roll(array_2D, shifts[index], axis=axis)

    return rolled


def center_of_mass(array, axis):
    """
    Compute the center of mass of an array over a given axis.

    Parameters
    ----------
    array : numpy.ndarray
        The input array.
    axis : int
        The axis on which to compute the center of mass.

    Returns
    -------
    float
        The position of the center of mass.
    """

    pos = np.arange(array.shape[axis])

    axes = list(range(array.ndim))
    axes.remove(axis)
    pos = np.expand_dims(pos, axes)

    return np.tensordot(pos, array, (axis, axis)).squeeze() / array.sum(axis)


def make_broadcastable(array, target):
    """
    Append the required amount of empty dimensions to an array to make it
    broadcastable with respect to another.

    Parameters
    ----------
    array : numpy.ndarray
        The array to be broadcasted.
    target : numpy.ndarray
        The target array, reference for the broadcast size.

    Returns
    -------
    np.ndarray
        The broadcastable array.

    Raises
    ------
    ValueError
        If the target array has fewer dimensions than the source.
    """

    if array.ndim > target.ndim:
        raise ValueError("Target array must have equal or more dimensions.")

    broadcast_shape = np.ones(target.ndim, dtype=int)
    for ax, dim in enumerate(array.shape):
        broadcast_shape[ax] = dim
    return array.reshape(broadcast_shape)


def transpose_dim_to(array, src_dim, target_dim):
    """
    Transpose the array so that the given dimension moves to the specified
    position and the order of the rest is unchanged.
    This is a convenience wrapper around numpy.transpose.

    Inversing the transpose is done as such::

        >>> N1, N2 = 1, 3
        >>> a0 = numpy.ones([2, 3, 4, 5])
        >>> a1 = transpose_dim_to(a0, N1, N2)
        >>> a2 = transpose_dim_to(a1, N2, N1)
        >>> a0 == a2


    Parameters
    ----------
    array : numpy.ndarray
        The array to be transposed.
    src_dim : int
        The dimension to be moved.
    target_dim : int
        The target position of the dimension.

    Returns
    -------
    numpy.ndarray
        The transposed array.
    """

    dims = list(range(array.ndim))
    dims.remove(src_dim)
    dims.insert(target_dim, src_dim)
    return np.transpose(array, dims)


def threshold_crop(array, threshold, axis, summed=False):
    """
    Crop an array to a relative threshold on given axis. The returned array is
    the smallest array that contains all elements superior to the threshold.
    The thresholding can be done per element, or based on the sum over the
    non cropping axes.

    Parameters
    ----------
    array : numpy.ndarray
        The input array.
    threshold : float
        Relative threshold for cropping.
    axis : int
        Axis along which to crop.
    summed : bool, optional
        Compute the summed thresholds, by default False

    Returns
    -------
    numpy.ndaray
        The cropped array.
    """

    dims = list(range(array.ndim))
    dims.remove(axis)
    dims = tuple(dims)

    if summed:
        amplitude = array.sum(dims)
    else:
        amplitude = array.max(dims)

    amplitude = amplitude / amplitude.max()
    thresh = np.nonzero(amplitude > threshold)[0]

    slices = [slice(None)] * array.ndim
    slices[axis] = slice(thresh[0], thresh[-1]+1)
    slices = tuple(slices)
    return array[slices]


def positive_index(index, size):
    """
    Return a positive index from any index. If the index is positive, it is
    returned. If negative, size+index will be returned.

    Parameters
    ----------
    index : int
        The input index.
    size : int
        The size of the indexed dimension.

    Returns
    -------
    int
        A positive index corresponding to the input index.

    Raises
    ------
    ValueError
        If the given index is not valid for the given size.
    """

    if not -size <= index <= size:
        raise ValueError(
            "Invalid index {} for size {}".format(index, size))

    if index < 0:
        index = size + index

    return index
