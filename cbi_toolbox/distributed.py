"""
The distributed module allows to distribute operations in MPI communicators.
"""

# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by François Marelli <francois.marelli@idiap.ch>

# This file is part of CBI Toolbox.

# CBI Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.

# CBI Toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with CBI Toolbox. If not, see <http://www.gnu.org/licenses/>.

import mpi4py.MPI as MPI
import numpy as np
import numpy.lib.format as npformat
from cbi_toolbox import utils


_MPI_dtypes = {'float64': MPI.DOUBLE}


def get_size(mpi_comm=MPI.COMM_WORLD):
    """
    Get the process count in the communicator.

    Parameters
    ----------
    mpi_comm : mpi4py.MPI.Comm, optional
        The MPI communicator, by default MPI.COMM_WORLD.

    Returns
    -------
    int
        The size of the MPI communicator.
    """

    return mpi_comm.Get_size()


def is_root_process(mpi_comm=MPI.COMM_WORLD):
    """
    Check if the current process is root.

    Parameters
    ----------
    mpi_comm : mpi4py.MPI.Comm, optional
        The MPI communicator, by default MPI.COMM_WORLD.

    Returns
    -------
    bool
        True if the current process is the root of the communicator.
    """

    return mpi_comm.Get_rank() == 0


def get_rank(mpi_comm=MPI.COMM_WORLD):
    """
    Get this process number in the communicator.

    Parameters
    ----------
    mpi_comm : mpi4py.MPI.Comm, optional
        The communicator, by default MPI.COMM_WORLD.

    Returns
    -------
    int
        The rank of the process.
    """

    return mpi_comm.Get_rank()


def wait_all(mpi_comm=MPI.COMM_WORLD):
    """
    Wait for all processes to reach this line (MPI barrier)
    This is just a wrapper for ease.

    Parameters
    ----------
    mpi_comm : mpi4py.MPI.Comm, optional
        The communicator, by default MPI.COMM_WORLD.
    """

    mpi_comm.Barrier()


def distribute_bin(dimension, mpi_comm=MPI.COMM_WORLD, rank=None, size=None):
    """
    Computes the start index and bin size to evenly split array-like data into
    multiple bins.

    Parameters
    ----------
    dimension : int
        The size of the array to distribute.
    mpi_comm : mpi4py.MPI.Comm, optional
        The communicator, by default MPI.COMM_WORLD.
    rank : int, optional
        The rank of the process, by default None (taken from communicator).
    size : int, optional
        The size of the communicator (number of splits to distribute to), by
        default None (taken from the communicator).

    Returns
    -------
    (int, int)
        The start index of this bin, and its size.
        The distributed data should be array[start:start + bin_size].

    Raises
    ------
    ValueError
        If rank and size are not both given or both None.
    """

    if rank is None and size is None:
        rank = mpi_comm.Get_rank()
        size = mpi_comm.Get_size()

    if rank is None or size is None:
        raise ValueError('Rank and size must be both given, or None')

    if size > dimension:
        size = dimension

    if rank >= size:
        return 0, 0

    bin_size = dimension // size
    large_bin_number = dimension - bin_size * size

    bin_index = 0

    if rank < large_bin_number:
        bin_size += 1
    else:
        bin_index += large_bin_number

    bin_index += rank * bin_size

    return bin_index, bin_size


def distribute_bin_all(dimension, mpi_comm=MPI.COMM_WORLD, size=None):
    """
    Computes the start indexes and bin sizes of all splits to distribute
    computations across a communicator.

    Parameters
    ----------
    dimension : int
        the size of the array to be distributed
    mpi_comm : mpi4py.MPI.Comm, optional
        the communicator, by default MPI.COMM_WORLD
    size : int, optional
        the size of the communicator, by default None (taken from communicator)

    Returns
    -------
    ([int], [int])
        The list of start indexes and the list of bin sizes to distribute data.
    """

    if size is None:
        size = mpi_comm.Get_size()

    original_size = size
    if size > dimension:
        size = dimension

    bin_size = dimension // size
    large_bin_number = dimension - bin_size * size

    bin_index = 0
    bin_indexes = []
    bin_sizes = []

    for j_index in range(original_size):
        if j_index >= size:
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


def to_mpi_datatype(np_datatype):
    """
    Returns the MPI datatype corresponding to the numpy dtype provided.


    Parameters
    ----------
    np_datatype : numpy.dtype or str
        The numpy datatype, or name.

    Returns
    -------
    mpi4py.MPI.Datatype
        The corresponding MPI datatype.

    Raises
    ------
    NotImplementedError
        If the numpy datatype is not listed in the conversion table.
    """
    if isinstance(np_datatype, np.dtype):
        dtype = np_datatype.name
    else:
        dtype = np_datatype
    try:
        return _MPI_dtypes[dtype]
    except KeyError:
        raise NotImplementedError(
            'Type not in conversion table: {}'.format(dtype))


def create_slice_view(axis, n_slices, array=None, shape=None, dtype=None):
    """
    Create a MPI vector datatype to access given slices of a non distributed
    array. If the array is not provided, its shape and dtype must be
    specified.

    Parameters
    ----------
    axis : int
        The axis on which to slice.
    n_slices : int
        How many contiguous slices to take.
    array : numpy.ndarray, optional
        The array to slice, by default None (then shape and dtype must be given).
    shape : the shape of the array to slice, optional
        The shape of the array, by default None.
    dtype : numpy.dtype or str, optional
        The datatype of the array, by default None.

    Returns
    -------
    mpi4py.MPI.Datatype
        The strided datatype allowing to access slices in the array.

    Raises
    ------
    ValueError
        If array, shape and dtype are all None.
    """

    if array is not None:
        shape = array.shape
        dtype = array.dtype

    elif shape is None or dtype is None:
        raise ValueError("array, or shape and dtype must be not None")

    axis = utils.positive_index(axis, len(shape))

    base_type = to_mpi_datatype(dtype)
    stride = np.prod(shape[axis:], dtype=int)
    block = np.prod(shape[axis + 1:], dtype=int) * n_slices
    count = np.prod(shape[:axis], dtype=int)
    extent = block * base_type.extent

    return base_type.Create_vector(count, block, stride).Create_resized(0, extent)


def compute_vector_extent(axis, array=None, shape=None, dtype=None):
    """
    Compute the extent in bytes of a sliced view of a given array.

    Parameters
    ----------
    axis : int
        Axis on which the slices are taken.
    array : numpy.ndarray, optional
        The array to slice, by default None (then shape and dtype must be given).
    shape : the shape of the array to slice, optional
        The shape of the array, by default None.
    dtype : numpy.dtype or str, optional
        The datatype of the array, by default None.

    Returns
    -------
    int
        The extent of the slices underlying data.

    Raises
    ------
    ValueError
        If array, shape and dtype are all None.
    """

    if array is not None:
        shape = array.shape
        dtype = array.dtype

    elif shape is None or dtype is None:
        raise ValueError("array, or shape and dtype must be not None")

    ndims = len(shape)
    axis = utils.positive_index(axis, ndims)

    base_type = to_mpi_datatype(dtype)
    return np.prod(shape[axis + 1:], dtype=int) * base_type.extent


def create_vector_type(src_axis, tgt_axis, array=None, shape=None, dtype=None,
                       block_size=1):
    """
    Create a MPI vector datatype to communicate a distributed array and split it
    along a different axis.

    Parameters
    ----------
    src_axis : int
        The original axis on which the array is distributed.
    tgt_axis : int
        The axis on which the array is to be distributed.
    array : numpy.ndarray, optional
        The array to slice, by default None (then shape and dtype must be given).
    shape : the shape of the array to slice, optional
        The shape of the array, by default None.
    dtype : numpy.dtype or str, optional
        The datatype of the array, by default None.
    block_size : int, optional
        The size of the distributed bin, by default 1.

    Returns
    -------
    mpi4py.MPI.Datatype
        The vector datatype used for transmission/reception of the data.

    Raises
    ------
    ValueError
        If array, shape and dtype are all None.
    ValueError
        If the source and destination axes are the same.
    NotImplementedError
        If the array has more than 4 axes (should work, but tests needed).
    ValueError
        If the block size is bigger than the source axis.
    """

    if array is not None:
        shape = array.shape
        dtype = array.dtype

    elif shape is None or dtype is None:
        raise ValueError("array, or shape and dtype must be not None")

    ndims = len(shape)
    src_axis = utils.positive_index(src_axis, ndims)
    tgt_axis = utils.positive_index(tgt_axis, ndims)

    if src_axis == tgt_axis:
        raise ValueError(
            "Source and target are identical, no communication should be "
            "performed")

    if len(shape) > 4:
        raise NotImplementedError(
            "This has never been tested for arrays with more than 4 axes.\n"
            "It will probably work, but please run a test before"
            "(and if works, tell me!)")

    if block_size > shape[src_axis]:
        raise ValueError(
            "Block size cannot be bigger than the dimension of the source axis")

    base_type = to_mpi_datatype(dtype)

    min_axis = min(src_axis, tgt_axis)
    max_axis = max(src_axis, tgt_axis)

    i_count = np.prod(shape[min_axis + 1:max_axis], dtype=int)
    i_block = np.prod(shape[max_axis + 1:], dtype=int)
    i_stride = np.prod(shape[max_axis:], dtype=int)
    i_extent = np.prod(shape[src_axis + 1:], dtype=int) * base_type.extent

    # only happens if the array is empty, avoid division by zero warnings
    if i_extent == 0:
        i_extent = 1

    inner_stride = base_type.Create_vector(
        i_count, i_block, i_stride).Create_resized(0, i_extent)

    o_count = np.prod(shape[:min_axis], dtype=int)
    o_block = block_size
    o_stride = (np.prod(shape[min_axis:], dtype=int)
                * base_type.extent) // i_extent
    o_extent = np.prod(shape[tgt_axis + 1:], dtype=int) * base_type.extent

    outer_stride = inner_stride.Create_vector(
        o_count, o_block, o_stride).Create_resized(0, o_extent)

    return outer_stride


def gather_full_shape(array, axis, mpi_comm=MPI.COMM_WORLD):
    """
    Gather the full shape of an array distributed across an MPI communicator
    along a given axis.

    Parameters
    ----------
    array : numpy.ndarray
        The distributed array.
    axis : int
        The axis on which the array is distributed.
    mpi_comm : mpi4py.MPI.Comm, optional
        The communicator, by default MPI.COMM_WORLD.

    Raises
    ------
    NotImplementedError
        This is not implemented yet.
    """

    raise NotImplementedError


def load(file_name, axis, mpi_comm=MPI.COMM_WORLD):
    """
    Load a numpy array across parallel jobs in the MPI communicator.
    The array is sliced along the chosen dimension, with minimal bandwidth.

    Parameters
    ----------
    file_name : str
        The numpy array file to load.
    axis : int
        The axis on which to distribute the array.
    mpi_comm : mpi4py.MPI.Comm, optional
        The MPI communicator used to distribute, by default MPI.COMM_WORLD.

    Returns
    -------
    (numpy.ndarray, tuple(int))
        The distributed array, and the size of the full array.

    Raises
    ------
    ValueError
        If the numpy version used to save the file is not supported.
    NotImplementedError
        If the array is saved in Fortran order.
    """

    header = None
    if is_root_process(mpi_comm):
        with open(file_name, 'rb') as fp:
            version, _ = npformat.read_magic(fp)

            if version == 1:
                header = npformat.read_array_header_1_0(fp)
            elif version == 2:
                header = npformat.read_array_header_2_0(fp)
            else:
                raise ValueError(
                    "Invalid numpy format version: {}".format(version))

            header = *header, fp.tell()

    header = mpi_comm.bcast(header, root=0)
    full_shape, fortran, dtype, header_offset = header

    if fortran:
        raise NotImplementedError(
            "Fortran-ordered (column-major) arrays are not supported")

    ndims = len(full_shape)
    axis = utils.positive_index(axis, ndims)

    i_start, bin_size = distribute_bin(full_shape[axis], mpi_comm)

    l_shape = list(full_shape)
    l_shape[axis] = bin_size

    l_array = np.empty(l_shape, dtype=dtype)

    slice_type = create_slice_view(
        axis, bin_size, shape=full_shape, dtype=dtype)
    slice_type.Commit()

    single_slice_extent = slice_type.extent
    if bin_size != 0:
        single_slice_extent /= bin_size

    displacement = header_offset + i_start * single_slice_extent
    base_type = to_mpi_datatype(l_array.dtype)

    fh = MPI.File.Open(mpi_comm, file_name, MPI.MODE_RDONLY)
    fh.Set_view(displacement, filetype=slice_type)

    fh.Read_all([l_array, l_array.size, base_type])
    fh.Close()
    slice_type.Free()

    return l_array, full_shape


def save(file_name, array, axis, full_shape=None, mpi_comm=MPI.COMM_WORLD):
    """
    Save a numpy array from parallel jobs in the MPI communicator.
    The array is gathered along the chosen dimension.

    Parameters
    ----------
    file_name : str
        The numpy array file to load.
    array : numpy.ndarray
        The distributed array.
    axis : int
        The axis on which to distribute the array.
    full_shape : tuple(int), optional
        The size of the full array, by default None.
    mpi_comm : mpi4py.MPI.Comm, optional
        The MPI communicator used to distribute, by default MPI.COMM_WORLD.
    """

    if full_shape is None:
        full_shape = gather_full_shape(array, axis, mpi_comm)

    axis = utils.positive_index(axis, len(full_shape))

    header_offset = None
    if is_root_process(mpi_comm):
        header_dict = {'shape': full_shape,
                       'fortran_order': False,
                       'descr': npformat.dtype_to_descr(array.dtype)}

        with open(file_name, 'wb') as fp:
            try:
                npformat.write_array_header_1_0(fp, header_dict)
            except ValueError:
                npformat.write_array_header_2_0(fp, header_dict)

            header_offset = fp.tell()
    header_offset = mpi_comm.bcast(header_offset, root=0)

    i_start, bin_size = distribute_bin(full_shape[axis], mpi_comm)

    slice_type = create_slice_view(
        axis, bin_size, shape=full_shape, dtype=array.dtype)
    slice_type.Commit()

    single_slice_extent = slice_type.extent
    if bin_size != 0:
        single_slice_extent /= bin_size

    displacement = header_offset + i_start * single_slice_extent
    base_type = to_mpi_datatype(array.dtype)

    fh = MPI.File.Open(mpi_comm, file_name, MPI.MODE_WRONLY | MPI.MODE_APPEND)
    fh.Set_view(displacement, filetype=slice_type)

    fh.Write_all([array, array.size, base_type])
    fh.Close()
    slice_type.Free()


def redistribute(array, src_axis, tgt_axis, full_shape=None,
                 mpi_comm=MPI.COMM_WORLD):
    """
    Redistribute an array along a different dimension.

    Parameters
    ----------
    array : numpy.ndarray
        The distributed array.
    src_axis : int
        The original axis on which the array is distributed.
    tgt_axis : int
        The axis on which the array is to be distributed.
    full_shape : tuple(int), optional
        The full shape of the array, by default None.
    mpi_comm : mpi4py.MPI.Comm, optional
        The MPI communicator used to distribute, by default MPI.COMM_WORLD.

    Returns
    -------
    np.ndarray
        The array distributed along the new axis.
    """

    if full_shape is None:
        full_shape = gather_full_shape(array, src_axis, mpi_comm)

    ndims = len(full_shape)
    src_axis = utils.positive_index(src_axis, ndims)
    tgt_axis = utils.positive_index(tgt_axis, ndims)

    if src_axis == tgt_axis:
        return array

    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()

    src_starts, src_bins = distribute_bin_all(full_shape[src_axis], mpi_comm)
    tgt_starts, tgt_bins = distribute_bin_all(full_shape[tgt_axis], mpi_comm)

    src_has_data = np.atleast_1d(src_bins)
    src_has_data[src_has_data > 0] = 1

    tgt_has_data = np.atleast_1d(tgt_bins)
    tgt_has_data[tgt_has_data > 0] = 1

    n_shape = list(full_shape)
    n_shape[tgt_axis] = tgt_bins[rank]
    n_array = np.empty(n_shape, dtype=array.dtype)

    send_datatypes = []
    recv_datatypes = []
    for ji in range(size):
        send_datatypes.append(create_vector_type(
            src_axis, tgt_axis, array, block_size=src_bins[rank]))
        recv_datatypes.append(create_vector_type(
            src_axis, tgt_axis, n_array, block_size=src_bins[ji]))

    send_extent = compute_vector_extent(tgt_axis, array)
    recv_extent = compute_vector_extent(src_axis, n_array)

    send_counts = np.multiply(tgt_bins, src_has_data[rank])
    send_displs = np.multiply(tgt_starts, send_extent)

    sendbuf = [array, send_counts, send_displs, send_datatypes]

    recv_counts = np.multiply(src_has_data, tgt_bins[rank])
    recv_displs = np.multiply(src_starts, recv_extent)
    recvbuf = [n_array, recv_counts, recv_displs, recv_datatypes]

    for ji in range(size):
        send_datatypes[ji].Commit()
        recv_datatypes[ji].Commit()

    mpi_comm.Alltoallw(sendbuf, recvbuf)

    for ji in range(size):
        send_datatypes[ji].Free()
        recv_datatypes[ji].Free()

    return n_array
