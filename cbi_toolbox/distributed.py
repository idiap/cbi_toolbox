import mpi4py.MPI as MPI
import numpy as np
import numpy.lib.format as npformat
from cbi_toolbox import arrays

"""
This module implements ways to distribute operations in MPI communicators.
"""

_MPI_dtypes = {'float64': MPI.DOUBLE}


def is_root_process(mpi_comm=MPI.COMM_WORLD):
    """
    Check if current process is root.

    :param mpi_comm:
    :return:
    """

    return mpi_comm.Get_rank() == 0


def get_rank(mpi_comm=MPI.COMM_WORLD):
    """
    Get process rank.

    :param mpi_comm:
    :return:
    """

    return mpi_comm.Get_rank()


def wait_all(mpi_comm=MPI.COMM_WORLD):
    """
    Wait for all processes to reach this line (MPI barrier)
    This is just a wrapper for ease.

    :param mpi_comm:
    :return:
    """

    mpi_comm.Barrier()


def distribute_bin(dimension, mpi_comm=MPI.COMM_WORLD, rank=None, size=None):
    """
    Computes the start and stop indexes to split computations across a communicator.

    :param dimension: the dimension of the work to split
    :param mpi_comm:
    :param size: optional
    :param rank: optional
    :return: bin start index, bin size
    """

    if rank is None and size is None:
        rank = mpi_comm.Get_rank()
        size = mpi_comm.Get_size()

    if rank is None or size is None:
        raise ValueError('Rank and size must be given, or none')

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
    Computes the start and stop indexes of all jobs to split computations across a communicator.

    :param dimension: the dimension of the work to split
    :param mpi_comm:
    :param size: optional if mpi_comm given
    :return: bin start indexes list, bin sizes list
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

    :param np_datatype: numpy.dtype or string
    :return:
    """
    if isinstance(np_datatype, np.dtype):
        dtype = np_datatype.name
    else:
        dtype = np_datatype
    try:
        return _MPI_dtypes[dtype]
    except KeyError:
        raise NotImplementedError('Type not in conversion table: {}'.format(dtype))


def create_slice_view(axis, n_slices, array=None, shape=None, dtype=None):
    """
    Create a MPI vector datatype to access given slices of a non distributed array.

    :param axis:
    :param array:
    :param shape:
    :param dtype:
    :return:
    """

    if array is not None:
        shape = array.shape
        dtype = array.dtype

    elif shape is None or dtype is None:
        raise ValueError("array, or shape and dtype must be not None")

    axis = arrays.positive_axis(axis, len(shape))

    base_type = to_mpi_datatype(dtype)
    stride = np.prod(shape[axis:], dtype=int)
    block = np.prod(shape[axis + 1:], dtype=int) * n_slices
    count = np.prod(shape[:axis], dtype=int)
    extent = block * base_type.extent

    return base_type.Create_vector(count, block, stride).Create_resized(0, extent)


def compute_vector_extent(axis, array=None, shape=None, dtype=None):
    """
    Compute the extent in bytes of a sliced view of a given array

    :param axis:
    :param array:
    :param shape:
    :param dtype:
    :return:
    """

    if array is not None:
        shape = array.shape
        dtype = array.dtype

    elif shape is None or dtype is None:
        raise ValueError("array, or shape and dtype must be not None")

    ndims = len(shape)
    axis = arrays.positive_axis(axis, ndims)

    base_type = to_mpi_datatype(dtype)
    return np.prod(shape[axis + 1:], dtype=int) * base_type.extent


def create_vector_type(src_axis, tgt_axis, array=None, shape=None, dtype=None, block_size=1):
    """
    Create a MPI vector datatype to communicate a distributed array.

    :param src_axis:
    :param tgt_axis:
    :param array:
    :param shape:
    :param dtype:
    :param block_size:
    :return:
    """

    if array is not None:
        shape = array.shape
        dtype = array.dtype

    elif shape is None or dtype is None:
        raise ValueError("array, or shape and dtype must be not None")

    ndims = len(shape)
    src_axis = arrays.positive_axis(src_axis, ndims)
    tgt_axis = arrays.positive_axis(tgt_axis, ndims)

    if src_axis == tgt_axis:
        raise ValueError("Source and target are identical, no communication should be performed")

    if len(shape) > 4:
        raise NotImplementedError("This has never been tested for arrays with more than 4 axes.\n"
                                  "It will probably work, but please run a test before (and if works, tell me!)")

    if block_size > shape[src_axis]:
        raise ValueError("Block size cannot be bigger than the dimension of the source axis")

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

    inner_stride = base_type.Create_vector(i_count, i_block, i_stride).Create_resized(0, i_extent)

    o_count = np.prod(shape[:min_axis], dtype=int)
    o_block = block_size
    o_stride = (np.prod(shape[min_axis:], dtype=int) * base_type.extent) // i_extent
    o_extent = np.prod(shape[tgt_axis + 1:], dtype=int) * base_type.extent

    outer_stride = inner_stride.Create_vector(o_count, o_block, o_stride).Create_resized(0, o_extent)

    return outer_stride


def gather_full_shape(array, axis, mpi_comm=MPI.COMM_WORLD):
    """
    Gather the full shape of an array distributed across an MPI communicator along a given axis.

    :param array:
    :param axis:
    :param mpi_comm:
    :return:
    """

    raise NotImplementedError


def load(file_name, axis, mpi_comm=MPI.COMM_WORLD):
    """
    Load a numpy array across parallel jobs in the MPI communicator.
    The array is sliced along the chosen dimension.

    :param file_name:
    :param axis: dimension on which to slice
    :param mpi_comm:
    :return: array slice, shape of the full array
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
                raise ValueError("Invalid numpy format version: {}".format(version))

            header = *header, fp.tell()

    header = mpi_comm.bcast(header, root=0)
    full_shape, fortran, dtype, header_offset = header

    if fortran:
        raise NotImplementedError("Fortran-ordered (column-major) arrays are not supported")

    ndims = len(full_shape)
    axis = arrays.positive_axis(axis, ndims)

    i_start, bin_size = distribute_bin(full_shape[axis], mpi_comm)

    l_shape = list(full_shape)
    l_shape[axis] = bin_size

    l_array = np.empty(l_shape, dtype=dtype)

    slice_type = create_slice_view(axis, bin_size, shape=full_shape, dtype=dtype)
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

    :param file_name:
    :param array:
    :param axis: dimension on which the array is distributed
    :param full_shape:
    :param mpi_comm:
    :return: array slice, shape of the full array
    """

    if full_shape is None:
        full_shape = gather_full_shape(array, axis, mpi_comm)

    axis = arrays.positive_axis(axis, len(full_shape))

    header_offset = None
    if is_root_process(mpi_comm):
        header_dict = {'shape': full_shape,
                       'fortran_order': False,
                       'descr': npformat.dtype_to_descr(array.dtype)}

        with open(file_name, 'wb') as fp:
            npformat._write_array_header(fp, header_dict, None)
            header_offset = fp.tell()
    header_offset = mpi_comm.bcast(header_offset, root=0)

    i_start, bin_size = distribute_bin(full_shape[axis], mpi_comm)

    slice_type = create_slice_view(axis, bin_size, shape=full_shape, dtype=array.dtype)
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


def redistribute(array, src_axis, tgt_axis, full_shape=None, mpi_comm=MPI.COMM_WORLD):
    """
    Redistribute an array along a different dimension.

    :param array: slice of the array to redistribute
    :param src_axis: initial distribution dimension
    :param tgt_axis: target distribution dimension
    :param full_shape: shape of the full array
    :param mpi_comm:
    :return:
    """

    if full_shape is None:
        full_shape = gather_full_shape(array, src_axis, mpi_comm)

    ndims = len(full_shape)
    src_axis = arrays.positive_axis(src_axis, ndims)
    tgt_axis = arrays.positive_axis(tgt_axis, ndims)

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
        send_datatypes.append(create_vector_type(src_axis, tgt_axis, array, block_size=src_bins[rank]))
        recv_datatypes.append(create_vector_type(src_axis, tgt_axis, n_array, block_size=src_bins[ji]))

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
