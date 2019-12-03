import mpi4py.MPI as MPI
import numpy as np
import numpy.lib.format as npformat

"""
This module implements ways to distribute operations in MPI communicators.
"""

_MPI_dtypes = {'float64': MPI.DOUBLE}


def distribute_bin(dimension, mpi_comm=None, rank=None, size=None):
    """
    Computes the start and stop indexes to split computations across a communicator.

    :param mpi_comm:
    :param dimension: the dimension of the work to split
    :return: bin start index, bin size
    """

    if mpi_comm is not None:
        rank = mpi_comm.Get_rank()
        size = mpi_comm.Get_size()

    elif rank is None or size is None:
        raise ValueError('Rank and size, or mpi_comm must be not None')

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

    base_type = to_mpi_datatype(dtype)
    stride = np.prod(shape[axis:], dtype=int)
    block = np.prod(shape[axis + 1:], dtype=int) * n_slices
    count = np.prod(shape[:axis], dtype=int)
    extent = block * base_type.Get_extent()[1]

    return base_type.Create_vector(count, block, stride).Create_resized(0, extent)


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
    i_extent = np.prod(shape[src_axis + 1:], dtype=int) * base_type.Get_extent()[1]

    inner_stride = base_type.Create_vector(i_count, i_block, i_stride).Create_resized(0, i_extent)

    o_count = np.prod(shape[:min_axis], dtype=int)
    o_block = block_size
    o_stride = np.prod(shape[min_axis:], dtype=int)
    o_extent = np.prod(shape[tgt_axis + 1:], dtype=int) * base_type.Get_extent()[1]

    outer_stride = inner_stride.Create_vector(o_count, o_block, o_stride).Create_resized(0, o_extent)

    return outer_stride


def load(file_name, axis, mpi_comm):
    """
    Load a numpy array across parallel jobs in the MPI communicator.
    The array is sliced along the chosen dimension.

    :param file_name:
    :param axis: dimension on which to slice
    :param mpi_comm:
    :return: array slice, shape of the full array
    """

    rank = mpi_comm.Get_rank()

    header = None
    if rank == 0:
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

    if not (-len(full_shape) <= axis < len(full_shape)):
        raise ValueError("Invalid axis {} for array of ndim {}".format(axis, len(full_shape)))

    if axis < 0:
        axis = len(full_shape) + axis

    i_start, bin_size = distribute_bin(full_shape[axis], mpi_comm)
 
    l_shape = list(full_shape)
    l_shape[axis] = bin_size

    l_array = np.empty(l_shape, dtype=dtype)

    slice_type = create_slice_view(axis, bin_size, shape=full_shape, dtype=dtype)
    slice_type.Commit()

    single_slice_extent = slice_type.Get_extent()[1]
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


def redistribute(array, src_dim, tgt_dim, full_shape, mpi_comm):
    """
    Redistribute an array along a different dimension.

    :param array: slice of the array to redistribute
    :param src_dim: initial distribution dimension
    :param tgt_dim: target distribution dimension
    :param full_shape: shape of the full array
    :param mpi_comm:
    :return:
    """

    raise NotImplementedError


generate_data = False

if generate_data:
    dims = [2, 3, 4, 5]
    source_array = np.empty(dims, dtype=np.float64)
    source_array.flat = np.arange(int(np.prod(dims)), dtype=np.float64)
    np.save('source.npy', source_array)
    exit(0)

source_array = np.load('source.npy')
# if MPI.COMM_WORLD.Get_rank() == 0:
# print("Here: \n{}\n\n".format(source_array))
# print('\n\n')
# print(source_array[:, :, 0:2, :])
# print('\n')
# exit(0)

file_name = 'source.npy'
axis = 3
comm = MPI.COMM_WORLD

from cbi_toolbox import arrays

sl_array, shape = load(file_name, axis, comm)
# print('{} \n{}\n\n'.format(comm.Get_rank(), sl_array))
# if comm.Get_rank() == 0:

rk = comm.Get_rank()
if rk < shape[axis]:
    start_i, bin_size = distribute_bin(shape[axis], comm)
    transposed_mat = arrays.transpose_dim_to(source_array, axis, 0)
    check_mat = transposed_mat[start_i:start_i+bin_size]
    check_mat = arrays.transpose_dim_to(check_mat, 0, axis)
    check = np.all(check_mat == sl_array)
    print(check)

# target_dim = 0
# out_array = redistribute(sl_array, axis, target_dim, shape, comm)
# print(out_array)
# print(np.all(out_array == source_array))
