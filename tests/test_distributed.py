import numpy as np
from cbi_toolbox import distributed
from cbi_toolbox import arrays
import os
import unittest

print('!!!!!!!!!!!!!!!!!!!!!!\n\n\tONGOING WORK\n\n!!!!!!!!!!!!!!!!!!!!!!')

source_name = 'test_source.npy'
save_name = 'test_save.npy'


def check_distributed_array(reference, array, axis):
    """
    Check that an array is correctly distributed on given axis

    :param reference: the full array
    :param array: the distributed array
    :param axis:
    :return: distribution is correct (bool)
    """

    s_index, b_size = distributed.distribute_bin(reference.shape[axis])
    check_mat = arrays.transpose_dim_to(source_array, axis, 0)
    check_mat = check_mat[s_index:s_index + b_size]
    check_mat = arrays.transpose_dim_to(check_mat, 0, axis)
    return np.all(check_mat == array)


def test_distributed_load(file_name, reference, axis):
    sl_array, f_shape = distributed.load(file_name, axis)
    check_distributed_array(reference, sl_array, axis)


def test_distributed_save(file_name, reference, array, axis):
    distributed.save(file_name, array, axis, reference.shape)
    if distributed.is_root_process():
        reloaded_array = np.load(save_name)
        return np.all(reference == reloaded_array)
    else:
        return True


source_array = None
if distributed.is_root_process():
    dims = [2, 3, 4, 5]
    source_array = np.empty(dims, dtype=np.float64)
    source_array.flat = np.arange(int(np.prod(dims)), dtype=np.float64)
    np.save(source_name, source_array)

distributed.wait_all()


os.remove(source_name)
os.remove(save_name)

# rk = comm.Get_rank()
#
# target_dim = 0
# out_array = redistribute(sl_array, axis, target_dim, comm, f_shape)
# print("R{}\n{}\n".format(rk, out_array))
# print(np.all(out_array == source_array))


class TestBsplines(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()


