import numpy as np
from cbi_toolbox import distributed

print('!!!!!!!!!!!!!!!!!!!!!!\n\n\tONGOING WORK\n\n!!!!!!!!!!!!!!!!!!!!!!')

source_name = 'test_source.npy'
save_name = 'test_save.npy'

source_array = None
if distributed.is_root_process():
    dims = [2, 3, 4]
    source_array = np.empty(dims, dtype=np.float64)
    source_array.flat = np.arange(int(np.prod(dims)), dtype=np.float64)
    np.save(source_name, source_array)

distributed.wait_all()

axis = 1

sl_array, f_shape = distributed.load(source_name, axis)

distributed.save(save_name, sl_array, axis, f_shape)

if distributed.is_root_process():
    test = np.load(save_name)
    print(np.all(source_array == test))


# rk = comm.Get_rank()
#
# target_dim = 0
# out_array = redistribute(sl_array, axis, target_dim, comm, f_shape)
# print("R{}\n{}\n".format(rk, out_array))
# print(np.all(out_array == source_array))

# axis = target_dim
# sl_array = out_array
# if rk < f_shape[axis]:
#     start_i, bin_s = distribute_bin(f_shape[axis], comm)
#     transposed_mat = arrays.transpose_dim_to(source_array, axis, 0)
#     check_mat = transposed_mat[start_i:start_i + bin_s]
#     check_mat = arrays.transpose_dim_to(check_mat, 0, axis)
#     check = np.all(check_mat == sl_array)
#     print(check)
