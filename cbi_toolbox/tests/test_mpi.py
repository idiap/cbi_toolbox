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
import unittest
import numpy as np
from cbi_toolbox import utils
try:
    from cbi_toolbox.parallel import mpi
    MPI_AVAILABLE = True
except ImportError:
    mpi = None
    MPI_AVAILABLE = False


def check_distributed_array(reference, array, axis):
    """
    Check that an array is correctly distributed on given axis

    :param reference: the full array
    :param array: the distributed array
    :param axis:
    :return: distribution is correct (bool)
    """

    s_index, b_size = mpi.distribute_mpi(reference.shape[axis])
    check_mat = utils.transpose_dim_to(reference, axis, 0)
    check_mat = check_mat[s_index:s_index + b_size]
    check_mat = utils.transpose_dim_to(check_mat, 0, axis)
    return np.all(check_mat == array)


def test_distributed_save(file_name, reference, array, axis):
    mpi.save(file_name, array, axis, reference.shape)
    if mpi.is_root_process():
        reloaded_array = np.load(file_name)
        return np.all(reference == reloaded_array)
    else:
        return True


@unittest.skipUnless(MPI_AVAILABLE, "MPI optional dependency not installed.")
class TestDistributedArrays(unittest.TestCase):
    ref_file = 'test_source.npy'
    tmp_file = 'test_save.npy'
    reference = None
    dims = [2, 3, 4, 5]
    ndims = None

    @classmethod
    def setUpClass(cls):
        cls.ndims = len(cls.dims)
        cls.reference = np.empty(cls.dims, dtype=np.float64)
        cls.reference.flat = np.arange(
            int(np.prod(cls.dims)), dtype=np.float64)

        if mpi.is_root_process():
            np.save(cls.ref_file, cls.reference)
        mpi.wait_all()

    @classmethod
    def tearDownClass(cls):
        if mpi.is_root_process():
            os.remove(cls.ref_file)

    def test_load_array(self):
        for axis in range(TestDistributedArrays.ndims):
            sl_array, f_shape = mpi.load(
                TestDistributedArrays.ref_file, axis)
            self.assertTrue(np.array_equal(
                f_shape, TestDistributedArrays.dims))
            self.assertTrue(check_distributed_array(
                TestDistributedArrays.reference, sl_array, axis))

    def test_distribute_array(self):
        for src_axis in range(TestDistributedArrays.ndims):
            sl_array, f_shape = mpi.load(
                TestDistributedArrays.ref_file, src_axis)

            for tgt_axis in range(TestDistributedArrays.ndims):
                temp_array = mpi.redistribute(
                    sl_array, src_axis, tgt_axis, f_shape)
                self.assertTrue(check_distributed_array(
                    TestDistributedArrays.reference, temp_array, tgt_axis))

    def test_save_array(self):
        for axis in range(TestDistributedArrays.ndims):
            sl_array, f_shape = mpi.load(
                TestDistributedArrays.ref_file, axis)
            mpi.save(TestDistributedArrays.tmp_file,
                     sl_array, axis, f_shape)

            if mpi.is_root_process():
                reloaded_array = np.load(TestDistributedArrays.tmp_file)
                self.assertTrue(np.array_equal(
                    TestDistributedArrays.reference, reloaded_array))
                os.remove(TestDistributedArrays.tmp_file)


if __name__ == "__main__":
    unittest.main()
