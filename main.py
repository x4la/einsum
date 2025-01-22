from torch.backends.cudnn import benchmark

import fast_einsum_v1
import fast_einsum_v2
import fast_einsum_v3
import fast_einsum_v4
import trash_fast_einsum_v5
import fast_einsum_v6
import fast_einsum_v7
import fast_einsum_v8
import test
import benchmark.benchmark as bm
import numpy as np
import time
import torch
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # TO DO: find a better solution
np.set_printoptions(precision=4)
# A = np.random.rand(10, 10).astype(np.float32)
# B = np.random.rand(10, 10).astype(np.float32)
# res = fast_einsum_v4.test_mm(A, B)
# # print(res)
# C = A @ B
# # print(C)
# assert np.allclose(C, res)

# test.test_einsum(fast_einsum_v7.fast_einsum)
# bm.benchmark_einsum((np.einsum, fast_einsum_v1.fast_einsum, fast_einsum_v2.fast_einsum, fast_einsum_v3.fast_einsum, fast_einsum_v4.fast_einsum, torch.einsum))
bm.benchmark_einsum((torch.einsum, fast_einsum_v7.fast_einsum, fast_einsum_v8.fast_einsum))
# print(fast_einsum_v7.calc_offsets_dim2((3, 3), (1, 9)))
# test.test_bmm2(fast_einsum_v3.bmm_wrapper)

# A = np.ones((5, 5))
# B = np.zeros((6, 5))
# A[-1] = np.sum(A, axis=0)
# print(A)
# np.set_printoptions(suppress=True, precision=4)
# file_path = r"/benchmark/torch_v7_108.npy"  # Relative path to the file
# data = np.load(file_path)
# print(data)
#
# last_two_columns = data[:, -2:]
# print(last_two_columns)
# rowwise_min = np.min(last_two_columns, axis=1)
# print(rowwise_min)
# result = np.sum(rowwise_min)
# print(result-143.4762 )
