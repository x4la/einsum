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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

bm.benchmark_einsum((np.einsum, fast_einsum_v1.fast_einsum, fast_einsum_v2.fast_einsum, fast_einsum_v3.fast_einsum,
                     fast_einsum_v4.fast_einsum, fast_einsum_v6.fast_einsum, fast_einsum_v7.fast_einsum, fast_einsum_v8.fast_einsum, torch.einsum))

