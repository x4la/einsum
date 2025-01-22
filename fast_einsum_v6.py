import ctypes
import numpy as np
import os
import time
from numpy.lib.stride_tricks import as_strided
from math import prod
import test


def bmm_wrapper(A, B, shape_A, shape_B,
                offset_batch_a, offset_left_a, offset_contract_a,
                offset_batch_b, offset_contract_b, offset_right_b):
    # print(offset_batch_a, offset_left_a, offset_contract_a,
    #       offset_batch_b, offset_contract_b, offset_right_b)
    b, m, k = shape_A
    b_, k_, n = shape_B
    assert b == b_ and k == k_, f"Batch size and contract dimension must match for multiplication. {b} != {b_} or {k} != {k_}"
    assert A.dtype == B.dtype, f"Tensors must be of the same data type. Got {A.dtype} and {B.dtype}."

    current_dir = os.path.dirname(os.path.abspath(__file__))
    dll_path = os.path.join(current_dir, 'cmake-build-debug', 'libbmm_v6.dll')
    lib = ctypes.CDLL(dll_path)
    # set up C++ function
    lib.my_bmm.argtypes = [
        ctypes.c_void_p,  # Pointer to A
        ctypes.c_void_p,  # Pointer to B
        ctypes.c_void_p,  # Pointer to C
        ctypes.c_int,     # b
        ctypes.c_int,     # m
        ctypes.c_int,     # n
        ctypes.c_int,     # k
        ctypes.POINTER(ctypes.c_int),  # offset_batch_a
        ctypes.POINTER(ctypes.c_int),  # offset_left_a
        ctypes.POINTER(ctypes.c_int),  # offset_contract_a
        ctypes.POINTER(ctypes.c_int),  # offset_batch_b
        ctypes.POINTER(ctypes.c_int),  # offset_contract_b
        ctypes.POINTER(ctypes.c_int),  # offset_right_b
        ctypes.c_char_p   # d_type
    ]

    if A.dtype == np.float32:
        data_type = b"f"
    elif A.dtype == np.float64:
        data_type = b"d"
    # elif A.dtype == np.int32:
    #     data_type = b"i"
    # elif A.dtype == np.int64:
    #     data_type = b"l"
    else:
        raise TypeError(f"datatype {A.dtype} is not supported")

    # result array C
    C = np.zeros((b, m, n), dtype=A.dtype)

    # Call the C++ function
    lib.my_bmm(
        A.ctypes.data_as(ctypes.c_void_p),
        B.ctypes.data_as(ctypes.c_void_p),
        C.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(b),
        ctypes.c_int(m),
        ctypes.c_int(n),
        ctypes.c_int(k),
        offset_batch_a.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        offset_left_a.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        offset_contract_a.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        offset_batch_b.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        offset_contract_b.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        offset_right_b.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_char_p(data_type)
    )
    return C


def calc_sizes_block(dimensions):
    strides = np.ones_like(dimensions)
    for i in range(len(dimensions) - 2, -1, -1):
        strides[i] = strides[i + 1] * dimensions[i + 1]
    return strides


def calc_offsets_dim2(dims, strides, n):
    if n == 0:
        return np.zeros(1, dtype=np.int32)
    offsets = np.zeros(prod(dims), dtype=np.int32)
    sizes_block = calc_sizes_block(dims)

    def fill(level, start, value):
        if level == n-1:
            for i in range(dims[level]):
                offsets[start] = i * strides[level] + value
                start += 1
        else:
            for i in range(dims[level]):
                fill(level+1, start+i*sizes_block[level], value + i*strides[level])

    offsets_ptr = offsets.ctypes.data_as(ctypes.c_void_p),
    fill(0, 0, 0)
    return offsets


def calc_offsets(a_dims, b_dims, a_strides, b_strides, fuse_points_a, fuse_points_b):
    """wrapper function that gourps dimension into batch, left and right using fuse_points like this: abcdef -> a|bcd|ef.
    Then, for each of the 3 new dims, offsets are calculated by invoking calc_offsets_dim.
    """
    dims_a, strides_a, offsets_a = [0] * 3, [0] * 3, [0] * 3
    dims_b, strides_b, offsets_b = [0] * 3, [0] * 3, [0] * 3
    for i in range(3):
        dims_a[i], strides_a[i] = np.array(a_dims[fuse_points_a[i]:fuse_points_a[i+1]]), np.array(a_strides[fuse_points_a[i]:fuse_points_a[i+1]])
        offsets_a[i] = calc_offsets_dim(dims_a[i], strides_a[i])

    for i in range(3):
        dims_b[i], strides_b[i] = b_dims[fuse_points_b[i]:fuse_points_b[i+1]], np.array(b_strides[fuse_points_b[i]:fuse_points_b[i+1]])
        offsets_b[i] = calc_offsets_dim(dims_b[i], strides_b[i])

    return offsets_a, offsets_b


def calc_offsets2(a_dims, b_dims, a_strides, b_strides, fuse_points_a, fuse_points_b):
    """wrapper function that gourps dimension into batch, left and right using fuse_points like this: abcdef -> a|bcd|ef.
    Then, for each of the 3 new dims, offsets are calculated by invoking calc_offsets_dim.
    """
    dims_a, strides_a, offsets_a = [0] * 3, [0] * 3, [0] * 3
    dims_b, strides_b, offsets_b = [0] * 3, [0] * 3, [0] * 3
    for i in range(3):
        dims_a[i], strides_a[i] = np.array(a_dims[fuse_points_a[i]:fuse_points_a[i+1]]), np.array(a_strides[fuse_points_a[i]:fuse_points_a[i+1]])
        offsets_a[i] = calc_offsets_dim2(dims_a[i], strides_a[i], len(dims_a[i]))

    for i in range(3):
        dims_b[i], strides_b[i] = b_dims[fuse_points_b[i]:fuse_points_b[i+1]], np.array(b_strides[fuse_points_b[i]:fuse_points_b[i+1]])
        offsets_b[i] = calc_offsets_dim2(dims_b[i], strides_b[i], len(dims_b[i]))

    return offsets_a, offsets_b


def calc_offsets_dim(dims, strides):
    offsets = np.zeros(prod(dims), dtype=np.int32)
    n = len(dims)
    if n == 0:
        return np.zeros(1, dtype=np.int32)
    state = np.zeros(n, dtype=np.int32)
    for i in range(prod(dims)):
        # print("hi from calc_offsets", state.dtype, strides.dtype)
        offsets[i] = np.dot(state, strides)
        state[n-1] += 1
        for j in range(n-1, -1, -1):
            if state[j] != dims[j]:
                break
            else:
                state[j] = 0
                state[j-1] += 1
    return offsets


def fast_einsum(subscripts, a, b=None):
    if b is None:
        return single_einsum(subscripts, a)
    else:
        str_a, \
            desired_shape_a, \
            fuse_points_a, \
            str_b, \
            desired_shape_b, \
            fuse_points_b, \
            str_res, \
            shape_res = parse(subscripts, a.shape, b.shape)

        t0 = time.time()
        a, b = single_einsum(str_a, a), single_einsum(str_b, b)
        if np.isscalar(a):
            a = np.array([a])
        if np.isscalar(b):
            b = np.array([b])
        a_dims, b_dims = a.shape, b.shape
        a_strides, b_strides = tuple(s//a.itemsize for s in a.strides), tuple(s//b.itemsize for s in b.strides)
        t1 = time.time()
        offsets_a, offsets_b = calc_offsets(a_dims, b_dims, a_strides, b_strides, fuse_points_a, fuse_points_b)
        t2 = time.time()
        # offsets_a_tilda, offsets_b_tilda = calc_offsets2(a_dims, b_dims, a_strides, b_strides, fuse_points_a, fuse_points_b)
        t3 = time.time()
        c = bmm_wrapper(a, b, desired_shape_a, desired_shape_b, *offsets_a, *offsets_b).reshape(shape_res)
        t4 = time.time()
        c = single_einsum(str_res, c)
        return c
        return c, [0, t2-t1, t3-t2, t4-t3, 0]


def parse(s, shape_a, shape_b):
    str_ab, _, str_c_in = s.partition("->")
    str_a_in, _, str_b_in = str_ab.partition(",")
    str_a_in, str_b_in, str_c_in = str_a_in.strip(), str_b_in.strip(), str_c_in.strip()
    str_a = "".join(dict.fromkeys(str_a_in))
    str_b = "".join(dict.fromkeys(str_b_in))

    char_to_dim_a = {char: shape_a[str_a.index(char)] for char in str_a}
    char_to_dim_b = {char: shape_b[str_b.index(char)] for char in str_b}
    char_to_dim = char_to_dim_a | char_to_dim_b

    left_idxs = "".join([char for char in str_a if (char in str_c_in and char not in str_b)])
    right_idxs = "".join([char for char in str_b if (char in str_c_in and char not in str_a)])
    common_idx_ab = "".join([char for char in str_a if char in str_b])
    batch_idxs = "".join([char for char in common_idx_ab if char in str_c_in])
    contract_idxs = "".join([char for char in common_idx_ab if char not in str_c_in])

    batch_dim = prod([char_to_dim_a[char] for char in batch_idxs])
    left_dim = prod([char_to_dim_a[char] for char in left_idxs])
    right_dim = prod([char_to_dim_b[char] for char in right_idxs])
    contract_dim = prod([char_to_dim_a[char] for char in contract_idxs])

    str_a = str_a_in + "->" + batch_idxs + left_idxs + contract_idxs
    str_b = str_b_in + "->" + batch_idxs + contract_idxs + right_idxs
    str_res = batch_idxs + left_idxs + right_idxs + "->" +str_c_in

    shape_a = tuple([dim for dim in (batch_dim, left_dim, contract_dim)])
    shape_b = tuple([dim for dim in (batch_dim, contract_dim, right_dim)])
    shape_res = tuple([char_to_dim[char] for char in batch_idxs + left_idxs + right_idxs])

    fuse_points_a = [0, len(batch_idxs), len(left_idxs)+len(batch_idxs), len(contract_idxs)+len(left_idxs)+len(batch_idxs)]
    fuse_points_b = [0, len(batch_idxs), len(contract_idxs)+len(batch_idxs), len(right_idxs)+len(contract_idxs)+len(batch_idxs)]

    return str_a, shape_a, fuse_points_a, str_b, shape_b, fuse_points_b, str_res, shape_res


def remove_diags(s, strides, shape):
    """ contains the main logic to remove diagonals

        params: s           - str containing indices of a tensor.       Ex: "ijkijk"
                strides     - strides of tensor.                        Ex: (256, 128, 64, 32, 16, 8)
                shape       - shape of tensor.                          Ex: (2, 2, 2, 2, 2, 2)

        returns: new_s      - str with all multiple indices removed     Ex: "ijk"
                              (order: first appearance)
                 new_strides- stride for unique index is computed as    Ex: (288, 144, 72)
                              the sum of all strides for that index
                 new_shape  - new shape corresponding to new_s          Ex: (2, 2, 2)
    """
    seen = set()
    repeated = set()
    new_strides = {}
    new_shape = {}
    for i, char in enumerate(s):
        if char in seen:
            repeated.add(char)
            new_strides[char] = new_strides[char] + strides[i]  # this is the main logic
        else:
            seen.add(char)
            new_strides[char] = strides[i]
            new_shape[char] = shape[i]
    new_s = ''.join(new_strides.keys())
    new_strides = tuple(new_strides.values())
    new_shape = tuple(new_shape.values())
    return new_s, new_strides, new_shape


def move_right(str_a, str_right, a):
    char_to_idx = {char: str_a.index(char) for char in str_a}
    order = []
    new_str_a = []
    idx_first_right = 0
    for char in str_a:
        if char not in str_right:
            order.append(char_to_idx[char])
            new_str_a.append(char)
            idx_first_right += 1
    for char in str_right:
        order.append(char_to_idx[char])
        new_str_a.append(char)
    str_a = "".join(char for char in new_str_a)
    return np.transpose(a, tuple(order)), str_a, idx_first_right


def sum_right(a, idx, s):
    """sums all dimension of a tensor "a" right of some index "idx" by reshaping a into a matrix  """
    shape = a.shape
    dim_left = prod(shape[:idx])
    dim_right = prod(shape[idx:])
    a = np.reshape(a, (dim_left, dim_right))
    a = a.sum(axis=1)
    a = np.reshape(a, shape[:idx])
    return a, s[:idx]


def transpose(a, str_a, str_c):
    char_to_idx = {char: str_a.index(char) for char in str_a}
    order = []
    for char in str_c:
        order.append(char_to_idx[char])
    return np.transpose(a, tuple(order))


def single_einsum(subscripts, a):
    str_a, _, str_c = subscripts.partition("->")

    # first, get rid of diagonals
    if len(set(str_a)) != len(str_a):  # check for repeated idxs in str_a
        str_a, new_strides, new_shape = remove_diags(str_a, a.strides, a.shape)
        a = as_strided(a, strides=new_strides, shape=new_shape)     # create new view of a

    if str_c == "":
        return np.sum(a)

    # second, get rid of summed indices
    sum_idxs = ''.join([char for char in str_a if char not in str_c])
    if len(sum_idxs) > 0:
        a, str_a, idx_first_right = move_right(str_a, sum_idxs, a)
        a, str_a = sum_right(a, idx_first_right, str_a)

    # third, transpose to desired shape
    if str_a != str_c:
        a = transpose(a, str_a, str_c)
    return a
