import numpy as np
import time
import torch
from ctypes import c_int32, POINTER, cast, c_float, c_double
import benchmark.benchmark as bm
import fast_einsum_v7

testcases = {
    "one_input_1D": ["i->", "i->i"],
    "one_input_2D": ["ij->", "ij->i", "ij->j", "ij->ij", "ij->ji", "ii->", "ii->i"],
    "one_input_3D": ["ijk->",
                     "ijk->i", "ijk->j", "ijk->k",
                     "ijk->ik", "ijk->ki", "ijk->ij", "ijk->ji", "ijk->jk", "ijk->kj",
                     "ijk->ijk", "ijk->ikj", "ijk->jik", "ijk->jki", "ijk->kij", "ijk->kji"],
    "two_input_1D_1D": ["i,j->", "i,j->i","i,j->j", "i,j->ij", "i,j->ji"],
    "two_input_1D_2D": ["i,jk->", "i,jk->i", "i,jk->j", "i,jk->k", "i,jk->ij", "i,jk->ji", "i,jk->ik", "i,jk->ki",
                        "i,jk->jk", "i,jk->kj", "i,jk->ijk", "i,jk->ikj", "i,jk->jik", "i,jk->jki", "i,jk->kij",
                        "i,jk->kji",
                        "i,ik->", "i,ik->i", "i,ik->k", "i,ik->ik", "i,ik->ki",
                        "i,ji->", "i,ji->i", "i,ji->j", "i,ji->ij", "i,ji->ji",
                        "i,jj->", "i,jj->i", "i,jj->j", "i,jj->ij", "i,jj->ji",
                        "i,ii->", "i,ii->i"],
    "two_input_2D_1D": ["ij,k->", "ij,k->i", "ij,k->j", "ij,k->k", "ij,k->ij", "ij,k->ji", "ij,k->ik", "ij,k->ki",
                        "ij,k->jk", "ij,k->kj", "ij,k->ijk", "ij,k->ikj", "ij,k->jik", "ij,k->jki", "ij,k->kij",
                        "ij,k->kji",
                        "ij,i->", "ij,i->i", "ij,i->j", "ij,i->ij", "ij,i->ji",
                        "ij,j->", "ij,j->i", "ij,j->j", "ij,j->ij", "ij,j->ji",
                        "ii,k->", "ii,k->i", "ii,k->k", "ii,k->ik", "ii,k->ki",
                        "i,ii->", "i,ii->i"],
    "two_input_2d_2d": ["ij,kl->", "ij,kl->i", "ij,kl->j", "ij,kl->k", "ij,kl->l", "ij,kl->ij", "ij,kl->ji",
                        "ij,kl->ik",
                        "ij,kl->ki", "ij,kl->il", "ij,kl->li", "ij,kl->jk", "ij,kl->kj", "ij,kl->jl", "ij,kl->lj",
                        "ij,kl->kl", "ij,kl->lk",
                        "ij,kl->ikl", "ij,kj->ki", "ij,ik->ji", "ii,ii->i",
                        "iiij,jk->ijk", "iij,iji->ji"
                        ],  # only some special cases, not exhaustive
    "special_cases":   ["abij,abjk->baik", "abij,abjk->bika", "abij,abjk->bk", "aijb,abjk->baik"]
}

BMM_TESTS = {
    "test0": [(1, 3, 4), (1, 3, 4)],
    "test1": [(10, 10, 10), (10, 10, 10)],
    "test2": [(20, 5, 15), (20, 7, 15)],
    "test3": [(8, 30, 1), (8, 20, 1)],
    "test4": [(1, 9, 3), (1, 12, 3)],
    "test5": [(4, 1, 12), (4, 1, 12)],
    "test6": [(4, 10, 12), (4, 1, 12)],
    "test7": [(4, 1, 12), (4, 10, 12)],
    "test8": [(1, 1, 5), (1, 5, 5)],
    "test9": [(400, 200, 500), (400, 300, 500)]
}

BMM_TESTS2 = {
   "test0": [(1, 2, 4), (1, 4, 3)],
    "test1": [(10, 10, 10), (10, 10, 10)],
    "test2": [(20, 5, 15), (20, 15, 7)],
    "test3": [(8, 30, 1), (8, 1, 20)],
    "test4": [(1, 9, 3), (1, 3, 12)],
    "test5": [(4, 1, 12), (4, 12, 1)],
    "test6": [(4, 10, 12), (4, 12, 1)],
    "test7": [(4, 1, 12), (4, 12, 10)],
    "test8": [(1, 1, 5), (1, 5, 5)],
    "test9": [(400, 64*6, 64*10), (400, 64*10, 64*4)],
    "test10": [(10, 64, 64), (10, 64, 64)],
    "test11": [(20, 64, 128), (20, 128, 64)],
    "test12": [(11, 128, 64), (11, 64, 64*3)],
    "test13": [(400, 200, 500), (400, 500, 300)],
}

SIZE_OF_DIM = 5
np.random.seed(0)


def get_num_dims_input(contract_str):
    in_str, _, out_str = contract_str.partition("->")
    A, _, B = in_str.partition(",")
    return len(A), len(B)


def get_num_dims_output(contract_str):
    in_str, _, out_str = contract_str.partition("->")
    C = out_str.partition(",")
    return len(C)


def test_einsum(my_fct):
    time_np = 0
    time_my_fct = 0
    for case in testcases:
        print("\ntesting case", case)
        for contract_str in testcases[case]:
            print(contract_str)
            num_dims_input = get_num_dims_input(contract_str)
            dims_A = tuple([SIZE_OF_DIM] * num_dims_input[0])
            Tensors = [np.random.rand(*dims_A)]
            if num_dims_input[1] > 0:
                dims_B = tuple([SIZE_OF_DIM] * num_dims_input[1])
                Tensors.append(np.random.rand(*dims_B))
            t0 = time.time()
            np_sol = np.einsum(contract_str, *Tensors)
            t1 = time.time()
            my_sol = my_fct(contract_str, *Tensors)
            t2 = time.time()
            time_np += t1 - t0
            time_my_fct += t2 - t1
            assert np.allclose(np_sol, my_sol)

        print("passed")
        print("-" * 20)
    print("time np:", time_np, " time my_fct:", time_my_fct)


def bmm_using_torch(A, B):
    """ A is of form (b, n, m), B is of form (b, p, m)
        returns C of form (b, n, p) """
    A_tilda = torch.tensor(A)
    B_tilda = torch.tensor(B.transpose(0, 2, 1))
    C = torch.bmm(A_tilda, B_tilda).numpy()
    return C


def bmm_using_torch2(A_tilda, B_tilda):
    """ A is of form (b, n, m), B is of form (b, m, p)
        returns C of form (b, n, p) """
    # A_tilda = torch.tensor(A)
    # B_tilda = torch.tensor(B)
    C = torch.bmm(A_tilda, B_tilda).numpy()
    return C


def test_bmm1(my_fct):
    """test for bmm routine that uses shapes (b,n,m) @ (b,p,m) = (b,n,p)"""
    time_torch = 0
    time_my_imp = 0
    for test in BMM_TESTS:
        A = np.random.rand(*(BMM_TESTS[test][0]))
        B = np.random.rand(*(BMM_TESTS[test][1]))

        t0 = time.time()
        C_tilda = bmm_using_torch(A, B)
        t1 = time.time()
        C = my_fct(A, B)
        t2 = time.time()
        assert np.allclose(C, C_tilda)
        print(test + " passed")
        time_torch += t1 - t0
        time_my_imp += t2 - t1
    print("bmm test passed! time torch:", time_torch, " vs my implementation:", time_my_imp)


def test_bmm2(my_fct):
    """test for bmm routine that uses shapes (b,n,m) @ (b,m,p) = (b,n,p)"""
    time_torch = 0
    time_my_imp = 0
    for test in BMM_TESTS2:
        A = np.random.rand(*(BMM_TESTS2[test][0]))
        B = np.random.rand(*(BMM_TESTS2[test][1]))
        A_tilda = torch.tensor(A)
        B_tilda = torch.tensor(B)

        t0 = time.time()
        C = my_fct(A, B)
        t1 = time.time()
        C_tilda = bmm_using_torch2(A_tilda, B_tilda)
        t2 = time.time()
        assert np.allclose(C, C_tilda)
        print(test + " passed")
        time_torch += t2 - t1
        time_my_imp += t1 - t0
    print("bmm test passed! time torch:", time_torch, " vs my implementation:", time_my_imp)


def test_index_mask(A, B, batch_mask, left_mask, right_mask):
    # used like this in einsum:
    # a = np.ascontiguousarray(single_einsum(str_a, a).reshape(desired_shape_a))
    # b = np.ascontiguousarray(single_einsum(str_b, b).reshape(desired_shape_b))
    # test.test_index_mask(a, a1, offsets_batch_a, offsets_left_a, offsets_contract_a)
    # test.test_index_mask(b, b1, offsets_batch_b, offsets_left_b, offsets_contract_b)
    b, m, n = A.shape
    ptr = B.ctypes.data
    for batch in range(b):
        for i in range(m):
            for j in range(n):
                offset = int((batch_mask[batch] + left_mask[i] + right_mask[j])*B.itemsize)
                value = cast(ptr+offset, POINTER(c_double)).contents.value
                assert np.isclose(A[batch, i, j], value)
    print("Mask Test passed!")
