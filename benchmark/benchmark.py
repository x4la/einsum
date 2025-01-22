import os
import numpy as np
import time

import torch
from fontTools.mtiLib import parseTable
from fontTools.ttLib.tables.ttProgram import instructions


DIMS = [
        [(300, 300, 300), (300, 300, 300)],
        [(400, 600, 200), (400, 200, 100)],
        [(400, 200, 500), (400, 500, 300)],
        [(1, 2000, 2000), (1, 2000, 2000)],
        [(1, 4, 2000), (1, 2000, 4)],
        [(1, 20, 800000), (1, 800000, 20)],
        [(1, 100, 40000), (1, 40000, 100)],
        [(1, 10000, 4), (1, 4, 10000)],
        [(1, 10000, 20), (1, 20, 10000)],
        [(1, 10000, 40), (1, 40, 10000)],
        [(1, 10000, 60), (1, 60, 10000)],
        [(1, 8000, 100), (1, 100, 8000)],
        [(1, 2, 10000000), (1, 10000000, 20)]]


def parse_file():
    dirname = os.path.dirname(__file__)
    PATH = os.path.join(dirname, 'benchmark_cases.txt')
    instructions = []
    dimensionsA = []
    dimensionsB = []
    dimensionsC = []

    # Read file
    with open(PATH, 'r') as f:
        for line in f:
            # Extract the instruction part and the dimensions
            parts = line.strip().split(" ")
            raw_instruction = parts[0]
            raw_dimensions = parts[1]

            a, b, c = raw_instruction.split("-")
            instruction = f"{a},{b}->{c}"
            instructions.append(instruction)

            # Build the dimensions dictionary
            sym_nums = raw_dimensions.split(";")[0:-1]
            dict = {}
            for sym_num in sym_nums:
                sym, num = sym_num.split(":")
                dict[sym] = int(num)

            # Dimensions for each tensor
            dimA = [dict[sym] for sym in a]
            dimB = [dict[sym] for sym in b]
            dimC = [dict[sym] for sym in c]
            dimensionsA.append(dimA)
            dimensionsB.append(dimB)
            dimensionsC.append(dimC)

    return instructions, dimensionsA, dimensionsB, dimensionsC

def create_random_aligned_tensor(dimensions, dtype=np.float64, alignment=64):
    """
    Creates a single aligned NumPy tensor filled with random values.

    Parameters:
    - dimensions (tuple): Shape of the tensor (e.g., (20, 30, 20)).
    - dtype (data-type): Data type of the tensor, default is np.float64.
    - alignment (int): Byte alignment, default is 64 bytes.

    Returns:
    - tensor (np.ndarray): A 64-byte aligned NumPy tensor filled with random values.
    """
    # Calculate the total number of bytes needed for the tensor
    itemsize = np.dtype(dtype).itemsize  # Size of one item in bytes
    nbytes = np.prod(dimensions) * itemsize

    # Ensure there is enough space for alignment
    buffer_size = nbytes + alignment - (nbytes % alignment)

    # Create a raw buffer and align it
    buffer = np.empty(buffer_size, dtype=np.uint8)
    offset = -buffer.ctypes.data % alignment

    # Create the aligned tensor
    tensor = np.ndarray(
        shape=dimensions,
        dtype=dtype,
        buffer=buffer[offset:offset + nbytes],
        order='C'
    )

    # Fill the tensor with random values
    tensor[...] = np.random.rand(*dimensions)

    return tensor


def benchmark_einsum(fcts):
    np.random.seed(0)
    np.set_printoptions(precision=4, suppress=True)
    MAX_INSTR = 96
    instructions, dimsA, dimsB, dimsC = parse_file()
    times = np.zeros((MAX_INSTR+1, len(fcts)))
    for i, instruction in enumerate(instructions):
        if i == MAX_INSTR:
            break
        results = [None]*len(fcts)
        A = np.random.rand(*dimsA[i])
        B = np.random.rand(*dimsB[i])
        print(i)
        for j, fct in enumerate(fcts):
            if fct == torch.einsum:
                A_tilda = torch.from_numpy(A)
                B_tilda = torch.from_numpy(B)
                t0 = time.time()
                results[j] = fct(instruction, A_tilda, B_tilda)
                t1 = time.time()
            else:
                t0 = time.time()
                results[j] = fct(instruction, A, B)
                t1 = time.time()

            times[i,j] = t1 - t0
    times[-1] = np.sum(times, axis=0)
    print(times)
    np.save(os.path.join(os.path.dirname(__file__), "benchmark"), times)



def profile_einsum(fcts):
    np.random.seed(0)
    np.set_printoptions(precision=4, suppress=True)
    TIMES_STAMPS = 6
    MAX_INSTR = 96
    instructions, dimsA, dimsB, dimsC = parse_file()
    times = np.zeros((MAX_INSTR+1, len(fcts) + TIMES_STAMPS))
    for i, instruction in enumerate(instructions):
        if i == MAX_INSTR:
            break
        results = [None]*len(fcts)
        # A = np.random.rand(*dimsA[i])
        # B = np.random.rand(*dimsB[i])
        # A = create_random_aligned_tensor(dimsA[i])
        # B = create_random_aligned_tensor(dimsB[i])
        # print(A.ctypes.data%64, B.ctypes.data%64)
        A = np.arange(np.prod(dimsA[i])).reshape(dimsA[i]).astype(np.float64)
        B = np.arange(np.prod(dimsB[i])).reshape(dimsB[i]).astype(np.float64)
        # print(B)
        print(i, A.ctypes.data%64, B.ctypes.data%64)
        for j, fct in enumerate(fcts):
            if fct == torch.einsum:
                A_tilda = torch.from_numpy(A)
                B_tilda = torch.from_numpy(B)
                t0 = time.time()
                results[j] = fct(instruction, A_tilda, B_tilda)
                t1 = time.time()
                # print(results[j])
            else:
                t0 = time.time()
                res = fct(instruction, A, B)
                t1 = time.time()
                times[i, :TIMES_STAMPS] = res[1]
                results[j] = res[0]
                # np.set_printoptions(precision=4, suppress=True)
                # print(res[0])

            times[i,j+TIMES_STAMPS] = t1 - t0
        # print(times)
        # print(results)
        # for i in range(len(fcts)-1):
        #     assert np.allclose(results[i], results[i+1])
        #     print("fct ", i, "and ", i+1, "passed")

    times[-1] = np.sum(times, axis=0)
    print(times)
    np.save(os.path.join(os.path.dirname(__file__), "benchmark"), times)


def benchmark_bmm(fcts):
    times = np.zeros((len(DIMS), len(fcts)))
    for i, case in enumerate(DIMS):
        results = [None]*len(fcts)
        A = np.random.rand(*case[0])
        B = np.random.rand(*case[1])
        B = np.ascontiguousarray(B.transpose(0, 2, 1))
        print(i)
        for j, fct in enumerate(fcts):
            if j == 1:
                B = np.ascontiguousarray(B.transpose(0, 2, 1))
            # assumes torch.bmm is the last function!
            if fct == torch.bmm:
                A = torch.from_numpy(A)
                B = torch.from_numpy(B)
            t0 = time.time()
            results[j] = fct(A, B)
            t1 = time.time()
            times[i,j] = t1 - t0
        for i in range(len(fcts)-1):
            assert np.allclose(results[i], results[i+1])
            print("fct ", i, "and ", i+1, "passed")

    np.set_printoptions(suppress=True, precision=4)
    np.save(os.path.join(os.path.dirname(__file__), "bmm"), times)

