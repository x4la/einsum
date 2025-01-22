#pragma once
#include <immintrin.h>
#include <cstdint>

static int8_t mask[32]
    __attribute__((aligned(64))) = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0};
static int8_t mask_double[16]
    __attribute__((aligned(64))) = {-1, -1, -1, -1, -1, -1, -1, -1,
                                    0,  0,  0,  0,  0,  0,  0,  0};

void kernel_16x6_float(float* blockA_packed,
                 float* blockB_packed,
                 float* C,
                 int mr,
                 int nr,
                 int kc,
                 int m);

void kernel_8x6_double(double* blockA_packed,
                 double* blockB_packed,
                 double* C,
                 int mr,
                 int nr,
                 int kc,
                 int m);
