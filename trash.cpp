#include "kernel.h"

void kernel_16x6(float* blockA_packed,
                 float* blockB_packed,
                 float* C,
                 int mr,
                 int nr,
                 int kc,
                 int m) {
    __m256 C00 = _mm256_setzero_ps();
    __m256 C10 = _mm256_setzero_ps();
    __m256 C01 = _mm256_setzero_ps();
    __m256 C11 = _mm256_setzero_ps();
    __m256 C02 = _mm256_setzero_ps();
    __m256 C12 = _mm256_setzero_ps();
    __m256 C03 = _mm256_setzero_ps();
    __m256 C13 = _mm256_setzero_ps();
    __m256 C04 = _mm256_setzero_ps();
    __m256 C14 = _mm256_setzero_ps();
    __m256 C05 = _mm256_setzero_ps();
    __m256 C15 = _mm256_setzero_ps();

    __m256 b_packFloat8;
    __m256 a0_packFloat8;
    __m256 a1_packFloat8;
    __m256i packed_mask0;
    __m256i packed_mask1;
//    load current C
    if (mr != 16) {
        packed_mask0 = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask[16 - mr]));
        packed_mask1 = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask[16 - mr + 8]));
        switch (nr) {
        case 1 :
            C00 = _mm256_maskload_ps(C, packed_mask0);
            C10 = _mm256_maskload_ps(&C[8], packed_mask1);
            break;
        case 2 :
            C00 = _mm256_maskload_ps(C, packed_mask0);
            C10 = _mm256_maskload_ps(&C[8], packed_mask1);
            C01 = _mm256_maskload_ps(&C[m], packed_mask0);
            C11 = _mm256_maskload_ps(&C[m + 8], packed_mask1);
            break;
        case 3 :
            C00 = _mm256_maskload_ps(C, packed_mask0);
            C10 = _mm256_maskload_ps(&C[8], packed_mask1);
            C01 = _mm256_maskload_ps(&C[m], packed_mask0);
            C11 = _mm256_maskload_ps(&C[m + 8], packed_mask1);
            C02 = _mm256_maskload_ps(&C[2 * m], packed_mask0);
            C12 = _mm256_maskload_ps(&C[2 * m + 8], packed_mask1);
            break;
        case 4 :
            C00 = _mm256_maskload_ps(C, packed_mask0);
            C10 = _mm256_maskload_ps(&C[8], packed_mask1);
            C01 = _mm256_maskload_ps(&C[m], packed_mask0);
            C11 = _mm256_maskload_ps(&C[m + 8], packed_mask1);
            C02 = _mm256_maskload_ps(&C[2 * m], packed_mask0);
            C12 = _mm256_maskload_ps(&C[2 * m + 8], packed_mask1);
            C03 = _mm256_maskload_ps(&C[3 * m], packed_mask0);
            C13 = _mm256_maskload_ps(&C[3 * m + 8], packed_mask1);
            break;
        case 5 :
            C00 = _mm256_maskload_ps(C, packed_mask0);
            C10 = _mm256_maskload_ps(&C[8], packed_mask1);
            C01 = _mm256_maskload_ps(&C[m], packed_mask0);
            C11 = _mm256_maskload_ps(&C[m + 8], packed_mask1);
            C02 = _mm256_maskload_ps(&C[2 * m], packed_mask0);
            C12 = _mm256_maskload_ps(&C[2 * m + 8], packed_mask1);
            C03 = _mm256_maskload_ps(&C[3 * m], packed_mask0);
            C13 = _mm256_maskload_ps(&C[3 * m + 8], packed_mask1);
            C04 = _mm256_maskload_ps(&C[4 * m], packed_mask0);
            C14 = _mm256_maskload_ps(&C[4 * m + 8], packed_mask1);
            break;
        case 6 :
            C00 = _mm256_maskload_ps(C, packed_mask0);
            C10 = _mm256_maskload_ps(&C[8], packed_mask1);
            C01 = _mm256_maskload_ps(&C[m], packed_mask0);
            C11 = _mm256_maskload_ps(&C[m + 8], packed_mask1);
            C02 = _mm256_maskload_ps(&C[2 * m], packed_mask0);
            C12 = _mm256_maskload_ps(&C[2 * m + 8], packed_mask1);
            C03 = _mm256_maskload_ps(&C[3 * m], packed_mask0);
            C13 = _mm256_maskload_ps(&C[3 * m + 8], packed_mask1);
            C04 = _mm256_maskload_ps(&C[4 * m], packed_mask0);
            C14 = _mm256_maskload_ps(&C[4 * m + 8], packed_mask1);
            C05 = _mm256_maskload_ps(&C[5 * m], packed_mask0);
            C15 = _mm256_maskload_ps(&C[5 * m + 8], packed_mask1);
            break;
        }
    } else {
        switch (nr) {
        case 1 :
            C00 = _mm256_loadu_ps(C);
            C10 = _mm256_loadu_ps(&C[8]);
            break;
        case 2 :
            C00 = _mm256_loadu_ps(C);
            C10 = _mm256_loadu_ps(&C[8]);
            C01 = _mm256_loadu_ps(&C[m]);
            C11 = _mm256_loadu_ps(&C[m + 8]);
            break;
        case 3 :
            C00 = _mm256_loadu_ps(C);
            C10 = _mm256_loadu_ps(&C[8]);
            C01 = _mm256_loadu_ps(&C[m]);
            C11 = _mm256_loadu_ps(&C[m + 8]);
            C02 = _mm256_loadu_ps(&C[2 * m]);
            C12 = _mm256_loadu_ps(&C[2 * m + 8]);
            break;
        case 4 :
            C00 = _mm256_loadu_ps(C);
            C10 = _mm256_loadu_ps(&C[8]);
            C01 = _mm256_loadu_ps(&C[m]);
            C11 = _mm256_loadu_ps(&C[m + 8]);
            C02 = _mm256_loadu_ps(&C[2 * m]);
            C12 = _mm256_loadu_ps(&C[2 * m + 8]);
            C03 = _mm256_loadu_ps(&C[3 * m]);
            C13 = _mm256_loadu_ps(&C[3 * m + 8]);
            break;
        case 5 :
            C00 = _mm256_loadu_ps(C);
            C10 = _mm256_loadu_ps(&C[8]);
            C01 = _mm256_loadu_ps(&C[m]);
            C11 = _mm256_loadu_ps(&C[m + 8]);
            C02 = _mm256_loadu_ps(&C[2 * m]);
            C12 = _mm256_loadu_ps(&C[2 * m + 8]);
            C03 = _mm256_loadu_ps(&C[3 * m]);
            C13 = _mm256_loadu_ps(&C[3 * m + 8]);
            C04 = _mm256_loadu_ps(&C[4 * m]);
            C14 = _mm256_loadu_ps(&C[4 * m + 8]);
            break;
        case 6 :
            C00 = _mm256_loadu_ps(C);
            C10 = _mm256_loadu_ps(&C[8]);
            C01 = _mm256_loadu_ps(&C[m]);
            C11 = _mm256_loadu_ps(&C[m + 8]);
            C02 = _mm256_loadu_ps(&C[2 * m]);
            C12 = _mm256_loadu_ps(&C[2 * m + 8]);
            C03 = _mm256_loadu_ps(&C[3 * m]);
            C13 = _mm256_loadu_ps(&C[3 * m + 8]);
            C04 = _mm256_loadu_ps(&C[4 * m]);
            C14 = _mm256_loadu_ps(&C[4 * m + 8]);
            C05 = _mm256_loadu_ps(&C[5 * m]);
            C15 = _mm256_loadu_ps(&C[5 * m + 8]);
            break;
        }
    }
//    do the calculation
    for (int p = 0; p < kc; p++) {
        a0_packFloat8 = _mm256_loadu_ps(blockA_packed);
        a1_packFloat8 = _mm256_loadu_ps(blockA_packed + 8);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed);
        C00 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C00);
        C10 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C10);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 1);
        C01 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C01);
        C11 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C11);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 2);
        C02 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C02);
        C12 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C12);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 3);
        C03 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C03);
        C13 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C13);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 4);
        C04 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C04);
        C14 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C14);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 5);
        C05 = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C05);
        C15 = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C15);

        blockA_packed += 16;
        blockB_packed += 6;
    }
//    now do the stores!
    if (mr != 16) {
        switch (nr) {
        case 1 :
            _mm256_maskstore_ps(C, packed_mask0, C00);
            _mm256_maskstore_ps(&C[8], packed_mask1, C10);
            break;
        case 2 :
            _mm256_maskstore_ps(C, packed_mask0, C00);
            _mm256_maskstore_ps(&C[8], packed_mask1, C10);
            _mm256_maskstore_ps(&C[m], packed_mask0, C01);
            _mm256_maskstore_ps(&C[m + 8], packed_mask1, C11);
            break;
        case 3 :
            _mm256_maskstore_ps(C, packed_mask0, C00);
            _mm256_maskstore_ps(&C[8], packed_mask1, C10);
            _mm256_maskstore_ps(&C[m], packed_mask0, C01);
            _mm256_maskstore_ps(&C[m + 8], packed_mask1, C11);
            _mm256_maskstore_ps(&C[2 * m], packed_mask0, C02);
            _mm256_maskstore_ps(&C[2 * m + 8], packed_mask1, C12);
            break;
        case 4 :
            _mm256_maskstore_ps(C, packed_mask0, C00);
            _mm256_maskstore_ps(&C[8], packed_mask1, C10);
            _mm256_maskstore_ps(&C[m], packed_mask0, C01);
            _mm256_maskstore_ps(&C[m + 8], packed_mask1, C11);
            _mm256_maskstore_ps(&C[2 * m], packed_mask0, C02);
            _mm256_maskstore_ps(&C[2 * m + 8], packed_mask1, C12);
            _mm256_maskstore_ps(&C[3 * m], packed_mask0, C03);
            _mm256_maskstore_ps(&C[3 * m + 8], packed_mask1, C13);
            break;
        case 5 :
            _mm256_maskstore_ps(C, packed_mask0, C00);
            _mm256_maskstore_ps(&C[8], packed_mask1, C10);
            _mm256_maskstore_ps(&C[m], packed_mask0, C01);
            _mm256_maskstore_ps(&C[m + 8], packed_mask1, C11);
            _mm256_maskstore_ps(&C[2 * m], packed_mask0, C02);
            _mm256_maskstore_ps(&C[2 * m + 8], packed_mask1, C12);
            _mm256_maskstore_ps(&C[3 * m], packed_mask0, C03);
            _mm256_maskstore_ps(&C[3 * m + 8], packed_mask1, C13);
            _mm256_maskstore_ps(&C[4 * m], packed_mask0, C04);
            _mm256_maskstore_ps(&C[4 * m + 8], packed_mask1, C14);
            break;
        case 6 :
            _mm256_maskstore_ps(C, packed_mask0, C00);
            _mm256_maskstore_ps(&C[8], packed_mask1, C10);
            _mm256_maskstore_ps(&C[m], packed_mask0, C01);
            _mm256_maskstore_ps(&C[m + 8], packed_mask1, C11);
            _mm256_maskstore_ps(&C[2 * m], packed_mask0, C02);
            _mm256_maskstore_ps(&C[2 * m + 8], packed_mask1, C12);
            _mm256_maskstore_ps(&C[3 * m], packed_mask0, C03);
            _mm256_maskstore_ps(&C[3 * m + 8], packed_mask1, C13);
            _mm256_maskstore_ps(&C[4 * m], packed_mask0, C04);
            _mm256_maskstore_ps(&C[4 * m + 8], packed_mask1, C14);
            _mm256_maskstore_ps(&C[5 * m], packed_mask0, C05);
            _mm256_maskstore_ps(&C[5 * m + 8], packed_mask1, C15);
            break;
        }
    } else {
        switch (nr) {
        case 1 :
            _mm256_storeu_ps(C, C00);
            _mm256_storeu_ps(&C[8], C10);
            break;
        case 2 :
            _mm256_storeu_ps(C, C00);
            _mm256_storeu_ps(&C[8], C10);
            _mm256_storeu_ps(&C[m], C01);
            _mm256_storeu_ps(&C[m + 8], C11);
            break;
        case 3 :
            _mm256_storeu_ps(C, C00);
            _mm256_storeu_ps(&C[8], C10);
            _mm256_storeu_ps(&C[m], C01);
            _mm256_storeu_ps(&C[m + 8], C11);
            _mm256_storeu_ps(&C[2 * m], C02);
            _mm256_storeu_ps(&C[2 * m + 8], C12);
            break;
        case 4 :
            _mm256_storeu_ps(C, C00);
            _mm256_storeu_ps(&C[8], C10);
            _mm256_storeu_ps(&C[m], C01);
            _mm256_storeu_ps(&C[m + 8], C11);
            _mm256_storeu_ps(&C[2 * m], C02);
            _mm256_storeu_ps(&C[2 * m + 8], C12);
            _mm256_storeu_ps(&C[3 * m], C03);
            _mm256_storeu_ps(&C[3 * m + 8], C13);
            break;
        case 5 :
            _mm256_storeu_ps(C, C00);
            _mm256_storeu_ps(&C[8], C10);
            _mm256_storeu_ps(&C[m], C01);
            _mm256_storeu_ps(&C[m + 8], C11);
            _mm256_storeu_ps(&C[2 * m], C02);
            _mm256_storeu_ps(&C[2 * m + 8], C12);
            _mm256_storeu_ps(&C[3 * m], C03);
            _mm256_storeu_ps(&C[3 * m + 8], C13);
            _mm256_storeu_ps(&C[4 * m], C04);
            _mm256_storeu_ps(&C[4 * m + 8], C14);
            break;
        case 6 :
            _mm256_storeu_ps(C, C00);
            _mm256_storeu_ps(&C[8], C10);
            _mm256_storeu_ps(&C[m], C01);
            _mm256_storeu_ps(&C[m + 8], C11);
            _mm256_storeu_ps(&C[2 * m], C02);
            _mm256_storeu_ps(&C[2 * m + 8], C12);
            _mm256_storeu_ps(&C[3 * m], C03);
            _mm256_storeu_ps(&C[3 * m + 8], C13);
            _mm256_storeu_ps(&C[4 * m], C04);
            _mm256_storeu_ps(&C[4 * m + 8], C14);
            _mm256_storeu_ps(&C[5 * m], C05);
            _mm256_storeu_ps(&C[5 * m + 8], C15);
            break;
        }
    }
}

//
//#include "kernel.h"
//
//#include <cstdio>
//
//void kernel_6x8( double* blockA_packed, // (mr x kc)
//                 double* blockB_packed, // (kc x nr)
//                 double* C,             // (nr x mr)
//                 int mr,    // 6
//                 int nr,    // 8
//                 int kc,
//                 int m) {
//    // printf("values: {%d}, {%d}, {%d}, {%d},\n", mr, nr, kc, m);
//    __m256d C00 = _mm256_setzero_pd();
//    __m256d C01 = _mm256_setzero_pd();
//    __m256d C10 = _mm256_setzero_pd();
//    __m256d C11 = _mm256_setzero_pd();
//    __m256d C20 = _mm256_setzero_pd();
//    __m256d C21 = _mm256_setzero_pd();
//    __m256d C30 = _mm256_setzero_pd();
//    __m256d C31 = _mm256_setzero_pd();
//    __m256d C40 = _mm256_setzero_pd();
//    __m256d C41 = _mm256_setzero_pd();
//    __m256d C50 = _mm256_setzero_pd();
//    __m256d C51 = _mm256_setzero_pd();
//
//    __m256d a_packDouble4;
//    __m256d b0_packDouble4;
//    __m256d b1_packDouble4;
//    __m256i packed_mask0;
//    __m256i packed_mask1;
//
//    C00 = _mm256_loadu_pd(C);
//    C01 = _mm256_loadu_pd(&C[4]);
//    C10 = _mm256_loadu_pd(&C[m]);
//    C11 = _mm256_loadu_pd(&C[m + 4]);
//    C20 = _mm256_loadu_pd(&C[2 * m]);
//    C21 = _mm256_loadu_pd(&C[2 * m + 4]);
//    C30 = _mm256_loadu_pd(&C[3 * m]);
//    C31 = _mm256_loadu_pd(&C[3 * m + 4]);
//    C40 = _mm256_loadu_pd(&C[4 * m]);
//    C41 = _mm256_loadu_pd(&C[4 * m + 4]);
//    C50 = _mm256_loadu_pd(&C[5 * m]);
//    C51 = _mm256_loadu_pd(&C[5 * m + 4]);
//
//    for (int k = 0; k < kc; k++) {
//        b0_packDouble4 = _mm256_loadu_pd(blockB_packed);
//        b1_packDouble4 = _mm256_loadu_pd(blockB_packed + 4);
//
//        a_packDouble4 = _mm256_broadcast_sd(blockA_packed);
//        C00 = _mm256_fmadd_pd(a_packDouble4, b0_packDouble4, C00);
//        C01 = _mm256_fmadd_pd(a_packDouble4, b1_packDouble4, C01);
//
//        a_packDouble4 = _mm256_broadcast_sd(blockA_packed + 1*kc);
//        C10 = _mm256_fmadd_pd(a_packDouble4, b0_packDouble4, C10);
//        C11 = _mm256_fmadd_pd(a_packDouble4, b1_packDouble4, C11);
//
//        a_packDouble4 = _mm256_broadcast_sd(blockA_packed + 2*kc);
//        C20 = _mm256_fmadd_pd(a_packDouble4, b0_packDouble4, C20);
//        C21 = _mm256_fmadd_pd(a_packDouble4, b1_packDouble4, C21);
//
//        a_packDouble4 = _mm256_broadcast_sd(blockA_packed + 3*kc);
//        C30 = _mm256_fmadd_pd(a_packDouble4, b0_packDouble4, C30);
//        C31 = _mm256_fmadd_pd(a_packDouble4, b1_packDouble4, C31);
//
//        a_packDouble4 = _mm256_broadcast_sd(blockA_packed + 4*kc);
//        C40 = _mm256_fmadd_pd(a_packDouble4, b0_packDouble4, C40);
//        C41 = _mm256_fmadd_pd(a_packDouble4, b1_packDouble4, C41);
//
//        a_packDouble4 = _mm256_broadcast_sd(blockA_packed + 5*kc);
//        C50 = _mm256_fmadd_pd(a_packDouble4, b0_packDouble4, C50);
//        C51 = _mm256_fmadd_pd(a_packDouble4, b1_packDouble4, C51);
//
//        blockA_packed += 1;
//        blockB_packed += 8;
//    }
//// now do the stores
//    _mm256_storeu_pd(C, C00);
//    _mm256_storeu_pd(&C[4], C01);
//    _mm256_storeu_pd(&C[m], C10);
//    _mm256_storeu_pd(&C[m + 4], C11);
//    _mm256_storeu_pd(&C[2 * m], C20);
//    _mm256_storeu_pd(&C[2 * m + 4], C21);
//    _mm256_storeu_pd(&C[3 * m], C30);
//    _mm256_storeu_pd(&C[3 * m + 4], C31);
//    _mm256_storeu_pd(&C[4 * m], C40);
//    _mm256_storeu_pd(&C[4 * m + 4], C41);
//    _mm256_storeu_pd(&C[5 * m], C50);
//    _mm256_storeu_pd(&C[5 * m + 4], C51);
//}



// Main function to calculate offsets
void calc_offsets_dim(const int* dims, const int* strides, const int* sizes_block, const int n, const int size, int* offsets) {
    if (n == 0) {
        offsets[0] = 0;
        return;
    }
    fill(dims, strides, sizes_block, offsets, 0, 0, 0, n);
}


int prod(const int* vec, int size) {
    int result = 1;
    for (int i = 0; i < size; ++i) {
        result *= vec[i];
    }
    return result;
}

// Helper function to calculate sizes_block based on dimensions
void calc_sizes_block(const int* dims, int* sizes_block, int size) {
    sizes_block[size - 1] = 1;
    for (int i = size - 2; i >= 0; --i) {
        sizes_block[i] = sizes_block[i + 1] * dims[i + 1];
    }
}
