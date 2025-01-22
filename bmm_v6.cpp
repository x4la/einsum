#include <iostream>

#include "kernel.h"

using namespace std;

#define min(x, y) ((x) < (y) ? (x) : (y))

#ifndef NTHREADS
    #define NTHREADS 6 // TO DO: dynamically
#endif

#define MC (16 * NTHREADS * 8)
#define NC (6 * NTHREADS * 70)
#define KC 792
#define MC_D (8 * NTHREADS * 8)
#define NC_D (6 * NTHREADS * 35)

static float blockA_packed[MC * KC] __attribute__((aligned(64)));
static float blockB_packed[NC * KC] __attribute__((aligned(64)));
static double blockA_packed_d[MC_D * KC] __attribute__((aligned(64)));
static double blockB_packed_d[NC_D * KC] __attribute__((aligned(64)));


void pack_panelB_d(double* B, double* blockB_packed, int nr, int kc, int jj, int pp, const int* offset_right_b, const int* offset_contract_b) {
    for (int p = 0; p < kc; p++) {
        for (int j = 0; j < nr; j++) {
            *blockB_packed++ = B[offset_right_b[jj+j] + offset_contract_b[pp+p]];
        }
        for (int j = nr; j < 6; j++) {
            *blockB_packed++ = 0;
        }
    }
}

void pack_blockB_d(double* B, double* blockB_packed, int nc, int kc, int p, int jj, const int* offset_right_b, const int* offset_contract_b) {
#pragma omp parallel for num_threads(NTHREADS)
    for (int j = 0; j < nc; j += 6) {
        int nr = min(6, nc - j);
        pack_panelB_d(B, &blockB_packed[j * kc], nr, kc, jj+j, p, offset_right_b, offset_contract_b );
    }
}

void pack_panelA_d(double* A, double* blockA_packed, int mr, int kc, int pp, int ii, const int* offset_left_a, const int* offset_contract_a) {
    for (int p = 0; p < kc; p++) {
        for (int i = 0; i < mr; i++) {
            *blockA_packed++ = A[offset_left_a[ii + i]+ offset_contract_a[pp+p]];
        }
        for (int i = mr; i < 8; i++) {
            *blockA_packed++ = 0;
        }
    }
}

void pack_blockA_d(double* A, double* blockA_packed, int mc, int kc, int p, int ii, const int* offset_left_a, const int* offset_contract_a) {
#pragma omp parallel for num_threads(NTHREADS)
    for (int i = 0; i < mc; i += 8) {
        int mr = min(8, mc - i);
        pack_panelA_d(A, &blockA_packed[i * kc], mr, kc, p, ii+i, offset_left_a, offset_contract_a);
    }
}

void bmm_double(double* A, double* B, double* C, const int b, const int m, const int n, const int k, \
const int* offset_batch_a, const int* offset_left_a, const int* offset_contract_a, \
const int* offset_batch_b, const int* offset_contract_b, const int* offset_right_b) {

    // printf("HI from v6");
    const int bsC = m * n;        // batch size for C

    for (int batch = 0; batch < b; batch++) {
        for (int j = 0; j < n; j += NC_D) {
            const int nc = min(NC_D, n - j);
            for (int p = 0; p < k; p += KC) {
                const int kc = min(KC, k - p);
                pack_blockB_d(&B[offset_batch_b[batch]], blockB_packed_d, nc, kc, p, j, offset_right_b, offset_contract_b);
                for (int i = 0; i < m; i += MC_D) {
                    const int mc = min(MC_D, m - i);
                    pack_blockA_d(&A[offset_batch_a[batch]], blockA_packed_d, mc, kc, p, i, offset_left_a, offset_contract_a);
#pragma omp parallel for collapse(2) num_threads(NTHREADS)
                    for (int jr = 0; jr < nc; jr += 6) {
                        for (int ir = 0; ir < mc; ir += 8) {
                            const int nr = min(6, nc - jr);
                            const int mr = min(8, mc - ir);
                            kernel_8x6_double(&blockA_packed_d[ir * kc],
                                        &blockB_packed_d[jr * kc],
                                        &C[batch * bsC + (j + jr) * m + (i + ir)],
                                        mr,
                                        nr,
                                        kc,
                                        m);
                        }
                    }
                }
            }
        }
    }
}



extern "C" {
    void my_bmm(
    void* A,       // Pointer to A with dims (b, m, k)
    void* B,       // Pointer to B with dims (b, k, n)
    void* C,       // Pointer to C with dims (b, m, n)
    int b,
    int m,
    int n,
    int k,
    const int* offset_batch_a, const int* offset_left_a, const int* offset_contract_a,
    const int* offset_batch_b, const int* offset_contract_b, const int* offset_right_b,
    const char* d_type
) {
        if(*d_type == *"d"){
            // Trick: bmm_double expects colum major layout. compute B.T @ A.T
            bmm_double(static_cast<double*>(B), static_cast<double*>(A), static_cast<double*>(C), b, n, m, k, \
            offset_batch_b, offset_right_b, offset_contract_b, offset_batch_a, offset_contract_a, offset_left_a);
        }
        // if(*d_type == *"f") {
        //     // Trick: bmm_float expects colum major layout. compute B.T @ A.T
        //     bmm_float(static_cast<float*>(B), static_cast<float*>(A), static_cast<float*>(C), b, n, m, k);
        // }
    }
}