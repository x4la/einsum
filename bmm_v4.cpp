#include <iostream>

#include "kernel.h"

using namespace std;

#define min(x, y) ((x) < (y) ? (x) : (y))

#ifndef NTHREADS
    #define NTHREADS 6
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

void pack_panelB(float* B, float* blockB_packed, int nr, int kc, int k) {
    for (int p = 0; p < kc; p++) {
        for (int j = 0; j < nr; j++) {
            *blockB_packed++ = B[j * k + p];
        }
        for (int j = nr; j < 6; j++) {
            *blockB_packed++ = 0;
        }
    }
}

void pack_blockB(float* B, float* blockB_packed, int nc, int kc, int k) {
#pragma omp parallel for num_threads(NTHREADS)
    for (int j = 0; j < nc; j += 6) {
        int nr = min(6, nc - j);
        pack_panelB(&B[j * k], &blockB_packed[j * kc], nr, kc, k);
    }
}

void pack_panelA(float* A, float* blockA_packed, int mr, int kc, int M) {
    for (int p = 0; p < kc; p++) {
        for (int i = 0; i < mr; i++) {
            *blockA_packed++ = A[p * M + i];
        }
        for (int i = mr; i < 16; i++) {
            *blockA_packed++ = 0;
        }
    }
}

void pack_blockA(float* A, float* blockA_packed, int mc, int kc, int M) {
#pragma omp parallel for num_threads(NTHREADS)
    for (int i = 0; i < mc; i += 16) {
        int mr = min(16, mc - i);
        pack_panelA(&A[i], &blockA_packed[i * kc], mr, kc, M);
    }
}

void bmm_float(float* A, float* B, float* C, int b, int m, int n, int k) {
    int bsA =  m * k;       // batch size for A
    int bsB = k * n;        // batch size for B
    int bsC = m * n;        // batch size for C

    for (int batch = 0; batch < b; batch++) {
        for (int j = 0; j < n; j += NC) {
            int nc = min(NC, n - j);
            for (int p = 0; p < k; p += KC) {
                int kc = min(KC, k - p);
                pack_blockB(&B[batch * bsB + j * k + p], blockB_packed, nc, kc, k);
                for (int i = 0; i < m; i += MC) {
                    int mc = min(MC, m - i);
                    pack_blockA(&A[batch * bsA + p * m + i], blockA_packed, mc, kc, m);
#pragma omp parallel for collapse(2) num_threads(NTHREADS)
                    for (int jr = 0; jr < nc; jr += 6) {
                        for (int ir = 0; ir < mc; ir += 16) {
                            int nr = min(6, nc - jr);
                            int mr = min(16, mc - ir);
                            kernel_16x6_float(&blockA_packed[ir * kc],
                                        &blockB_packed[jr * kc],
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

void pack_panelB_d(double* B, double* blockB_packed, int nr, int kc, int k) {
    for (int p = 0; p < kc; p++) {
        for (int j = 0; j < nr; j++) {
            *blockB_packed++ = B[j * k + p];
        }
        for (int j = nr; j < 6; j++) {
            *blockB_packed++ = 0;
        }
    }
}

void pack_blockB_d(double* B, double* blockB_packed, int nc, int kc, int k) {
#pragma omp parallel for num_threads(NTHREADS)
    for (int j = 0; j < nc; j += 6) {
        int nr = min(6, nc - j);
        pack_panelB_d(&B[j * k], &blockB_packed[j * kc], nr, kc, k);
    }
}

void pack_panelA_d(double* A, double* blockA_packed, int mr, int kc, int M) {
    for (int p = 0; p < kc; p++) {
        for (int i = 0; i < mr; i++) {
            *blockA_packed++ = A[p * M + i];
        }
        for (int i = mr; i < 8; i++) {
            *blockA_packed++ = 0;
        }
    }
}

void pack_blockA_d(double* A, double* blockA_packed, int mc, int kc, int M) {
#pragma omp parallel for num_threads(NTHREADS)
    for (int i = 0; i < mc; i += 8) {
        int mr = min(8, mc - i);
        pack_panelA_d(&A[i], &blockA_packed[i * kc], mr, kc, M);
    }
}

void bmm_double(double* A, double* B, double* C, int b, int m, int n, int k) {
    int bsA =  m * k;       // batch size for A
    int bsB = k * n;        // batch size for B
    int bsC = m * n;        // batch size for C

    for (int batch = 0; batch < b; batch++) {
        for (int j = 0; j < n; j += NC_D) {
            const int nc = min(NC_D, n - j);
            for (int p = 0; p < k; p += KC) {
                const int kc = min(KC, k - p);
                pack_blockB_d(&B[batch * bsB + j * k + p], blockB_packed_d, nc, kc, k);
                for (int i = 0; i < m; i += MC_D) {
                    const int mc = min(MC_D, m - i);
                    pack_blockA_d(&A[batch * bsA + p * m + i], blockA_packed_d, mc, kc, m);
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
    const char* d_type
) {
        if(*d_type == *"d"){
            // Trick: bmm_double expects colum major layout. compute B.T @ A.T
            bmm_double(static_cast<double*>(B), static_cast<double*>(A), static_cast<double*>(C), b, n, m, k);
        }
        if(*d_type == *"f") {
            // Trick: bmm_float expects colum major layout. compute B.T @ A.T
            bmm_float(static_cast<float*>(B), static_cast<float*>(A), static_cast<float*>(C), b, n, m, k);
        }
    }
}