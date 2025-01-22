#include <iostream>
#include <omp.h>

using namespace std;

#define min(x, y) ((x) < (y) ? (x) : (y))

void bmm_double(
    void* A,       // Pointer to A with dims (b, n, m)
    void* B,       // Pointer to B with dims (b, m, p)
    void* C,       // Pointer to C with dims (b, n, p)
    int b,
    int n,
    int m,
    int p
) {
    auto castA = (double*) A;
    auto castB = (double*) B;
    auto castC = (double*) C;
    int bsA =  n * m;       // batch size for A
    int bsB = m * p;        // batch size for B
    int bsC = n * p;        // batch size for C
    int BS = 64;            // block size

#pragma omp parallel for collapse(2)
    for (int batch = 0; batch < b; ++batch) {
        for (int ii = 0; ii < n; ii+=BS){
            for (int kk = 0; kk < m; kk+=BS){
                for (int jj = 0; jj < p; jj+=BS){
                    int i_end = min(ii + BS, n);
                    int k_end = min(kk + BS, m);
                    int j_end = min(jj + BS, p);
                    for (int i = ii; i < i_end; ++i) {
                        for (int k = kk; k < k_end; ++k) {
                            for (int j = jj; j < j_end; ++j) {
                                castC[batch * bsC + p*i + j] += castA[batch * bsA + m*i + k] * castB[batch * bsB +p*k + j];
                            }
                        }
                    }
                }
            }
        }
    }
}


void bmm_float(
    void* A,       // Pointer to A with dims (b, n, m)
    void* B,       // Pointer to B with dims (b, p, m)
    void* C,       // Pointer to C with dims (b, n, p)
    int b,
    int n,
    int m,
    int p
    ) {
    auto castA = (float*) A;
    auto castB = (float*) B;
    auto castC = (float*) C;
    int bsA =  n * m;       // batch size for A
    int bsB = m * p;        // batch size for B
    int bsC = n * p;        // batch size for C

    int BS = 64;
#pragma omp parallel for collapse(2)
    for (int batch = 0; batch < b; ++batch) {
        for (int ii = 0; ii < n; ii+=BS){
            for (int kk = 0; kk < m; kk+=BS){
                for (int jj = 0; jj < p; jj+=BS){
                    int i_end = min(ii + BS, n);
                    int k_end = min(kk + BS, m);
                    int j_end = min(jj + BS, p);
                    for (int i = ii; i < i_end; ++i) {
                        for (int k = kk; k < k_end; ++k) {
                            for (int j = jj; j < j_end; ++j) {
                                castC[batch * bsC + p*i + j] += castA[batch * bsA + m*i + k] * castB[batch * bsB +p*k + j];
                            }
                        }
                    }
                }
            }
        }
    }
}

extern "C" {
    void my_bmm(
    void* A,       // Pointer to A with dims (b, n, m)
    void* B,       // Pointer to B with dims (b, p, m)
    void* C,       // Pointer to C with dims (b, n, p)
    int b,
    int n,
    int m,
    int p,
    const char* d_type
) {
        if(*d_type == *"d"){
            bmm_double(A, B, C, b, n, m, p);
        }
        if(*d_type == *"f") {
            bmm_float(A, B, C, b, n, m, p);
        }
    }

}