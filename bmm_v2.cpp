#include <iostream>
#include <omp.h>

using namespace std;


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
#pragma omp parallel for collapse(2) // distributes work (to 6 cores on my machine)
    for (int batch = 0; batch < b; ++batch) {
        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < m; ++k) {
                for (int j = 0; j < p; ++j) {
                    castC[batch * bsC + i * p + j] += castA[batch * bsA + i * m + k] * castB[batch * bsB + k * p + j];
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
    int bsA =  n * m;
    int bsB = m * p;
    int bsC = n * p;
#pragma omp parallel for collapse(2) // distributes work (to 6 cores on my machine)
    for (int batch = 0; batch < b; ++batch) {
        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < m; ++k) {
                for (int j = 0; j < p; ++j) {
                    castC[batch * bsC + i * p + j] += castA[batch * bsA + i * m + k] * castB[batch * bsB + k * p + j];
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