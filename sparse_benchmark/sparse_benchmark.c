#define _POSIX_C_SOURCE 199309L 
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>   // for AVX2 / FMA intrinsics

#define TOL 1e-7
#define REPEAT 100

// ==============================
// Timing utility
// ==============================
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ==============================
// Naive dense multiplication (original)
// ==============================
void dense_multiply(int rows, int cols, const double* A, const double* x, double* y) {
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < cols; j++) {
            sum += A[i * cols + j] * x[j];
        }
        y[i] = sum;
    }
}

// ==============================
// AVX2 + FMA optimized dense multiplication
// Requires: -mavx2 -mfma (or -march=native)
// ==============================
void dense_multiply_avx2(int rows, int cols, const double* A, const double* x, double* y) {
    for (int i = 0; i < rows; i++) {
        __m256d sum0 = _mm256_setzero_pd();
        __m256d sum1 = _mm256_setzero_pd();
        int j = 0;

        // 8-way unrolled: process 8 doubles per iteration (2 AVX registers)
        for (; j + 7 < cols; j += 8) {
            // Prefetch next cache line
            _mm_prefetch((const char*)&A[i * cols + j + 16], _MM_HINT_T0);

            // Load 8 consecutive elements of A
            __m256d a0 = _mm256_loadu_pd(&A[i * cols + j]);
            __m256d a1 = _mm256_loadu_pd(&A[i * cols + j + 4]);

            // Gather x values (non-contiguous, use setr)
            __m256d x0 = _mm256_setr_pd(x[j], x[j+1], x[j+2], x[j+3]);
            __m256d x1 = _mm256_setr_pd(x[j+4], x[j+5], x[j+6], x[j+7]);

            // Fused multiply-add
            sum0 = _mm256_fmadd_pd(a0, x0, sum0);
            sum1 = _mm256_fmadd_pd(a1, x1, sum1);
        }

        // Combine the two accumulators
        sum0 = _mm256_add_pd(sum0, sum1);
        double partial[4];
        _mm256_storeu_pd(partial, sum0);
        double dot = partial[0] + partial[1] + partial[2] + partial[3];

        // Remainder
        for (; j < cols; j++) {
            dot += A[i * cols + j] * x[j];
        }

        y[i] = dot;
    }
}

// ==============================
// CSR conversion (unchanged)
// ==============================
void dense_to_csr(
    int rows, int cols, const double* A,
    double* values, int* col_idx, int* row_ptr, int* nnz
) {
    int count = 0;
    row_ptr[0] = 0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double val = A[i * cols + j];
            if (val != 0.0) {
                values[count] = val;
                col_idx[count] = j;
                count++;
            }
        }
        row_ptr[i + 1] = count;
    }

    *nnz = count;
}

// ==============================
// CSR baseline
// ==============================
void csr_multiply(
    int rows,
    const double* values,
    const int* col_idx,
    const int* row_ptr,
    const double* x,
    double* y
) {
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; k++) {
            sum += values[k] * x[col_idx[k]];
        }
        y[i] = sum;
    }
}

// ==============================
// CSR with 2-way unrolling (kept for completeness)
// ==============================
void csr_multiply_optimized(
    int rows,
    const double* values,
    const int* col_idx,
    const int* row_ptr,
    const double* x,
    double* y
) {
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        int start = row_ptr[i];
        int end = row_ptr[i + 1];

        int k = start;
        for (; k + 1 < end; k += 2) {
            sum += values[k] * x[col_idx[k]];
            sum += values[k + 1] * x[col_idx[k + 1]];
        }
        for (; k < end; k++) {
            sum += values[k] * x[col_idx[k]];
        }
        y[i] = sum;
    }
}

// ==============================
// Sparse matrix generation (unchanged)
// ==============================
void generate_matrix(double* A, int rows, int cols, double density) {
    for (int i = 0; i < rows * cols; i++) {
        if ((double)rand() / RAND_MAX < density) {
            A[i] = ((double)rand() / RAND_MAX) * 10.0;
        } else {
            A[i] = 0.0;
        }
    }
}

// ==============================
// Robust relative error check
// ==============================
int check(double* a, double* b, int n) {
    for (int i = 0; i < n; i++) {
        double diff = fabs(a[i] - b[i]);
        double rel_tol = TOL * fmax(1.0, fabs(b[i]));   // scaled tolerance
        if (diff > rel_tol) return 0;
    }
    return 1;
}

// ==============================
// MAIN
// ==============================
int main() {
    srand(42);

    int rows = 1000;
    int cols = 1000;

    // Allocate matrices and vectors
    double* A = malloc(rows * cols * sizeof(double));
    double* x = malloc(cols * sizeof(double));

    double* y_dense    = malloc(rows * sizeof(double));
    double* y_avx2     = malloc(rows * sizeof(double));
    double* y_csr      = malloc(rows * sizeof(double));
    double* y_opt      = malloc(rows * sizeof(double));

    double* values    = malloc(rows * cols * sizeof(double));
    int*    col_idx   = malloc(rows * cols * sizeof(int));
    int*    row_ptr   = malloc((rows + 1) * sizeof(int));

    for (int i = 0; i < cols; i++) {
        x[i] = ((double)rand() / RAND_MAX) * 10.0;
    }

    printf("Density | NNZ   |  Dense(s) |  AVX2(s) |   CSR(s) | Speedup(AVX2/CSR)\n");
    printf("------------------------------------------------------------------\n");

    for (double density = 0.01; density <= 0.30; density += 0.05) {

        generate_matrix(A, rows, cols, density);

        int nnz;
        dense_to_csr(rows, cols, A, values, col_idx, row_ptr, &nnz);

        // ---- Naive dense ----
        double t1 = get_time();
        for (int r = 0; r < REPEAT; r++)
            dense_multiply(rows, cols, A, x, y_dense);
        double t2 = get_time();

        // ---- AVX2 dense ----
        double t1a = get_time();
        for (int r = 0; r < REPEAT; r++)
            dense_multiply_avx2(rows, cols, A, x, y_avx2);
        double t2a = get_time();

        // ---- CSR baseline ----
        double t3 = get_time();
        for (int r = 0; r < REPEAT; r++)
            csr_multiply(rows, values, col_idx, row_ptr, x, y_csr);
        double t4 = get_time();

        // ---- CSR 2-way unrolled ----
        double t5 = get_time();
        for (int r = 0; r < REPEAT; r++)
            csr_multiply_optimized(rows, values, col_idx, row_ptr, x, y_opt);
        double t6 = get_time();

        // ===== Correctness checks =====
        // All implementations must match the naive dense result
        if (!check(y_dense, y_avx2, rows) ||
            !check(y_dense, y_csr, rows)  ||
            !check(y_dense, y_opt, rows)) {
            printf("ERROR: Results mismatch at density %.2f!\n", density);
            return 1;
        }

        double dense_t  = (t2 - t1)   / REPEAT;
        double avx2_t   = (t2a - t1a) / REPEAT;
        double csr_t    = (t4 - t3)   / REPEAT;
        double opt_t    = (t6 - t5)   / REPEAT;

        double speedup_avx2_csr = avx2_t / csr_t;   // >1 means CSR faster than AVX2

        printf("%6.2f | %6d | %9.6f | %8.6f | %8.6f | %12.2fx\n",
               density, nnz, dense_t, avx2_t, csr_t, 1.0/speedup_avx2_csr);
    }

    free(A); free(x);
    free(y_dense); free(y_avx2); free(y_csr); free(y_opt);
    free(values); free(col_idx); free(row_ptr);

    return 0;
}
