# Sparse Matrix–Vector Multiplication Benchmark

This repository benchmarks the performance of **dense** and **Compressed Sparse Row (CSR)** matrix–vector multiplication (`y = A * x`).
A 1000×1000 matrix is filled with random values at varying densities (1% to 26%), and three kernels are compared:

1. **Naive dense** – straightforward double loop.
2. **AVX2-optimized dense** – uses SIMD intrinsics, loop unrolling, and prefetching.
3. **CSR sparse** – multiplies only non-zero elements after converting the matrix to CSR format.

The goal is to determine the **density crossover point** where a sparse format becomes faster than a heavily optimized dense routine.

---

## Table of Contents

- [Methodology](#methodology)
- [Code Structure](#code-structure)
- [Building and Running](#building-and-running)
- [Results](#results)
- [Interpreting the Results](#interpreting-the-results)
- [Key Findings](#key-findings)
- [System Specifications](#System-Specifications)
- [Conclusion](#conclusion)

---

## Methodology

### 1. Dense Multiplication (Naive)

A simple nested loop:

```c
for (int i = 0; i < rows; i++) {
    double sum = 0.0;
    for (int j = 0; j < cols; j++)
        sum += A[i*cols + j] * x[j];
    y[i] = sum;
}
```

It performs `rows × cols` multiply-add operations regardless of sparsity.

---

### 2. AVX2-Optimized Dense Multiplication

Leverages 256-bit SIMD (AVX2) and FMA (fused multiply-add) instructions:

- Processes **8 doubles per iteration** using two `__m256d` vectors.
- Accumulates results in two separate vector registers to maximise **instruction-level parallelism**.
- Uses `_mm_prefetch` to hint the hardware to load future cache lines early.
- Handles remaining columns with a **scalar tail loop**.

> **Requirement:** A CPU with AVX2 and FMA support (Intel Haswell+ / AMD Excavator+).

---

### 3. CSR Multiplication

The dense matrix is first converted to **Compressed Sparse Row (CSR)** format:

| Field | Description |
|---|---|
| `values[k]` | Non-zero entries |
| `col_indices[k]` | Column index of each non-zero |
| `row_ptrs[i]` | Start index in `values` for row `i` |

Multiplication then runs in **O(nnz)** time:

```c
for (int i = 0; i < rows; i++)
    for (int k = row_ptr[i]; k < row_ptr[i+1]; k++)
        y[i] += values[k] * x[col_idx[k]];
```

A second CSR variant with **2-way unrolling** is also included for completeness, though it shows no significant speed difference under modern compilers.

---

### 4. Timing & Validation

- Each kernel is executed **100 times** and the average time is reported.
- Timing uses `clock_gettime(CLOCK_MONOTONIC, …)`.
- Results are validated against the naive dense kernel with a mixed absolute/relative tolerance of `1e-7`.
- The matrix is regenerated for each density step with a fixed random seed (`42`) for reproducibility.

---

## Code Structure

| Function | Description |
|---|---|
| `get_time()` | Returns monotonic time in seconds |
| `dense_multiply()` | Naive dense matrix-vector product |
| `dense_multiply_avx2()` | SIMD-optimized dense product (AVX2 + FMA) |
| `dense_to_csr()` | Converts dense matrix to CSR format |
| `csr_multiply()` | Standard CSR multiplication |
| `csr_multiply_optimized()` | CSR with 2-way loop unrolling |
| `generate_matrix()` | Fills a matrix with random non-zeros at a given density |
| `check()` | Validates two result vectors within tolerance |
| `main()` | Runs the benchmark for densities from 0.01 to 0.30 |

> All memory is allocated once and reused; no dynamic allocation occurs inside the timed loops.

---

## Building and Running

### Requirements

- A C compiler supporting **C11** (GCC or Clang).
- A CPU with **AVX2 and FMA** support (virtually all x86-64 chips from ~2014 onward).
- **Linux/Unix** environment (for `clock_gettime`).

### Compilation

```bash
gcc -O3 -march=native -o benchmark sparse_benchmark.c -lm
```

The flag `-march=native` automatically enables AVX2, FMA, and other optimizations for your specific CPU.
If you prefer explicit flags, use `-mavx2 -mfma`.

### Execution

```bash
./benchmark
```

No command-line arguments are needed.

---

## Results

The following output was obtained on a modern x86-64 system.
All times are **per multiplication** (average over 100 repeats).

```
Density | NNZ   |  Dense(s) |  AVX2(s) |   CSR(s) | Speedup(AVX2/CSR)
------------------------------------------------------------------
  0.01 |  10048 |  0.001195 | 0.000204 | 0.000007 |         0.03x
  0.06 |  60040 |  0.001204 | 0.000214 | 0.000054 |         0.25x
  0.11 | 110229 |  0.001204 | 0.000213 | 0.000103 |         0.49x
  0.16 | 159680 |  0.001297 | 0.000288 | 0.000177 |         0.61x
  0.21 | 210379 |  0.001228 | 0.000226 | 0.000242 |         1.08x
  0.26 | 259418 |  0.001178 | 0.000224 | 0.000297 |         1.33x
```

### Column Definitions

| Column | Meaning |
|---|---|
| `Density` | Fraction of non-zero elements in the 1000×1000 matrix |
| `NNZ` | Actual number of non-zero entries |
| `Dense(s)` | Time for the naive double-loop dense multiplication |
| `AVX2(s)` | Time for the AVX2-optimized dense multiplication |
| `CSR(s)` | Time for the CSR sparse multiplication |
| `Speedup(AVX2/CSR)` | `CSR time / AVX2 time` — values < 1 mean CSR is faster; > 1 mean AVX2 is faster |

---

## Interpreting the Results

- **At 1% density:** CSR is ~33× faster than even the optimised AVX2 dense kernel. The AVX2 kernel still does 1,000,000 operations; CSR does only 10,048.

- **At 6% density:** CSR is still ~4× faster.

- **Around 11–16% density:** CSR holds a lead of 1.6–2×.

- **Crossover (~20% density):** The two approaches are nearly equal (ratio ≈ 1.0). At slightly higher densities, AVX2 begins to overtake CSR.

- **At 26% density:** AVX2 is 1.33× faster than CSR. The overhead of indirect addressing (`col_idx` lookups) and non-contiguous memory access begins to outweigh the benefit of skipping zeros.


---

## Key Findings

1. **Sparse formats win decisively when the matrix is truly sparse (< 10% non-zeros).**
   CSR can be tens of times faster due to its O(nnz) complexity and memory savings.

2. **The crossover point lies around 20% density on this hardware.**
   Beyond that, an aggressive SIMD-optimised dense routine becomes the better choice.

3. **Compiler optimisations matter.**
   The 2-way unrolled CSR shows no speedup over plain CSR under `-O3 -march=native`; the compiler already generates efficient code.

4. **Always benchmark against a realistic dense baseline.**
   A naive dense implementation can overstate the benefits of sparsity by an order of magnitude.

5. **CSR scaling is nearly linear in nnz**, confirming its O(nnz) behaviour.

---
## System Specifications
Processor: AMD Ryzen 9 6900HS
OS: Linux (Ubunbtu)

## Conclusion

For a 1000×1000 matrix:

| Density Range | Recommended Kernel |
|---|---|
| < ~10% | CSR (sparse) — large speedup |
| ~10–20% | CSR still faster, but gap closes |
| ~20% | Roughly equal performance |
| > ~20% | AVX2 dense — cache-friendly SIMD wins |

Choose your multiplication strategy based on the **actual sparsity of your data**. When in doubt, measure — the crossover point is hardware-dependent and can shift with matrix size, cache hierarchy, and memory bandwidth.
