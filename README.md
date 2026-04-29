# RV-Sparse: Sparse Matrix-Vector Multiplication via CSR

## Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [Mathematical Background](#2-mathematical-background)
3. [CSR Format — Theory and Construction](#3-csr-format--theory-and-construction)
4. [Algorithm Design](#4-algorithm-design)
5. [Implementation](#5-implementation)
6. [Step-by-Step Trace](#6-step-by-step-trace)
7. [Correctness and Complexity Analysis](#7-correctness-and-complexity-analysis)
8. [Test Harness and Results](#8-test-harness-and-results)
9. [Result Inference](#9-result-inference)
10. [Build and Run](#10-build-and-run)

---

## 1. Problem Statement

Implement `sparse_multiply` — a function that:

- Scans a row-major dense matrix `A` and identifies all non-zero elements
- Extracts them into **Compressed Sparse Row (CSR)** format using caller-provided buffers
- Computes the matrix-vector product `y = A * x` using the CSR data
- Writes the result directly into a caller-provided output buffer

**Critical Constraint:** Zero dynamic memory allocation. Every buffer is pre-allocated by the caller.

### Function Signature

```c
void sparse_multiply(
    int rows,           // number of rows in A
    int cols,           // number of columns in A
    const double* A,    // input matrix, row-major, size = rows * cols
    const double* x,    // input vector, size = cols
    int* out_nnz,       // output: number of non-zero elements found
    double* values,     // output buffer: non-zero values (CSR)
    int* col_indices,   // output buffer: column index of each non-zero (CSR)
    int* row_ptrs,      // output buffer: row boundaries (CSR), size = rows + 1
    double* y           // output vector: result of A * x, size = rows
);
```

---

## 2. Mathematical Background

### 2.1 Matrix-Vector Product

Given a matrix $A \in \mathbb{R}^{m \times n}$ and a vector $\mathbf{x} \in \mathbb{R}^{n}$, the product $\mathbf{y} = A\mathbf{x}$ produces a vector $\mathbf{y} \in \mathbb{R}^{m}$ where each element is defined as:

$$y_i = \sum_{j=0}^{n-1} A_{ij} \cdot x_j \quad \text{for } i = 0, 1, \ldots, m-1$$

This is a **dot product** of row $i$ of $A$ with the vector $\mathbf{x}$.

### 2.2 Why Sparse Multiplication is More Efficient

For a dense matrix the naive computation performs $m \times n$ multiplications and additions regardless of how many elements are zero.

For a sparse matrix with $\text{nnz}$ non-zero elements, if we only process the non-zeros:

$$y_i = \sum_{\substack{j=0 \\ A_{ij} \neq 0}}^{n-1} A_{ij} \cdot x_j$$

The number of operations drops from $O(mn)$ to $O(\text{nnz})$.

For a matrix with density $\rho$ (fraction of non-zero elements):

$$\text{nnz} = \rho \cdot m \cdot n$$

The speedup factor is:

$$\text{Speedup} = \frac{mn}{\rho \cdot mn} = \frac{1}{\rho}$$

So at 10% density ($\rho = 0.10$), sparse multiplication is theoretically **10× faster** than dense multiplication.

### 2.3 Row-Major Memory Layout

The matrix $A$ is stored as a flat 1D array in **row-major** order. The 2D element $A_{ij}$ maps to the 1D index:

$$\text{index}(i, j) = i \cdot \text{cols} + j$$

Visually for a $3 \times 4$ matrix:

```
2D:                    1D Memory (A[]):
A[0][0] A[0][1] A[0][2] A[0][3]    →   A[0]  A[1]  A[2]  A[3]
A[1][0] A[1][1] A[1][2] A[1][3]    →   A[4]  A[5]  A[6]  A[7]
A[2][0] A[2][1] A[2][2] A[2][3]    →   A[8]  A[9]  A[10] A[11]
```

---

## 3. CSR Format — Theory and Construction

### 3.1 Structure

CSR (Compressed Sparse Row) represents a sparse matrix using exactly three arrays:

| Array | Size | Contents |
|---|---|---|
| `values[]` | nnz | The non-zero values, in row-major order |
| `col_indices[]` | nnz | The column index of each non-zero value |
| `row_ptrs[]` | rows + 1 | For row $i$: the index in `values[]` where row $i$ begins |

These three arrays together fully describe the sparse matrix with no information loss.

### 3.2 Concrete Example

Consider the matrix:

$$A = \begin{bmatrix} 5 & 0 & 0 & 2 \\ 0 & 0 & 8 & 0 \\ 0 & 3 & 0 & 6 \end{bmatrix}$$

Reading non-zeros in row-major order:

| k | Value | Row | Col |
|---|-------|-----|-----|
| 0 | 5     | 0   | 0   |
| 1 | 2     | 0   | 3   |
| 2 | 8     | 1   | 2   |
| 3 | 3     | 2   | 1   |
| 4 | 6     | 2   | 3   |

This gives:

```
values      = [ 5,  2,  8,  3,  6 ]
               k=0  k=1  k=2  k=3  k=4

col_indices = [ 0,  3,  2,  1,  3 ]

row_ptrs    = [ 0,  2,  3,  5 ]
               ^    ^    ^    ^
               |    |    |    sentinel = total nnz
               |    |    row 2 starts at k=3
               |    row 1 starts at k=2
               row 0 starts at k=0
```

### 3.3 Row Slice Interpretation

The non-zeros belonging to row $i$ occupy the slice:

$$k \in [\text{row\_ptrs}[i],\ \text{row\_ptrs}[i+1])$$

That is, starting at index `row_ptrs[i]` (inclusive) up to `row_ptrs[i+1]` (exclusive).

The number of non-zeros in row $i$ is:

$$\text{nnz}_i = \text{row\_ptrs}[i+1] - \text{row\_ptrs}[i]$$

**Special cases:**

- If $\text{row\_ptrs}[i] = \text{row\_ptrs}[i+1]$, row $i$ has **zero** non-zeros (entire row is zero)
- $\text{row\_ptrs}[\text{rows}] = \text{nnz}$ always — this is the **sentinel** that closes the last row's slice

### 3.4 CSR-Based Sparse Matrix-Vector Product

Using CSR, the matrix-vector product $\mathbf{y} = A\mathbf{x}$ becomes:

$$y_i = \sum_{k=\text{row\_ptrs}[i]}^{\text{row\_ptrs}[i+1]-1} \text{values}[k] \cdot x[\text{col\_indices}[k]]$$

Applying this to row 2 of the example above (with $\mathbf{x} = [x_0, x_1, x_2, x_3]^T$):

$$y_2 = \text{values}[3] \cdot x[\text{col\_indices}[3]] + \text{values}[4] \cdot x[\text{col\_indices}[4]]$$
$$y_2 = 3 \cdot x_1 + 6 \cdot x_3$$

Elements $x_0$ and $x_2$ are **never accessed** — their corresponding matrix entries are zero.

---

## 4. Algorithm Design

The function operates in two sequential, independent phases:

### Phase 1: CSR Construction

```
nnz ← 0
for i = 0 to rows-1:
    row_ptrs[i] ← nnz                    // bookmark: this row starts here
    for j = 0 to cols-1:
        val ← A[i * cols + j]
        if val ≠ 0:
            values[nnz]      ← val        // store the non-zero value
            col_indices[nnz] ← j          // store its column
            nnz ← nnz + 1                 // advance cursor
row_ptrs[rows] ← nnz                      // sentinel
*out_nnz ← nnz                            // report total to caller
```

**Key invariant:** At the moment `row_ptrs[i] = nnz` executes, `nnz` equals the count of all non-zeros found in rows `0` through `i-1`. This makes it exactly the correct starting index for row `i`'s entries.

### Phase 2: Sparse Matrix-Vector Multiply

```
for i = 0 to rows-1:
    sum ← 0
    for k = row_ptrs[i] to row_ptrs[i+1]-1:
        sum ← sum + values[k] * x[col_indices[k]]
    y[i] ← sum
```

**No dynamic allocation anywhere.** Every write targets a buffer provided by the caller.

---

## 5. Implementation

```c
void sparse_multiply(
    int rows, int cols, const double* A, const double* x,
    int* out_nnz, double* values, int* col_indices, int* row_ptrs,
    double* y
) {
    int nnz = 0;

    /* ── Phase 1: Build CSR from dense matrix A ──────────────────────── */
    for (int i = 0; i < rows; i++) {
        row_ptrs[i] = nnz;               // row i's entries start at current nnz
        for (int j = 0; j < cols; j++) {
            double val = A[i * cols + j]; // row-major index
            if (val != 0.0) {
                values[nnz]      = val;   // store value
                col_indices[nnz] = j;     // store its column
                nnz++;                    // advance cursor
            }
        }
    }
    row_ptrs[rows] = nnz;                // sentinel: closes last row's range
    *out_nnz = nnz;                      // write count back to caller

    /* ── Phase 2: Compute y = A * x using CSR ────────────────────────── */
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        for (int k = row_ptrs[i]; k < row_ptrs[i + 1]; k++) {
            sum += values[k] * x[col_indices[k]];
        }
        y[i] = sum;                      // write result into caller's buffer
    }
}
```

### Design Decisions

**`val != 0.0` is safe here.** The test harness initialises the matrix with `calloc`, which sets all bytes to zero — producing IEEE 754 `+0.0` exactly. Non-zero entries are then assigned as random doubles. There is no floating-point arithmetic that could accidentally produce a near-zero result, so exact comparison is reliable. In a general library one would use an epsilon threshold.

**`*out_nnz = nnz` uses pointer dereference.** `out_nnz` is an `int*` pointing to the caller's variable. Writing `*out_nnz` stores into the caller's memory. Writing `out_nnz` (without `*`) would only change the local copy of the pointer and have no effect on the caller.

**`row_ptrs` has size `rows + 1`, not `rows`.** The extra slot holds the sentinel. Without it, Phase 2's upper bound `row_ptrs[i+1]` would read beyond the array for the last row.

**Phase 1 and Phase 2 are fully separated.** CSR construction is completed entirely before multiplication begins. This makes both phases independently verifiable and keeps the logic clean.

---

## 6. Step-by-Step Trace

### Input

$$A = \begin{bmatrix} 5 & 0 & 2 \\ 0 & 0 & 0 \\ 0 & 3 & 0 \end{bmatrix}, \quad \mathbf{x} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$$

### Phase 1 Trace

| Iteration | Action | `nnz` | `values` | `col_indices` | `row_ptrs` |
|---|---|---|---|---|---|
| Start | — | 0 | `[]` | `[]` | `[?, ?, ?, ?]` |
| i=0 | `row_ptrs[0] = 0` | 0 | `[]` | `[]` | `[0, ?, ?, ?]` |
| i=0, j=0 | val=5 ≠ 0, append | 1 | `[5]` | `[0]` | `[0, ?, ?, ?]` |
| i=0, j=1 | val=0, skip | 1 | `[5]` | `[0]` | `[0, ?, ?, ?]` |
| i=0, j=2 | val=2 ≠ 0, append | 2 | `[5, 2]` | `[0, 2]` | `[0, ?, ?, ?]` |
| i=1 | `row_ptrs[1] = 2` | 2 | `[5, 2]` | `[0, 2]` | `[0, 2, ?, ?]` |
| i=1, j=0..2 | all zero, skip | 2 | `[5, 2]` | `[0, 2]` | `[0, 2, ?, ?]` |
| i=2 | `row_ptrs[2] = 2` | 2 | `[5, 2]` | `[0, 2]` | `[0, 2, 2, ?]` |
| i=2, j=0 | val=0, skip | 2 | `[5, 2]` | `[0, 2]` | `[0, 2, 2, ?]` |
| i=2, j=1 | val=3 ≠ 0, append | 3 | `[5, 2, 3]` | `[0, 2, 1]` | `[0, 2, 2, ?]` |
| i=2, j=2 | val=0, skip | 3 | `[5, 2, 3]` | `[0, 2, 1]` | `[0, 2, 2, ?]` |
| Sentinel | `row_ptrs[3] = 3` | 3 | `[5, 2, 3]` | `[0, 2, 1]` | `[0, 2, 2, 3]` |

**Note how `row_ptrs[1] = row_ptrs[2] = 2`** — row 1 is all zeros, its slice `[2, 2)` is empty.

### Phase 2 Trace

$$y_0 = \sum_{k=0}^{1} \text{values}[k] \cdot x[\text{col\_indices}[k]]$$
$$= \text{values}[0] \cdot x[\text{col\_indices}[0]] + \text{values}[1] \cdot x[\text{col\_indices}[1]]$$
$$= 5 \cdot x[0] + 2 \cdot x[2] = 5 \times 1 + 2 \times 3 = 11$$

$$y_1 = \sum_{k=2}^{1} (\text{empty range}) = 0$$

$$y_2 = \sum_{k=2}^{2} \text{values}[k] \cdot x[\text{col\_indices}[k]]$$
$$= \text{values}[2] \cdot x[\text{col\_indices}[2]] = 3 \cdot x[1] = 3 \times 2 = 6$$

### Manual Verification

$$\mathbf{y} = A\mathbf{x} = \begin{bmatrix} 5 & 0 & 2 \\ 0 & 0 & 0 \\ 0 & 3 & 0 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} = \begin{bmatrix} 5(1) + 0(2) + 2(3) \\ 0(1) + 0(2) + 0(3) \\ 0(1) + 3(2) + 0(3) \end{bmatrix} = \begin{bmatrix} 11 \\ 0 \\ 6 \end{bmatrix} \checkmark$$

---

## 7. Correctness and Complexity Analysis

### 7.1 Correctness

The sparse product computes exactly the same sum as the dense product:

$$y_i^{\text{dense}} = \sum_{j=0}^{n-1} A_{ij} \cdot x_j = \sum_{\substack{j=0 \\ A_{ij} = 0}}^{n-1} \underbrace{0 \cdot x_j}_{=\ 0} + \sum_{\substack{j=0 \\ A_{ij} \neq 0}}^{n-1} A_{ij} \cdot x_j = y_i^{\text{sparse}}$$

Dropping zero terms does not change the sum. Therefore $\mathbf{y}^{\text{sparse}} = \mathbf{y}^{\text{dense}}$ exactly — no approximation, no rounding difference introduced.

### 7.2 Time Complexity

| Operation | Dense | Sparse (CSR) |
|---|---|---|
| CSR construction | — | $O(mn)$ — one scan of $A$ |
| Matrix-vector multiply | $O(mn)$ | $O(\text{nnz})$ |
| **Total** | $O(mn)$ | $O(mn + \text{nnz})$ |

Since $\text{nnz} \leq mn$ always, sparse is never worse. At density $\rho$: $\text{nnz} = \rho \cdot mn$, so multiply cost is $O(\rho \cdot mn)$ vs $O(mn)$ for dense — a factor of $\frac{1}{\rho}$ improvement.

### 7.3 Space Complexity

| Buffer | Size | Owner |
|---|---|---|
| `A` (input) | $mn$ doubles | Caller |
| `x` (input) | $n$ doubles | Caller |
| `values` | $\leq mn$ doubles | Caller |
| `col_indices` | $\leq mn$ ints | Caller |
| `row_ptrs` | $m + 1$ ints | Caller |
| `y` (output) | $m$ doubles | Caller |
| Function's own stack | $O(1)$ — just `nnz`, `i`, `j`, `k`, `val`, `sum` | Stack |

**The function itself allocates zero heap memory.** All buffers are owned and managed by the caller.

### 7.4 Edge Case Handling

| Edge Case | Behaviour |
|---|---|
| Entire row is zero | `row_ptrs[i] == row_ptrs[i+1]`, inner loop runs 0 times, `y[i] = 0` |
| Entire matrix is zero | `nnz = 0`, all `row_ptrs` equal 0, `y` is all zeros |
| Entirely dense matrix | Every element stored, `nnz = rows * cols`, works correctly |
| Single element matrix | One row, one column, handled naturally |
| Last row boundary | Sentinel `row_ptrs[rows] = nnz` provides upper bound |

---

## 8. Test Harness and Results

The test harness runs 100 randomised iterations. Each iteration:

1. Generates a random matrix of size $(\text{rows} \times \text{cols})$ with rows and cols in $[5, 45]$
2. Applies a random density $\rho \in [0.05, 0.40]$
3. Generates a random vector $\mathbf{x}$
4. Computes the **reference result** via naive dense multiply
5. Calls `sparse_multiply` and compares against reference using a mixed tolerance:

$$\text{tolerance}_i = 10^{-7} + 10^{-7} \cdot |y^{\text{ref}}_i|$$

This is a standard **mixed absolute-relative tolerance** — it is tight for small values and scales appropriately for large ones.

### Results

```
Iter  0 [ 36x 28, density=0.11, nnz= 113]: PASS (Max error: 0.00e+00)
Iter  1 [ 26x 13, density=0.25, nnz=  89]: PASS (Max error: 0.00e+00)
...
Iter 99 [ 28x  7, density=0.21, nnz=  39]: PASS (Max error: 0.00e+00)

All tests passed! (100/100 iterations passed)
```

**Max error is exactly `0.00e+00` on every iteration** — the sparse and dense results are bit-for-bit identical. This is expected because we compute the same floating-point additions in the same order; the only difference is that zero terms are omitted, which contributes nothing to the sum.

---

## 9. Result Inference

The following analysis is drawn directly from the actual test run output across all 100 iterations.

### 9.1 Headline Result

```
All tests passed! (100/100 iterations passed)
```

Every iteration produced a **Max error of exactly `0.00e+00`** — not close to zero, but precisely zero. This means the sparse CSR result and the brute-force dense reference were bit-for-bit identical on every single test case.

---

### 9.2 Test Coverage Statistics

The 100 iterations collectively exercised a wide and non-trivial parameter space:

| Metric | Min | Max | Observation |
|---|---|---|---|
| Rows | 5 | 45 | Full range of the harness |
| Cols | 5 | 45 | Full range of the harness |
| Density $\rho$ | 0.05 | 0.39 | Near-empty to nearly half-full |
| nnz | 4 | 644 | 160× spread in non-zero count |

**Approximate aggregate statistics across all 100 runs:**
- Average density: ~0.21 (21% non-zeros)
- Average nnz per test: ~145
- Total non-zero elements processed across all iterations: ~14,500
- Total matrix elements scanned (dense): ~70,000+

Every one of these accesses produced a correct result.

---

### 9.3 Stress Cases Identified in the Output

The harness randomly generates test parameters, but several iterations landed on particularly demanding cases worth noting explicitly:

#### Near-Empty Matrices (Extreme Sparsity)

```
Iter 47 [  9x  5, density=0.14, nnz=   4]   ← only 4 non-zeros in a 45-element matrix
Iter 29 [  6x 27, density=0.08, nnz=  12]   ← 8% density, nearly all zeros
Iter 70 [ 16x  9, density=0.08, nnz=  13]   ← 8% density again
Iter 84 [ 10x 45, density=0.06, nnz=  15]   ← 6% density, closest to all-zero
```

These test the most important edge case: **rows that are entirely zero**. When a whole row has no non-zeros, `row_ptrs[i] == row_ptrs[i+1]`, the inner multiply loop runs zero iterations, and `y[i]` correctly stays `0.0`. All passed without issue.

#### Dense-ish Matrices (High Fill Rate)

```
Iter 13 [ 43x 45, density=0.35, nnz= 644]   ← largest nnz in the entire run
Iter 17 [ 45x 36, density=0.39, nnz= 611]   ← highest density in the entire run
Iter 97 [ 39x 32, density=0.37, nnz= 467]   ← large matrix, high density
Iter 75 [ 40x 37, density=0.34, nnz= 476]   ← similarly large
```

At 35–39% density the CSR buffers are heavily populated. These tests verify that the `nnz` cursor advances correctly through hundreds of consecutive appends without off-by-one errors in indexing.

#### Highly Rectangular Matrices

```
// Far more columns than rows — wide matrix
Iter  5 [  5x 24, density=0.18, nnz=  24]
Iter 28 [ 13x 45, density=0.10, nnz=  61]
Iter 84 [ 10x 45, density=0.06, nnz=  15]

// Far more rows than columns — tall matrix
Iter  6 [ 45x  8, density=0.14, nnz=  48]
Iter 20 [ 26x  5, density=0.27, nnz=  30]
Iter 67 [ 28x  5, density=0.32, nnz=  43]
```

Rectangular matrices stress the `i * cols + j` row-major indexing formula — an off-by-one in `cols` would immediately corrupt access patterns for non-square matrices. All passed, confirming the indexing formula is correct for all aspect ratios.

---

### 9.4 What Zero Error Actually Means

The tolerance formula used by the harness is:

$$\text{tol}_i = 10^{-7} + 10^{-7} \cdot |y^{\text{ref}}_i|$$

This is a **mixed absolute-relative tolerance** — already a generous bound. Yet the actual error wasn't merely within this tolerance — it was zero. This is a stronger result than "close enough" and has a specific mathematical explanation:

The sparse multiply computes:

$$y_i^{\text{sparse}} = \sum_{\substack{k\ :\ A_{ij} \neq 0}} A_{ij} \cdot x_j$$

The dense reference computes:

$$y_i^{\text{dense}} = \sum_{j=0}^{n-1} A_{ij} \cdot x_j$$

Since $A_{ij} = 0$ contributes exactly $0.0 \times x_j = 0.0$ in IEEE 754 arithmetic (no rounding, no ULP error), dropping these terms changes nothing. The non-zero terms are accumulated in **the same order** in both implementations. Identical operands, identical order → identical floating-point result → zero error.

---

### 9.5 nnz Verification Sanity Check

The reported `nnz` values are consistent with the stated density and matrix dimensions:

$$\text{expected nnz} \approx \rho \times \text{rows} \times \text{cols}$$

Spot checks from the output:

| Iteration | Rows × Cols | Density | Expected nnz | Reported nnz | Match |
|---|---|---|---|---|---|
| Iter 13 | 43 × 45 = 1935 | 0.35 | ~677 | 644 | ✓ |
| Iter 17 | 45 × 36 = 1620 | 0.39 | ~632 | 611 | ✓ |
| Iter 47 | 9 × 5 = 45 | 0.14 | ~6 | 4 | ✓ |
| Iter 29 | 6 × 27 = 162 | 0.08 | ~13 | 12 | ✓ |

The small deviations from expected nnz are expected — density is a probability, so actual non-zero count varies around the mean. This confirms the CSR construction correctly counted and stored every non-zero found.

---

### 9.6 Summary of Inferences

| Claim | Evidence from Results |
|---|---|
| CSR construction is correct for all matrix shapes | Rectangular matrices (wide and tall) all PASS |
| Zero-row handling works | Near-empty matrices (nnz=4, nnz=12) all PASS |
| High-fill matrices handled without buffer errors | nnz=644 iteration PASS with zero error |
| Row-major indexing formula is correct | No failures across any rows/cols combination |
| `row_ptrs` sentinel correctly closes last row | Every iteration's last row computed correctly |
| No dynamic allocation issues | 100 iterations, zero segfaults, zero failures |
| Numerically identical to dense reference | Max error = 0.00e+00 on all 100 iterations |

The result is unambiguous: **the implementation is correct, robust, and numerically exact across the full parameter space exercised by the test harness.**

---

## 10. Build and Run

### Requirements
- GCC (any modern version)
- Standard C library + `libm`

### Compile

```bash
gcc -o run challenge.c -lm
```

> **Note:** `-lm` must come **after** the source file. GCC's linker resolves symbols left to right — placing `-lm` before `challenge.c` causes an `undefined reference to fmax` error because the math library is processed before the code that needs it.

### Run

```bash
./run
```

### Expected Output

```
All tests passed! (100/100 iterations passed)
```

### Exit Code

The program returns `0` on full pass, `1` if any iteration fails — suitable for use in CI pipelines or automated test scripts.


