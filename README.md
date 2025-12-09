# einsum (development version)

This repository contains the development work for a lightweight, high-performance `einsum` implementation based on a **Transpose–Transpose–BMM–Transpose** strategy. It includes all experimental versions (`v1`–`v8`) of the C++ BMM kernels and the Python interface, as well as the benchmarking and test scripts used during optimization.

The project was created for the **Algorithm-Engineering** course at FSU Jena and explores how view-based layouts together with a tuned BMM backend can significantly outperform NumPy’s `einsum` and achieve strong performance on memory-bound cases compared to PyTorch.

