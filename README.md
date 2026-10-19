# Tiled 2D stencil PDE solver on NVIDIA Ampere (Barkley model)

CUDA C++ optimization study of an explicit 2D reaction–diffusion solver (Barkley model of excitable media) on an NVIDIA RTX A5000 (Ampere, sm_86).

- Problem design and optimization strategy: Ferenc Hegedűs (Budapest University of Technology and Economics)
- Implementation: János Jonatán Gedei, student research project (TDK), 2024
- Discretization: finite differences with a 9-point (Patra–Karttunen / Mehrstellen) Laplacian stencil, 256 × 256 grid, FP32

## Versions and results

Each folder is a self-contained step of the optimization. Runtime is the average time per iteration over 3,500,000 iterations.

| Version | Technique | Time per iteration | vs. fastest |
| --- | --- | --- | --- |
| `v3.4_measure` | Naive: one kernel launch per time step | 15,640 ns | +565% |
| `v4.2` | Persistent cooperative kernel, iteration loop inside the kernel, `grid.sync()` between steps | 2,355 ns | +0.1% |
| `v5.2` | v4.2 + persistent L2 cache (access policy window on the solution buffers) | 2,352 ns | fastest |
| `v6.2_smem_fix` | Shared-memory tiles, naive halo handling | 2,789 ns | +18.6% |
| `v6.6_full_1D_smem` | Shared-memory tiles; tile halos stored in a separate L2-persistent buffer so that both row and column halo exchanges are coalesced | 2,522 ns | +7.2% |

Main finding: keeping the whole working set L2-resident through a persistent kernel gives a 6.6× speedup. The shared-memory variants did not improve on it. The likely limits are the cost of grid-wide synchronization, extra branching, and register pressure (59 of 64 registers per thread in v6.6).

## Correctness

Every version is validated against a MATLAB reference solution after 5,000 iterations (`reference/u_comp_5000.txt`). `tools/comp_2.py` compares two solutions and `tools/show_1.py` plots one.

## Building and running

Requires the CUDA toolkit and an Ampere GPU (sm_86). In each version folder:

```
make
./kernel.exe
```

Notes:

- `DEVICE_ID` is hard-coded to 1 (the second GPU of the original two-GPU machine). Set it to 0 in `kernel.cu` on a single-GPU system.
- Initial conditions are read from `u_ff.txt` / `v_ff.txt` (`u_fc.txt` / `v_fc.txt` in v6.6) in the working directory.
- Code comments are partly in Hungarian.

The project was supported by the NVIDIA Academic Hardware Grant Program.
