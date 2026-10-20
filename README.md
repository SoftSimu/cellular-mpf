# Phase-field cell simulator for NVIDIA GH200

CUDA C++ solvers for interacting, deformable cells with run-and-tumble motility.
The canonical repository is [SoftSimu/phase-field-gh200](https://github.com/SoftSimu/phase-field-gh200).

| Executable | Geometry | Devices |
| --- | --- | --- |
| `cell_gh200` | Periodic 2D monolayer | One GPU |
| `cell_gh200_3d` | Periodic 3D, substrate slab, or two-hard-wall channel | One GPU |
| `cell_gh200_3d_two_gpu` | Substrate slab or two-hard-wall channel | Two matching, peer-connected GPUs |

## Model

Each cell is a scalar phase field. Explicit Euler integration combines
interfacial relaxation, a cell area/volume constraint, overlap repulsion and
advection. Cell velocity combines active propulsion and interaction-induced
translation. Counter-based Philox draws determine tumble times and directions.
Lengths are in unit lattice spacings; time is in model units.

The 2D formulation follows [Palmieri et al., *Scientific Reports* 5, 11745
(2015)](https://doi.org/10.1038/srep11745). It uses a nine-point Laplacian,
binary32 fields, and default Allen–Cahn mobility `M0=0.5`. An optional per-cell
mobility scales passive relaxation, not advection or velocity; see
[cell-specific mobility](docs/cell-specific-mobility.md). Coefficients are defined
in `common/model_coefficients.cuh` and `include/params.cuh`.

The 3D extension uses a 27-point Laplacian. Periodic cells have full-sphere target
volume and unrestricted 3D polarity. A substrate slab has neutral 90-degree
contact, hemispherical target volume and planar translation. A hard-wall channel
has two resolved steric walls, full-volume cells and unrestricted 3D motion.
These are distinct models; see [equations, boundaries and initialization](docs/three-dimensional-solver.md).

## Build

Requirements: CMake 3.24+, a C++17 compiler, and CUDA 12+. The default target is
`sm_90`. Production builds have been exercised on GH200 with NVHPC 26.3/CUDA
13.1; single-GPU checks also passed on H100 with GCC 12.3/CUDA 13.2. Other
hardware/toolchain combinations need their own verification. MPI is not used.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build --config Release --parallel 8
```

This builds both single-GPU solvers and the small initializer/export tools.
Use `-DPF_BUILD_3D=OFF` for 2D only. Add `-DPF_BUILD_TWO_GPU_SUBSTRATE=ON` to
build the two-GPU runner; it supports throughput storage and requires CUDA
peer access between the selected devices.

The default is the optimized production build. Do not add `-use_fast_math`.
Check the compiler's ptxas spill report when changing toolchains. Fatal integrity
checks remain enabled; `-DPF_DIAGNOSTIC_ALARMS=ON` additionally enables the
high-frequency support-edge counter, which is off by default.

```sh
cmake -S . -B build-diagnostic -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90 -DPF_DIAGNOSTIC_ALARMS=ON
cmake --build build-diagnostic --config Release --parallel 8
```

Optional `cmake --install build --prefix <directory>` installs the executables,
Python tools and documentation. CMake enables lossless output compression when
it finds `zstd.h` and libzstd; otherwise select `none` at runtime.

## Run and restart

Python 3.10+ can run the small argument-file examples:

```sh
python examples/run_example.py --executable build/cell_gh200
python examples/3d/run_example.py --executable build/cell_gh200_3d
```

Each invocation creates a new output directory. For direct use, a small channel
run with contour recording is:

```sh
./build/cell_gh200_3d --geometry channel --channel-height 8 \
  --N 4 --radius 4 --lambda 2 --rho 0.02 --brick-edge 48 \
  --kappa 0.1 --wall-kappa 0.1 \
  --dt 0.01 --t-end 0.1 --aging-time 0.01 --seed 1234 --polarity-seed 5678 \
  --out trajectory.txt --trajectory-interval 1 \
  --checkpoint-dir checkpoints --checkpoint-interval 5 \
  --surface-out interfaces.pfs3d --surface-interval 1 --surface-compression none

./build/cell_gh200_3d -c checkpoints/checkpoint.pf3d --t-end 0.2 \
  --out trajectory.txt --checkpoint-dir checkpoints \
  --surface-out interfaces-next.pfs3d --surface-interval 1
```

For two GPUs, replace the executable with `cell_gh200_3d_two_gpu` and add
`--device 0 --peer-device 1 --memory-mode throughput`. Run `--help` for all
inputs. Time and output intervals are integration units and accepted steps,
respectively: with `dt=0.01` and `tau=10000`, one persistence time is one million
steps. `--aging-time` is included in `--t-end`.

Checkpoints retain fields, stable cell IDs, polarity, adaptive storage and the
random-stream information needed for continuation. Use `checkpoint.bin` in 2D
and `checkpoint.pf3d` in 3D. A 3D continuation restores its stored model and
reduction grouping; it does not accept new model parameters. Pin the source,
compiler and executable for long trajectories. No cross-architecture bitwise
identity is promised.

## Outputs and postprocessing

- Text trajectories contain cell positions, velocities and shape measurements;
  their header defines the columns and model metadata.
- Checkpoints contain complete restart fields and are written transactionally.
- Optional [2D boundary samples](docs/boundary-output.md), [2D velocity
  components](docs/velocity-output.md), and [projected 3D boundaries](docs/boundary-output-3d.md)
  are independent observations.
- [3D contour files](docs/surface_output_3d.md) retain original binary32 samples
  around every cell's `phi=0.5` interface. They are not full fields or checkpoints.
  Use a new contour filename for each restart segment.

The optional CPU tools reconstruct surfaces, export PLY meshes and measure
persistent contact-graph changes. Install `tools/requirements-contacts.txt` in a
Python 3.11+ environment and follow [the postprocessing commands](docs/contact_exchange_3d.md).
Graph-exchange candidates are not confirmed elementary 3D T transitions.
Full-field slab rendering uses `tools/requirements-visualization.txt`.

## Small tests

The retained CPU references check model coefficients, Philox vectors, 2D
mobility and dense 3D one-step calculations. GPU smokes check output and restart;
the optional two-GPU smoke compares it with the single-GPU channel solver.

```sh
cmake -S . -B build -DBUILD_TESTING=ON -DPF_ENABLE_GPU_TESTS=ON
cmake --build build --config Release --parallel 8
ctest --test-dir build -C Release --output-on-failure
```

Omit `PF_ENABLE_GPU_TESTS` for the CPU checks. Optional contour/contact tests
require the Python analysis dependencies and `-DPF_ENABLE_CONTACT_TESTS=ON`.
These checks do not replace convergence, density or equilibration studies for
the intended experiment.

## Limits

Cells use adaptive local bricks, not a narrow-band approximation. Bricks grow
without clipping supported fields; insufficient memory or unsupported geometry
stops advancement. The 3D memory modes and support limits are described in the
[solver documentation](docs/three-dimensional-solver.md). The two-GPU runner
does not support periodic XYZ or multi-node execution. Steric walls have no
adhesion or wall friction; the neutral substrate model excludes multilayer
translation. Contour cadence and contact thresholds need scientific calibration.

## License and citation

LGPL-3.0-or-later, matching SymPhas. Keep both [LICENSE.txt](LICENSE.txt) and
[COPYING](COPYING). Citation metadata is in [CITATION.cff](CITATION.cff).
No third-party source is vendored. Dependencies retain their respective licenses.
The Philox algorithm is described by [Salmon et al., SC '11](https://doi.org/10.1145/2063384.2063405).
