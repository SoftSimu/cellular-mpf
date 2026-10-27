# Solver reference

[Return to the README](../README.md). Commands run from the repository root.

## Model and numerical scheme

Each cell has a binary32 phase field advanced with explicit Euler integration.
The passive terms describe interfacial relaxation, an area/volume constraint
and overlap repulsion; advection translates the cell with its active and
interaction-induced velocity. Counter-based Philox draws determine tumble
times and polarity. All lengths use unit lattice spacing and time is in model
units; no physical-unit conversion is implied.

The [2D equations and mobility guide](cell-specific-mobility.md) defines the
coefficients and the per-cell Allen-Cahn mobility. The 2D stencil uses a
nine-point isotropic Laplacian and centered gradients on a periodic square.
Mobility scales every passive phase-field term, but never advection,
translation velocity, polarity or the tumble stream. Coefficient definitions
are in [`common/model_coefficients.cuh`](../common/model_coefficients.cuh) and
[`include/params.cuh`](../include/params.cuh).

The [3D guide](three-dimensional-solver.md) gives geometry choices and run
examples. The 3D solver uses an isotropic 27-point Laplacian, with face, edge,
corner and centre weights of 14, 3, 1 and -128, divided by 30. Coefficients and
boundary terms are defined in [`include/pf3d/params.cuh`](../include/pf3d/params.cuh)
and [`include/pf3d/reference.hpp`](../include/pf3d/reference.hpp). Periodic cells
have full-sphere target volume and unrestricted 3D polarity. Substrate cells have hemispherical target
volume, neutral 90-degree contact and planar translation. The hard-wall channel
contains full-volume cells with unrestricted 3D motion between resolved steric
walls. These are distinct geometries, not interchangeable boundary settings
for a single physical model. Per-cell mobility maps are 2D-only.

The 2D and 3D numerical cores share coefficient and Philox primitives in
`common/`; their fields, stencils, counters and update kernels are
dimension-specific. All 3D geometries share checkpoint, adaptive storage,
measurement and general update code. The two-GPU runner partitions whole cells
and exchanges aggregate interaction contributions through CUDA peer access;
it does not use MPI or domain-decomposed cell fields.

## Requirements

- CMake 3.24+, a C++17 compiler and CUDA 12+.
- NVIDIA hardware supporting the chosen CUDA architecture; `sm_90` is the default.
- The 3D bulk-copy loader requires NVIDIA nvcc 12.5+ and explicit architecture targets
  of 90 or newer. H100/GH200 are tested; newer architectures are not certified.
- Python 3.10+ for examples, readers and the small CPU-test configuration.
- Python 3.11+ and the pinned analysis dependencies for contour/contact tests
  or full-field visualization.

The current 3D bulk-copy path has been exercised on GH200 with NVHPC 26.3/CUDA
13.1. A clean Windows build with MSVC 19.44/CUDA 12.5.82 passed all ten retained
CPU and postprocessing tests; this does not establish GPU performance.
Earlier single-GPU checks on H100 with GCC 12.3/CUDA 13.2 predate the bulk-copy
path. Two-device execution requires matching GPUs with CUDA peer access;
the tested configuration is two GH200s on one node.

On Windows, put the matching toolkit's `bin` directory first on `PATH`; an
older `cudart64_12.dll` can lack the entry points needed by the 3D loader.

## Build

The default optimized build includes both single-GPU solvers and the small
initializer/export tools:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build --config Release --parallel 8
```

Use `-DPF_BUILD_3D=OFF` for 2D only. Set `-DPF_BUILD_TWO_GPU_SUBSTRATE=ON`
to add `cell_gh200_3d_two_gpu`; this requires 3D and supports substrate or
channel geometry with throughput storage. The option name is retained for
compatibility; both bounded-z geometries use the same runner.

Fatal integrity checks remain enabled in the optimized build. The diagnostic
variant adds the high-frequency support-edge counter:

```sh
cmake -S . -B build-diagnostic -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90 -DPF_DIAGNOSTIC_ALARMS=ON
cmake --build build-diagnostic --config Release --parallel 8
```

Do not add `-use_fast_math`. The build preserves the validated contraction
policy and prints ptxas register/spill reports. Review those reports when
changing compilers; successful compilation alone does not establish equal
performance. Optional `cmake --install build --config Release --prefix <path>`
installs executables and Python tools in `bin` and documentation/examples in
`share/doc/phase_field_gh200`. CMake enables lossless compression when it finds
both `zstd.h` and libzstd; otherwise select `none` at runtime.

## Minimal run

The [four-cell 2D example](../examples/smoke.args) checks initialization,
trajectory output and checkpoint writing in two timesteps:

```sh
python3 examples/run_example.py \
  --executable ./build/cell_gh200 --output-dir ./example-output
```

For a research run, generate and retain an explicit centre table. The 2D
domain rule `L=ceil(sqrt(N*pi*R^2/rho))` gives `L=1554` for the inputs below;
the initializer's `--side` must match that derived side.

```sh
./build/palmieri_centres --N 288 --side 1554 --radius 49 --seed 1234 \
  --out initial_centres.csv

./build/cell_gh200 --N 288 --radius 49 --rho 0.90 \
  --dt 0.01 --t-end 100 --lambda 7 --kappa 10 --mu 1 --xi 1500 \
  --tau 10000 --v-A 0.01 --gamma 1 --seed 1234 --polarity-seed 1234 \
  --initial-centres initial_centres.csv \
  --out trajectory.txt --checkpoint-dir checkpoints --checkpoint-interval 10000
```

The table uses stable `global_id,x,y` identities. Without a table, the 2D
solver uses grid-and-jitter placement, as in the smoke example. The soft cohort
consists of the lowest `round(cancer_fraction*N)` global IDs. In 3D the seeded
soft cohort is sampled without replacement instead. Retain both simulation
and polarity seeds for paired experiments.

After building 3D, its small example also exercises checkpoint continuation:

```sh
python3 examples/3d/run_example.py --executable ./build/cell_gh200_3d
```

A small channel with contour recording is:

```sh
./build/cell_gh200_3d --geometry channel --channel-height 8 \
  --N 4 --radius 4 --lambda 2 --rho 0.02 --brick-edge 48 \
  --kappa 0.1 --wall-kappa 0.1 --dt 0.01 --t-end 0.1 \
  --aging-time 0.01 --seed 1234 --polarity-seed 5678 \
  --out trajectory.txt --trajectory-interval 1 \
  --checkpoint-dir checkpoints --checkpoint-interval 5 \
  --surface-out interfaces.pfs3d --surface-interval 1 --surface-compression none
```

This is a low-memory software example, not a calibrated experimental setup.
For two GPUs use `cell_gh200_3d_two_gpu` with
`--device 0 --peer-device 1 --memory-mode throughput`.
See [3D examples](three-dimensional-solver.md) and
[relaxed 2D placement transfer](relaxed-2d-slab-initialization.md) before
preparing larger runs.

## Outputs and restart behavior

Time arguments use integration units. `--aging-time` is included in
`--t-end`; with `dt=0.01` and `tau=10000`, one persistence time is one million
steps. Output intervals are accepted integration steps, not trajectory frames.

For 2D, `-c checkpoints/checkpoint.bin --t-end <later-time>` restores fields,
stable IDs, polarity, step, mobility, random streams and adaptive-window
state. Existing trajectories are extended only after exact metadata and
complete-frame validation. The cached interface measure is updated at
`--full-moment` cadence and retained between those measurements.

For 3D, use `-c checkpoints/checkpoint.pf3d`. The CRC-protected checkpoint
retains each logical brick, field, origin, velocity, polarity, stable identity
and resolved measurement grouping. Model parameters are restored from the
checkpoint rather than replaced by fresh-run options. A conflicting grouping
is rejected unless the documented promoted-regroup option is explicitly used.
The logical payload remains cubic even when bounded-z GPU allocations omit
exterior planes; height-matched storage saves device memory, not checkpoint
disk space.

The optimized 3D kernels reuse stencil and moment intermediates and move
interior bounds decisions outside voxel loops. The stencil weights, time step
and precision are unchanged; regrouping floating-point operations can change
roundoff between builds. The complete single-GPU channel update also bounds
base-cell overlap before accumulating the integer interaction field. Values
outside that bound use checked additions, as do promoted cells and partitioned
two-GPU updates. Fatal integrity checks and checkpoint formats are unchanged.

Repeated continuations preserve the tumble stream with the same accepted
step, stable IDs, seeds and parameters. Pin source, compiler and executable
for long trajectories; no cross-architecture bitwise identity is promised.
Trajectory cadence is restored unless explicitly changed. A cadence change
requires a new trajectory path. Boundary/contour streams also use new filenames
for each restart segment.

The [output guides](../README.md#outputs-and-restart) describe independent
velocity, boundary and contour recorders. For full 3D contours, follow
[surface recording](surface_output_3d.md) and
[offline reconstruction/contact analysis](contact_exchange_3d.md). A contact
graph change is a candidate event, not proof of an elementary 3D T transition.

## Tests

Tests are opt-in. The retained CPU references check common coefficient/Philox
primitives, 2D mobility and independent dense 3D one-step calculations:

```sh
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --config Release --parallel 8
ctest --test-dir build -C Release --output-on-failure
```

Add `-DPF_ENABLE_GPU_TESTS=ON` to register the 2D and 3D output/restart smokes.
If `PF_BUILD_TWO_GPU_SUBSTRATE` is also enabled, the two-device smoke compares
channel checkpoints, trajectories and contour samples with the single-device
solver and exercises active tumbles and split continuation. Across device
counts, the five-step test permits floating-point roundoff in fields and
measurements; IDs, polarity, counters and contour samples must match exactly.
The same two-device executable must restart byte-identically. These tests run on
the caller's GPU allocation; they never submit cluster jobs themselves.

For the optional contour/contact tests, install
`tools/requirements-contacts.txt` in a Python 3.11+ environment and enable
`-DPF_ENABLE_CONTACT_TESTS=ON` together with `BUILD_TESTING`.
These checks establish software behavior for their fixtures, not convergence,
equilibration or scientific validity for arbitrary experiment parameters.
