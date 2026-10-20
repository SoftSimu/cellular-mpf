# Cellular MPF

Cellular MPF simulates the motion and deformation of interacting cells on
NVIDIA GPUs. Each cell has its own phase field, so cell shapes and contacts
evolve along with cell motion. The two-dimensional solver implements the
active multiphase-field model of [Palmieri et al. (2015)](https://doi.org/10.1038/srep11745),
with separate controls for interfacial energy, passive relaxation and propulsion.

The canonical repository is [SoftSimu/cellular-mpf](https://github.com/SoftSimu/cellular-mpf).
It contains the 2D monolayer solver and a 3D extension:

| Executable | Geometry | Devices |
| --- | --- | --- |
| `cell_gh200` | Periodic 2D monolayer | One GPU |
| `cell_gh200_3d` | Periodic 3D, substrate slab, or two-hard-wall channel | One GPU |
| `cell_gh200_3d_two_gpu` | Substrate slab or two-hard-wall channel | Two matching, peer-connected GPUs |

## Build

You need CUDA 12+, CMake 3.24+, and a C++17 compiler. Python 3.10+ runs the
examples and file readers. The commands below build the 2D solver for `sm_90`:

```sh
git clone https://github.com/SoftSimu/cellular-mpf.git
cd cellular-mpf
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90 -DPF_BUILD_3D=OFF
cmake --build build --config Release --parallel 8
```

Omit `-DPF_BUILD_3D=OFF` to also build the single-GPU 3D solver. Add
`-DPF_BUILD_TWO_GPU_SUBSTRATE=ON` to build the two-GPU runner; it requires CUDA
peer access and throughput storage. MPI is not used.

The optimized production build is the default. Fatal integrity checks remain
enabled; `-DPF_DIAGNOSTIC_ALARMS=ON` additionally enables the high-frequency
support-edge counter, which is off by default. Do not add `-use_fast_math`.
See the [build and hardware notes](docs/solver-reference.md#requirements) for
tested configurations, the diagnostic build and installation.

## Run the example

```sh
python3 examples/run_example.py \
  --executable ./build/cell_gh200 --output-dir ./example-output
```

This runs four cells for two timesteps and checks that a trajectory and a
checkpoint were written. Each invocation creates a new directory under
`example-output`; parameters are in [examples/smoke.args](examples/smoke.args).

Use `./build/cell_gh200 --help` to list all options. A
[longer example](docs/solver-reference.md#minimal-run) shows how to generate
initial cell positions and pass them to the simulator. After building 3D,
run its small example with:

```sh
python3 examples/3d/run_example.py --executable ./build/cell_gh200_3d
```

## Model parameters

| Quantity | Command-line option |
| --- | --- |
| Interfacial coefficient | `--gamma` for normal cells; `--gamma-cancer` for soft cells |
| Soft-cell fraction | `--cancer-fraction` |
| Passive relaxation mobility | `--phase-field-mobility`; individual values through `--phase-field-mobility-map` |
| Propulsion speed | `--v-A` |
| Polarity-persistence time | `--tau` |
| Friction coefficient in interaction advection | `--xi` |

The `cancer` options name the soft-cell population. A 2D mobility map assigns
values to stable cell IDs, allowing relaxation kinetics to vary at fixed
interfacial coefficients. Mobility scales passive relaxation, not advection
or velocity; see the [mobility guide](docs/cell-specific-mobility.md).
The [solver reference](docs/solver-reference.md#model-and-numerical-scheme)
describes the discretization and coefficient definitions. Lengths are in unit
lattice spacings and time is in model units.

## Outputs and restart

| Output | Option | Contents |
| --- | --- | --- |
| Trajectory | `--out` | Positions, velocities and shape measurements; the header defines the columns |
| Checkpoint | `--checkpoint-dir` | Full cell fields and state for restarting |
| 2D velocity contributions | `--velocity-moments` | Propulsion, interaction-advection and passive-relaxation contributions |
| 2D cell boundaries | `--boundary-out` | Field samples for reconstructing contours |
| Projected 3D boundaries | `--boundary-out` | Per-cell silhouettes or substrate footprints |
| 3D contours | `--surface-out` | Original binary32 samples around every cell's `phi=0.5` interface |

Recorders are optional. See the [2D velocity](docs/velocity-output.md),
[2D boundary](docs/boundary-output.md), [projected 3D boundary](docs/boundary-output-3d.md)
and [3D contour](docs/surface_output_3d.md) guides for formats and cadence.
Contours are observations, not full fields or checkpoints; use a new contour
filename for each restart segment.

To continue a 2D simulation, pass `-c path/to/checkpoint.bin` and a later
`--t-end`; 3D uses `checkpoint.pf3d`. Checkpoints retain stable cell IDs,
polarity and the random-stream state needed for continuation. See
[restart settings](docs/solver-reference.md#outputs-and-restart-behavior).

## Tests and documentation

Enable the small CPU references and GPU smoke tests explicitly:

```sh
cmake -S . -B build -DBUILD_TESTING=ON -DPF_ENABLE_GPU_TESTS=ON
cmake --build build --config Release --parallel 8
ctest --test-dir build -C Release --output-on-failure
```

Omit `PF_ENABLE_GPU_TESTS` for CPU-only checks. The optional two-GPU smoke
compares channel outputs and split restarts against the single-GPU solver.
[Test details](docs/solver-reference.md#tests) also cover the optional
contour/contact tests and their Python dependencies.

Further documentation covers:

- [Model, build variants, initialization and restart](docs/solver-reference.md).
- [Three-dimensional equations, geometries and storage](docs/three-dimensional-solver.md).
- [Starting 3D simulations from relaxed 2D positions](docs/relaxed-2d-slab-initialization.md).
- [Reconstructing 3D contours and analyzing contact exchanges](docs/contact_exchange_3d.md).

## Limits

Cells use adaptive local bricks, not a narrow-band approximation. Fields are
not clipped to fit; unsupported geometry or insufficient memory stops
advancement. The two-GPU runner does not support periodic XYZ or multi-node
execution. Steric walls have no adhesion or wall friction; the neutral
substrate model excludes multilayer translation. Contact-graph exchange
candidates are not confirmed elementary 3D T transitions. Convergence,
equilibration, contour cadence and contact thresholds need validation for the
intended experiment. Cross-architecture bitwise identity is not promised.

## Citation and license

Software citation details are in [CITATION.cff](CITATION.cff). The underlying
2D model is described by Palmieri et al., *Scientific Reports* **5**, 11745
(2015), [doi:10.1038/srep11745](https://doi.org/10.1038/srep11745).

The code is distributed under LGPL-3.0-or-later, matching SymPhas. Keep both
[LICENSE.txt](LICENSE.txt) and [COPYING](COPYING). No third-party source is
vendored; dependencies retain their respective licenses. The Philox algorithm
is described by [Salmon et al., SC '11](https://doi.org/10.1145/2063384.2063405).
