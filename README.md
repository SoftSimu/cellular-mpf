# Cellular MPF

Cellular MPF simulates the motion and deformation of interacting cells on
NVIDIA GPUs. Each cell has its own phase field, so cell shapes and contacts
evolve along with cell motion. The two-dimensional solver implements the
active multiphase-field model of [Palmieri et al. (2015)](https://doi.org/10.1038/srep11745),
with separate controls for interfacial energy, passive relaxation and propulsion.

This repository accompanies *Relaxation kinetics controls cell transport in
crowded active monolayers*, by Steven A. Silber and Mikko Karttunen. The paper
uses the two-dimensional solver, `cell_gh200`. The repository also includes a
[three-dimensional solver](docs/three-dimensional-solver.md) for periodic
systems, cells on a substrate and cells confined between two walls.

## Build

You need an NVIDIA GPU, CUDA 12 or newer, CMake 3.24 or newer, and a C++17
compiler. Python 3.10 or newer is used by the example scripts and file readers.
The commands below build the 2D solver on Linux for H100 or GH200 GPUs
(`sm_90`):

```bash
git clone https://github.com/SoftSimu/cellular-mpf.git
cd cellular-mpf
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DPF_BUILD_3D=OFF
cmake --build build --parallel 8
```

For other architectures, Windows builds and the optional 3D executables, see
the [build notes](docs/solver-reference.md#requirements).

## Run the example

```bash
python3 examples/run_example.py \
  --executable ./build/cell_gh200 \
  --output-dir ./example-output
```

This runs four cells for two timesteps and checks that a trajectory and a
checkpoint were written. Each invocation creates a new directory under
`example-output`. The parameters are in [examples/smoke.args](examples/smoke.args).

Use `./build/cell_gh200 --help` to list all options. A
[longer example](docs/solver-reference.md#minimal-run) shows how to generate
initial cell positions and pass them to the simulator. Simulation preparation,
parameter choices and transport measurements for the paper are described in
its Methods and Supplementary Information.

## Model parameters

The main 2D options correspond to the following quantities in the paper:

| Quantity | Command-line option |
| --- | --- |
| Interfacial coefficient | `--gamma` for normal cells; `--gamma-cancer` for soft cells |
| Soft-cell fraction | `--cancer-fraction` |
| Passive relaxation mobility | `--phase-field-mobility`; individual values through `--phase-field-mobility-map` |
| Propulsion speed | `--v-A` |
| Polarity-persistence time | `--tau` |
| Friction coefficient in interaction advection | `--xi` |

The `cancer` options name the soft-cell population. A mobility map assigns
values to individual cell IDs, allowing relaxation kinetics to be varied at
fixed interfacial coefficients. See the [mobility guide](docs/cell-specific-mobility.md)
for the file format and restart options. The equations and numerical scheme
are given in the [solver reference](docs/solver-reference.md#model-and-numerical-scheme).

## Outputs

| Output | Option | Contents |
| --- | --- | --- |
| Trajectory | `--out` | Cell positions, velocities, polarities, areas and interface measures |
| Checkpoint | `--checkpoint-dir` | Cell fields and simulation state for restarting |
| Velocity contributions | `--velocity-moments` | Propulsion, interaction-advection and passive-relaxation contributions |
| Cell boundaries | `--boundary-out` | Field samples for reconstructing contours and tracking neighbor exchanges |

Velocity and boundary recording are optional. Their formats, sampling options
and Python readers are described in the [velocity](docs/velocity-output.md)
and [boundary](docs/boundary-output.md) guides. Set `--boundary-interval` in
integration steps to choose how often boundaries are saved.

To continue a simulation, pass `-c path/to/checkpoint.bin` and a later
`--t-end`. See [outputs and restarts](docs/solver-reference.md#outputs-and-restart-behavior)
for trajectory continuation and recorder settings.

## Tests and documentation

Run the tests registered by the build with:

```bash
ctest --test-dir build --output-on-failure
```

The [test guide](docs/solver-reference.md#tests) describes the optional GPU
and checkpoint tests. Further documentation covers:

- [Implementation, validation and performance measurements](docs/solver-reference.md).
- [Three-dimensional model and geometries](docs/three-dimensional-solver.md).
- [Starting 3D simulations from relaxed 2D cell positions](docs/relaxed-2d-slab-initialization.md).
- [Three-dimensional boundary output](docs/boundary-output-3d.md).

## Citation and license

Software citation details are in [CITATION.cff](CITATION.cff).

The underlying model is described in B. Palmieri, Y. Bresler, D. Wirtz and
M. Grant, *Multiple scale model for cell migration in monolayers: elastic
mismatch between cells enhances motility*, Scientific Reports **5**, 11745
(2015), [doi:10.1038/srep11745](https://doi.org/10.1038/srep11745).

The code is distributed under the GNU Lesser General Public License, version 3
or later. See [LICENSE.txt](LICENSE.txt) and [COPYING](COPYING).
