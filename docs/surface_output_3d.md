# Three-dimensional interface observations

`--boundary-out` saves a basal or maximum-intensity **2D projection**. It cannot
resolve contacts between stacked cells. `--surface-out` instead saves the samples
needed to reconstruct each cell's three-dimensional 0.5 interface.

This is optional, read-only output. It changes neither the equations nor the
timestep, field precision, polarity, random draws, or checkpoint format. It does
not itself implement a contact graph or identify rearrangements. The separate
[contact/exchange tools](contact_exchange_3d.md) measure geometric proximity and
persistent graph changes, with explicit limits on topological interpretation.

## Recording

Build normally. If CMake finds `zstd.h` and libzstd, compression is available;
otherwise select `none`. For example, add these flags to an ordinary run or
checkpoint continuation:

```sh
--surface-out leg_001.pfs3d --surface-interval 10000 --surface-compression zstd
```

Cadence is in **accepted steps**, on an absolute grid. With `dt=0.01` and
`tau=10000`, 10000 steps are 0.01 tau. Every segment also includes its initial and
last accepted state. A restart requires a **new filename**; existing files are
never appended or overwritten. De-duplicate repeated segment endpoints by
persistent cell ID and accepted step when combining observations. Two files at
the same step must agree before one is discarded.

On two GPUs, packing reads each cell on its owning GPU. It does not gather full
fields or rebuild the aggregate merely for this observer. If another observer
already gathered the state, the recorder can reuse that complete primary view.
Capture and writing finish before the source buffer is reused. Slow storage
therefore applies backpressure; frames are not silently dropped.

The device packing buffer and matching pinned host buffer are each bounded by
64 MiB **per participating GPU**, plus small count/metadata arrays. CPU compression
uses one additional bounded chunk buffer. Large cells are split into chunks;
there is no full-brick host copy or per-voxel flag allocation. These are output
buffers, not a narrow-band approximation to the dynamics. The current CUDA launch
supports at most 65535 selected cells in one device view; larger views are rejected.

## Retained information

Inside means `phi >= 0.5`. Around each logical brick, one layer of zero-valued
vertices is added for output. A vertex is saved when any of its 26 neighbours
has the opposite classification. This is exactly the union of the corners of
all crossing cubes: opposite corners of a cube differ by at most one lattice
site in each direction, and two such neighbours belong to a common cube.

Every corner value of every crossing cube is retained at its original float32
precision, including ties at 0.5. Each vertex appears once per cell. The rule
defines the binary interface consistently in degenerate cases; a volume of
identically 0.5 values is not interpreted as an additional interior surface.
Samples away from crossing cubes are omitted. Consequently:

- The dense field's crossing-cube data can be recovered exactly.
- A different isovalue, arbitrary overlap integral, or complete restart cannot
  be recovered. Retain ordinary checkpoints for those purposes.
- Missing interior vertices must **not** be filled with zero: that creates a
  false internal interface. Use `crossing_cubes()` to recover only valid cubes.
- Mesh extraction still needs an explicit ambiguity-handling convention. This
  stream does not prescribe triangulation or claim a topological guarantee for
  an unspecified marching-cubes implementation.

Every frame carries persistent cell IDs, unwrapped brick origins, actual support
edges, stiffness, active speed, radius, polarity, velocity, and tumble count.
The file header stores the lattice dimensions and spacing, boundary conditions,
resolved channel walls, interface width, timestep, tau, isovalue, and cadence.
No cell identity is inferred from vertical position or output order.

## Coordinates and artificial caps

For a brick of edge B, padded coordinates run from 0 to B+1. World lattice
coordinates are `origin + padded_coordinate - 1`. Periodic comparisons must use
the global box lengths; do not independently wrap a cell's local brick.

In a resolved channel, physical height is
`z_physical = (z_lattice - channel_padding + 0.5) * dz`; the slit lies between
0 and `channel_height * dz`. Wall padding is part of the computational array,
not extra accessible volume. A substrate slab uses its original basal plane.

Only actual computational planes are recorded, not reflected PDE ghosts.
Zero extension can therefore close the basal side of a substrate cell. This
cap is an **output convention**, not a free interface or a new wall force.
`closure_flags()` identifies cubes touching the brick's exterior shell and
those crossing a bounded computational boundary. Exclude or separately classify
these caps in cell-cell contact analysis. A crossing at an x/y brick edge also
deserves a support-sizing warning rather than being treated as a real cell face.

## Reading and verification

The reader itself needs Python 3.9+. Install `numpy` for reconstruction/tests and
`zstandard` to read compressed files:

```sh
python -m pip install numpy zstandard
python tools/pf3d_surface.py leg_001.pfs3d
python tests/3d/surface_contract_test.py
```

The reader keeps one bounded frame in memory, with a default 512 MiB decoded
frame limit. Raise `--max-frame-mib` deliberately for larger trusted frames.
Only frames with a valid commit footer are yielded. Reading to exhaustion checks
the terminal completion marker. `--recover` permits an incomplete tail and
returns earlier committed frames, but does not ignore checksum corruption.

```python
from pf3d_surface import SurfaceReader, crossing_cubes, closure_flags

with SurfaceReader("leg_001.pfs3d") as reader:
    for frame in reader:
        for cell in frame.cells:
            lower_corners, phi8 = crossing_cubes(cell)
            caps = closure_flags(reader.header, cell, lower_corners)
            # phi8 corner order: 000,100,010,110,001,101,011,111.
            # Construct interfaces from these complete cubes, not a zero-filled field.
```

For the inexpensive synthetic GPU comparison, configure with
`-DPF_ENABLE_GPU_TESTS=ON`, build `pf3d_surface_gpu_test`, and run:

```sh
ctest --test-dir build --output-on-failure -R '^pf3d_surface_(contract|gpu)$'
```

Fixtures compare every retained sample and reconstructed crossing cube to an
independent dense CPU oracle, including exact 0.5 values, isolated voxels,
checkerboards, promoted bricks, non-consecutive IDs, unwrapped origins, and
compact/cubic bounded storage. CPU tests also check corruption, bounds, and
every tail truncation of a small frame transaction.

## Cost and exchange analysis

Capture cost and compressed bytes are reported per frame. Measure these on the
actual evolved population before choosing a production cadence: compressed
surface output can still accumulate to hundreds of GiB over a long trajectory.
Use bounded run segments and archive completed segments; do not rely on scratch
to retain a whole campaign.

A 3D neighbour exchange is not automatically the four-cell T1 event of a planar
tiling. Scientific interpretation needs a calibrated three-dimensional proximity/contact
definition, periodic-image handling, exclusion of wall caps, persistence or
hysteresis, and a temporal-resolution check. Diffuse cells may have separated
0.5 surfaces even when they interact. Occasional full checkpoints can calibrate
a surface-distance criterion against the underlying phase fields. Recording
surfaces is the data-collection step, not evidence that these measurements are
already validated.

## Binary contract

`common/surface_format_3d.h` defines the little-endian version1 records. A file
contains a checksummed header, then frames consisting of a frame header, one
cell header and zero or more vertex chunks per cell, and a frame commit footer.
A terminal marker records successful stream closure. CRC-32/ISO-HDLC checks the
header, decoded chunks, and on-disk frame transaction; it is not authentication.
Vertex records contain a uint64 padded-grid index and float32 sample (12 bytes).
Cell/chunk order is not a persistent identity: different device ownership can
change packing order without changing the saved samples.

The standalone mesh exporter and contact/change/catalog workflow are described
in [contact and exchange analysis](contact_exchange_3d.md). Both consume these
saved files; neither needs a running simulator. Keep the `.pfs3d` streams, their
run manifests and occasional full checkpoints together when archiving a run.
Exported meshes alone do not retain the original crossing-cube samples.
