# Three-dimensional contacts and exchange candidates

The surface recorder preserves the original float32 samples needed for each
cell's 0.5 interface. The offline tools below measure **geometric proximity**
between those interfaces and persistent changes in the resulting graph. They
do not change the simulation and do not equate every contact change with a T1.

## Run

Use Python 3.11+ in a separate analysis environment. Install the dependencies
in `tools/requirements-contacts.txt`; no CUDA device is needed.

```sh
python -m pip install -r tools/requirements-contacts.txt
python tools/pf3d_analyze_contacts.py measure leg_001.pfs3d leg_002.pfs3d \
  --gap 2 --workers 4 --output contacts.jsonl
python tools/pf3d_analyze_contacts.py track contacts.jsonl \
  --area-on 10 --area-off 5 --persistence-steps 20000 \
  --max-frame-gap 10000 --exchange-window 20000 --output changes.jsonl
python tools/pf3d_analyze_contacts.py catalog changes.jsonl --output-dir events
```

**These numbers illustrate syntax, not calibrated scientific thresholds.** Gap
and area are in the physical length and area units of the simulation. Persistence,
maximum frame gap and exchange window are accepted steps. Multiply by `dt/tau`
to express a step interval in units of tau. A positive persistence interval needs
at least two decisive observations. `--stride 2` on `track` provides a cadence
check without re-extracting surfaces; set the allowed frame gap accordingly.

Inputs are ordered run segments from the same cell cohort and geometry. Codec
and cadence may differ and are recorded per segment. Repeated endpoints must
agree in **all recorded cell metadata and exact samples**, independent of packing
order. Other overlaps, changed identities, invalid checksums and inconsistent
geometry are errors. Outputs are created exclusively, never overwritten.

`measure` holds one decoded frame plus extracted meshes and per-cell distance
indices, not the whole time series. Its default decoded frame limit is 512 MiB;
the dense reconstruction limit is 32 million padded voxels per cell. Raise these
explicitly for larger trusted data. `--limit-frames 1` is useful for a diagnostic
snapshot but labels the output `bounded_prefix`: unread stream completion is not
certified. `--recover` accepts a truncated tail, never checksum corruption, and
labels the result `recovered_prefix`. A failure record or absent summary makes a
table unusable by `track`.

`--workers` parallelizes independent frames on CPUs without changing their
geometry or output order. The pending queue is bounded by the worker count;
memory therefore grows with that count, including each worker's meshes and
distance indices. Start with a small number and reserve sufficient CPU memory.
For larger populations explicitly raise `--max-frame-mib`, for example to 2048.
Set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` when using several workers to avoid
oversubscribing the allocation. Geometry is computed once; subsequent threshold,
persistence and cadence scans use the contact table and need no GPU or meshes.

## Definition and geometry

For cells i and j, measure the area of i's interface whose Euclidean distance
to **any triangle** of j is at most `gap`. The two directed areas need not be
equal. The symmetric score is their minimum; both directed measurements are
retained. This is neither an overlap integral nor a mechanical adhesion/contact
law. Deeply interpenetrating or nested cells can have an interaction graph that
differs from this proximity graph.

Missing interior samples are filled by sign propagation behind the recorded
negative layer, not by zero filling. Every mixed cube is checked to have all
eight original samples. Lewiner marching cubes supplies an explicit ambiguity
convention. Exact 0.5 values are included by lowering the extraction level by
one binary64 ULP; saved float32 values are not perturbed. Near-zero-area
degeneracies are removed. Thus the retained samples are exact, while the mesh
is a specified interpolated representation, not an exact continuum surface.

Cells remain coherent in an unwrapped periodic image. All potentially close
images are tested, with their union used for distance: a surface patch is never
counted twice. There is no z wrapping in a substrate slab or two-wall channel.
Only **computational zero-extension caps** are removed. The real phase field in
steric-wall padding is retained, not cut at the nominal slit plane. Per-cell
diagnostics bound how much surface area lies outside the nominal walls.

An interface with recorded fatal/unknown integrity flags, truncated by its local
support, empty at 0.5, or close to a periodic self-image is unobservable and
censored. The nonfatal support-sizing hint alone is not a loss of observability.
Disconnected, nonmanifold, or unexpectedly
open interfaces are reported; `track` censors them when interpreting graph changes.
An open surface at a known computational cap is allowed. Persistent cell IDs—not height,
colour, or output order—determine identity.

## Area bounds

The distance to a triangle mesh is 1-Lipschitz. For each source triangle, its
centroid distance and enclosing radius therefore bound whether its entire area
is inside or outside the gap. Ambiguous triangles are subdivided, by default
at most twice, stopping sooner at radius `--resolution`. Exact point-to-triangle
queries include faces, edges and vertices; nearest-vertex distance is not used.

Unresolved area contributes only to the upper bound. A bounded refinement
workspace can stop refinement early and sets `resource_limited`; it never turns
an unresolved patch into a definite contact. The reported lower/upper interval
bounds integration error **on the extracted mesh**, subject to floating-point
tolerance. It does not bound isosurface interpolation error, model error or
unsampled temporal events. Tightening refinement cannot substitute for scientific
gap/threshold calibration.

## Persistent changes and topology

A previously absent contact forms only when the symmetric **lower** area bound
stays at or above `area_on` for the requested persistence. A contact is lost only
when the **upper** bound stays at or below `area_off`, with
`0 <= area_off < area_on`. Intermediate observations retain the established
hysteretic state but interrupt a pending confirmation. Initial contacts are a
baseline, not formation events. A missing cell or excessive inter-frame gap
censors observations; re-establishing an unknown state is not a measured change.

Each confirmed change records the interval between the observation preceding
its sustained threshold crossing and the first decisive changed observation,
plus the later confirmation step. Pending changes at observation gaps, missing
cells and the end are explicitly censored. A pending establishment of an unknown
state is labelled separately from a pending contact change.
The output contains counts, not exposure-normalized rates.

Two deliberately limited annotations accompany the measured changes:

- A **four-cell graph-exchange candidate** is a lost edge AB and gained edge CD
  between four distinct cells, close in time, with all four cross-links stable.
  Pairings are provisional, not independently counted T1 events. Simultaneous
  alternatives can make the association ambiguous, including alternatives that
  appear in a later frame. Later conflicting associations name the earlier
  provisional pairings; consumers must reconcile these references before making
  any exclusive pairing count.
- An **ET/TE-compatible graph candidate** is a single-edge gain/loss between two
  cells with a stable common-neighbour triangle. This is compatible with the
  five-cell edge-to-triangle/triangle-to-edge motif, but the graph alone cannot
  establish a shared tricellular line, contact-face geometry or junction collapse.
  Supporting triples and ambiguity are retained; no confirmed 3D topological
  event count is emitted.

## Final tables and geometric inspection

`catalog` reads a completed tracking table, checks its counts and event references,
and reconciles the entire observed history. Both the old and new assignments
become ambiguous if a later candidate competes for the same change. Overlapping
candidate intervals sharing a cell are also marked; using the same cell at
separate times is not itself an ambiguity. It writes:

- `contact_changes.csv`: persistent gains/losses, stable cell IDs, observed
  crossing intervals and confirmation times;
- `exchange_candidates.csv`: four-cell candidates and conflicting assignments;
- `bulk_candidates.csv`: ET/TE-compatible graph annotations;
- `catalog.json`: source hash, settings, reconciled counts and censored endpoints.

These are operational geometric measurements, not confirmed junction-collapse
counts. All times use the absolute simulation clock in units of tau, including
any preparation. A `recovered_prefix` or `bounded_prefix` input needs explicit
`--allow-prefix` and retains that label. Existing output directories are refused.

Inspect candidate cells directly from the recorded interfaces, independently of
the simulator. For example, replace the illustrative IDs/steps below with entries
from the event table:

```sh
python tools/pf3d_export_surfaces.py leg_001.pfs3d leg_002.pfs3d \
  --steps 100000 110000 120000 --ids 7 11 23 31 --anchor-id 7 \
  --max-frame-mib 2048 --output-dir candidate_meshes
```

The binary PLY files open in VTK/ParaView or other mesh viewers. Each file contains
one whole cell, with binary64 coordinates and triangle indices. `manifest.json`
preserves accepted steps, stable IDs, cell parameters, polarity/tumble metadata,
file hashes, removed-cap areas and any unobservable cells. Only a PASS manifest
certifies that every input was read to its clean terminal marker. Requested
steps must exist; restarts are deduplicated only after exact endpoint comparison.
The optional anchor translates whole cells to nearby periodic images for viewing;
it does not deform them, wrap individual vertices or affect contact measurement.
PLY is an inspection export, not a replacement for the exact float32 samples in
the recorded stream. Mesh readers may choose lower internal coordinate precision.

This distinction follows the explicit incidence-graph treatment in
["Graph topological transformations in space-filling cell aggregates"](https://doi.org/10.1371/journal.pcbi.1012089).
Actual 3D transition classification needs junction/face evidence in the relevant
surface sequence, not just a planar neighbour-swap rule applied to a bulk graph.

## Checks before scientific use

The retained tests exercise dense-versus-sparse interfaces, physical axes and
walls, analytic distance/area cases, periodic images, uncertainty, persistent
graph histories, restart deduplication, corruption and censored gaps:

```sh
python tests/3d/surface_contract_test.py
python tests/3d/contact_geometry_test.py
python tests/3d/contact_exchange_test.py
python tests/3d/contact_cli_test.py
```

Alternatively configure CMake with `-DBUILD_TESTING=ON -DPF_ENABLE_CONTACT_TESTS=ON`, then run
`ctest --test-dir build -R '^pf3d_contact_' --output-on-failure`. Analysis
dependencies are optional and do not become simulator build requirements.

Before reporting results, use real full checkpoints to compare surface proximity
with phase-field overlap, inspect contact patches, and test a small grid of gap,
area thresholds and persistence. Compare the same series at several sampling
cadences. Check connected components, cap removal, wall penetration and periodic
seams. Only then select an operational contact definition. Old projected-boundary
files and widely spaced full checkpoints cannot reconstruct missing bulk exchange
histories. The existing long runs need new surface-recording segments to supply
those histories; this postprocessor does not retroactively create them.
