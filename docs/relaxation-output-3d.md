# Spatial relaxation measurements in substrate layers

`--relaxation-out segment.pfr3d --relaxation-interval 100` enables an independent, read-only observation of the current phase fields. It also enables the accepted-step in-plane propulsion and advection integrals. It is supported for substrate slabs with free, neutral, or zero-field upper boundaries. It does not cover periodic 3D motion or the repulsive-wall channel, whose additional force would need a separate contribution.

For each cell, the recorder measures its volume `V = sum(phi²)` and unwrapped volume centroid `C`. The contribution of a free-energy derivative `g` to the centroid velocity is

    -2 M / V * sum((r - C) phi g).

The interfacial derivative is `-2 gamma Laplacian(phi) + 60 gamma/lambda² phi(1-phi)(1-2phi)`. The overlap derivative is `120 kappa/lambda² phi sum_{j!=i}(phi_j²)`. The volume-constraint derivative is a cell-wide multiple of `phi`, so its instantaneous centroid moment is zero. Volume regulation still changes subsequent shapes and interactions.

The observer uses the current stored float fields, the solver's quantized aggregate, its 27-point Laplacian and the same boundary reflections. The spatial sums and derivative evaluation use double precision. It computes its own volume and centroid; cached solver moments can describe the preceding step. It never refreshes solver measurements, draws random numbers, or alters fields, velocities, or recentering. On two GPUs it reads each cell on its owner, without gathering phase fields. The current launch supports at most 65,535 owned cells on each device.

Centroid coordinates match the ordinary trajectory: unwrapped x and y are lattice coordinates, and z is height above the substrate. The substrate lies half a voxel below the first cell-centred plane. Surface exporters may use a cell-corner convention; account for that offset when overlaying coordinates rather than changing the displacement definition.

## Sampling and interpretation

Frames contain the initial state, every accepted step divisible by the requested interval, and the final state. Choose a new filename on each restart. The checkpointed propulsion and actual-advection integrals retain their previous zero point; take differences between endpoints. Spatial rates have no checkpoint accumulators. Join segments only after verifying a common boundary state and identical parameters and cell identities.

Enabling this output also enables the existing `record_planar_motion` checkpoint flag and its four integrals. That flag persists after restart, even if spatial output is then disabled. Trajectories consequently gain four integral columns. Use a new trajectory file if the earlier file lacked those columns; appending an incompatible schema is rejected. These output-state changes do not alter the physical trajectory or checkpoint format.

To check quadrature during a long run, add `--relaxation-dense-every D`, where `D` is a positive multiple of the regular interval `S`. The first `S` steps of each absolute `D`-step period are then sampled every step, including both endpoints. For example, `S=1000, D=1000000` gives one dense 1,000-step window per million steps. Compare its discrete sum with the estimate using only the window endpoints. The schedule is recorded in the checksummed metadata and follows the same absolute clock through restarts. These short windows test the sparse integration; they do not change the timestep or equations.

`python tools/pf3d_relaxation.py segment.pfr3d` checks the stream and reports an integrated per-cell displacement budget. The integration targets the explicit solver's pre-step sum: for `S` steps, a linearly interpolated rate contributes `dt * [S*g_start + (S-1)/2*(g_end-g_start)]`. At interval one this is the exact sum of observed pre-step rates. For a larger interval, validate convergence against finer sampling on representative evolving states.

The central advection stencil contributes a factor `B_a = sum(phi(r) phi(r+e_a))/V` along each in-plane axis. The reader reports `(B_a-1)` times the recorded advection increment, holding `B_a` at the interval's start. Propulsion is integrated on every accepted step, including tumbles; it is not reconstructed from sampled polarity angles. The remaining closure residual includes finite-Euler changes in the centroid normalization, float storage/evaluation, and sparse quadrature error. It is reported separately, not assigned to shape relaxation.

## Binary layout

All records are little-endian. `common/relaxation_format_3d.h` defines the layouts, and `tools/pf3d_relaxation.py` provides the reader.

- An 80-byte `PF3DRLX1` header gives population, initial step, timestep, cadence, box, boundary flags, and metadata length, followed by UTF-8 model/reduction metadata. CRC-32 covers both.
- Each 32-byte `P3RLFRM1` frame header gives step, time, population, and CRC-32 covering header and rows. Each 152-byte row stores an int64 cell ID and 18 doubles: centroid xyz, volume, effective mobility, gamma, propulsion-integral xy, advection-integral xy, interfacial centroid-rate xyz, overlap centroid-rate xyz, and advection factors xy.
- A 24-byte `P3RLEND1` record confirms the number of frames and final step. Interrupted or failed writes without this record are not accepted as complete segments.

Rows are sorted by persistent cell ID. The writer refuses existing files and conflicting output paths. A restart does not append to or overwrite an earlier segment. The reader rejects truncated records, checksum errors, identity changes, and missing sampling-grid frames.
