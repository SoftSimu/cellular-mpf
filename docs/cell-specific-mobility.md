# Two-dimensional model and cell mobility

For cell i, let `A0_i=pi*R_i^2` and `V_i=integral(phi_i^2 dA)`. With reference
mobility `M0=0.5`, the passive right-hand side and cell velocity are

```text
passive_rhs_i = gamma_i lap(phi_i)
             - (30 gamma_i/lambda^2) phi_i(1-phi_i)(1-2phi_i)
             + (2 mu/A0_i)(A0_i-V_i) phi_i
             - (60 kappa/lambda^2) phi_i sum_(j!=i) phi_j^2,

v_i = v_A,i p_i + (60 kappa/(xi lambda^2))
      integral(phi_i grad(phi_i) sum_(j!=i) phi_j^2 dA).
```

The cell-pair energy uses ordered pairs. Its variation and `M0=0.5` give the
interaction coefficient `60*kappa/lambda^2`; division by `xi` gives the passive
translation coefficient. The nine-point Laplacian uses axial weights 4,
diagonal weights 1 and centre weight -20, divided by 6 on the unit lattice.
See [Palmieri et al., Scientific Reports 5, 11745 (2015)](https://doi.org/10.1038/srep11745).

## Mobility

The 2D solver stores a physical Allen-Cahn mobility `M_i` for every cell.
The reference and default are `M0=0.5`. Its update is

```text
dphi_i/dt = (M_i/M0) * passive_rhs_i - v_i . grad(phi_i).
```

`passive_rhs_i` includes the interfacial Laplacian, double-well, area-constraint
and overlap-repulsion terms. Mobility never scales advection, the interaction
velocity, active speed, polarity or tumble stream. Default-mobility cells use
the default arithmetic expression without an extra multiply by one.
Changing `M_i` to compensate `gamma_i` also changes area and overlap relaxation;
it is not a surface-tension-only adjustment.

## Fresh starts

`--phase-field-mobility M` sets the uniform mobility, default `0.5`.
`--phase-field-mobility-map PATH` applies overrides keyed by stable global ID.
Unlisted cells keep the uniform value. For example, save this as `mobility.txt`:

```text
# global_id mobility
7 0.75
12 0.25
```

```bash
./build/cell_gh200 --N 100 --t-end 10 \
  --phase-field-mobility 0.5 --phase-field-mobility-map mobility.txt \
  --out trajectory.txt --checkpoint-dir checkpoints
```

Maps accept blank lines and whole-line `#` comments. Each other line must have
exactly one integer ID and one mobility value. Malformed rows, duplicate or
unknown IDs, empty maps, and values that are not finite and positive in
binary32 are rejected. IDs are not row numbers in an external centre table.

The explicit-Euler guard checks the resolved maximum
`max_i[(M_i/M0)*gamma_i]`; a map cannot bypass that check. Increased area and
overlap relaxation also produce timestep-margin warnings. The solver never
changes `dt` automatically; convergence remains the user's responsibility.

## Checkpoints and restart precedence

Current checkpoint format **1** retains its fixed tile and parameter layouts.
An optional `MOBI` sidecar contains one float32 mobility per cell. New writers
always emit it, including when every value is `0.5`. Current-format files
without `MOBI` load with all mobilities set to `0.5`.

After loading, restart options have exactly three interpretations:

1. An explicit uniform mobility rebases every cell; map entries then override.
2. A map alone overlays its listed IDs on stored values, leaving all other
   cells untouched. The parameter record does not store the writing run's
   uniform mobility, so it must not be used to rebase unlisted cells.
3. With neither option supplied, stored values are preserved, or the all-`0.5`
   default when the input has no `MOBI`.

## Outputs and dimensional scope

Boundary metadata and velocity-component rows record each cell's actual
mobility. Passive velocity components include its `M_i/M0` multiplier;
advection components do not. Plain-text trajectory columns are unchanged and
do not record mobility: retain the input map and checkpoints with the run.

Mobility maps apply only to `cell_gh200`. Exporting relaxed 2D centres creates
positions for fresh 3D cells, not a transfer of mobility or phase-field state.
The exporter's provenance records checkpoint mobility metadata when available.
See [boundary output](boundary-output.md) and [velocity output](velocity-output.md).
