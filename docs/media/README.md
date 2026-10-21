# Simulation images and movies

## Monolayer and slab

The [overview image](simulation-overview.png) shows two saved simulation states:

- **[6,400-cell monolayer](monolayer-6400.png).** The full periodic square,
  reconstructed from a saved phase-field checkpoint.
- **[500-cell slab](slab-500.png).** A five-layer simulation confined between
  two steric walls, viewed from above and from the side. The wall outlines mark
  the confinement. The side faces are cuts through the periodic simulation box.

Colours distinguish cell identities. They do not represent stiffness, speed or
another measured quantity. Both images show the `phi=0.5` interfaces. The 3D
surface tessellation is reduced for rendering, without smoothing the geometry.
The [source and rendering record](provenance.json) identifies the saved states.

## Large monolayer

The [large-system image](monolayer-large.png) shows 126,025 cells on one GH200.
Initial centres are placed randomly with a minimum periodic separation of one
cell radius. Passive relaxation lasts 0.1 persistence times, followed by
0.1 persistence times of active motion. All cells have gamma = 1 and mobility 0.5.

The [full-resolution image](monolayer-large-full.png) and marked detail show
the same final recorded boundaries. Colours distinguish cell identities.
The [run record](provenance.json) gives parameters and checks. This demonstration
is separate from the replicate ensembles used for the paper.

## Kinetic comparison

The [README preview](kinetic-control.gif) compares a reference monolayer (left)
with slower passive relaxation in 30% of the cells (right). Each has 400 cells
with interfacial coefficient γ = 1 and ξ = 375. Purple marks the selected cells
in both panels. Their mobility is M = 0.5 in the reference and M = 0.175 on
the right; the other cells retain M = 0.5.

The [HD preview](kinetic-control.mp4) shows active ages 10–18τ at 25 fps,
using original frames spaced by 0.04τ. The [full comparison](kinetic-control-full.mp4)
shows 50τ at 20 fps. The preview plays at 1τ per second; the full movie plays
at 2.5τ per second.
[Recording and rendering details](provenance.json).

## Soft-cell examples

These illustrations show cell shapes reconstructed from saved phase fields.
Each simulation contains 400 cells in a periodic square and shows 50 persistence
times of active motion.

- **[Single soft cell](single-soft-cell.mp4).** The teal cell has interfacial
  coefficient γ = 1 in the all-normal reference on the left and γ = 0.35 on the
  right. All other cells have γ = 1. Its trail shows the preceding 8 persistence
  times. The view stays fixed throughout.
- **[Binary mixtures](binary-mixtures.mp4).** Soft cells (teal, γ = 0.35) make up
  5% of the monolayer on the left and 50% on the right. Normal cells are red
  (γ = 1).

Both pairs start from the same equilibrated all-normal configuration before
the soft-cell coefficients are assigned. Passive preparation lasts one
persistence time, followed by active motion with propulsion speed 0.01.
The scale bars show five nominal cell radii. Saved frames are 0.125 persistence
times apart in both movies. Each MP4 is 1920 × 1080, 20 frames per second and
about 22 seconds long.

The movies were generated for illustration, separately from the replicate
ensembles used for the paper's statistical comparisons. The simulation source
was `0a79433c1ce9edec6406423e51e7cfb15392a777`. The
[single-cell still](single-soft-cell.png) is a frame from the single-cell
movie. The GIF was converted from the binary-mixture MP4.
