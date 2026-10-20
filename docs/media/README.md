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

The [large-system image](monolayer-large.png) shows 126,025 cells after
0.1 persistence times on one GH200 GPU. All cells have gamma = 1,
radius 49 and passive relaxation mobility 0.5, at nominal area fraction 0.90.
The run starts from the built-in grid-and-jitter placement, without passive
aging. The complete periodic box and its marked detail use the same final
recorded boundaries. Colours are assigned by cell identity.

The [full-resolution image](monolayer-large-full.png) contains the whole box.
The [run record](provenance.json) gives the source version, seeds, timing,
memory use and checks. This is a software-capacity demonstration, separate
from the replicate ensembles used for the paper.

## Movies

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
about 22 seconds long. The GIF shows an eight-second excerpt at the same
playback speed.

The movies were generated for illustration, separately from the replicate
ensembles used for the paper's statistical comparisons. The simulation source
was `0a79433c1ce9edec6406423e51e7cfb15392a777`. The
[single-cell still](single-soft-cell.png) is a frame from the single-cell
movie. The GIF was converted from the binary-mixture MP4.
