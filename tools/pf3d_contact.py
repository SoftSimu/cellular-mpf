"""Geometric proximity patches on recorded three-dimensional cell interfaces.

This is an observation definition, not an adhesion law. Gap and area thresholds
must be calibrated for the model and sampling cadence before reporting rates.
Distances are to triangles, not their vertices. Patch areas include explicit
lower/upper bounds for unresolved triangles of the reconstructed mesh.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from itertools import product
import math

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

from pf3d_surface import Cell, Header, SurfaceFormatError


class UnobservableInterface(ValueError):
    """Geometry is incomplete; censor this cell, not the whole stream."""


@dataclass(frozen=True)
class Mesh:
    id: int
    vertices: np.ndarray
    faces: np.ndarray
    half_level_ties: int = 0
    removed_cap_area: float = 0.0
    wall_area_lower: float = 0.0
    wall_area_upper: float = 0.0

    @cached_property
    def centroids(self):
        return self.vertices[self.faces].mean(axis=1)

    @cached_property
    def areas(self):
        tri = self.vertices[self.faces]
        return 0.5 * np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)

    @cached_property
    def radii(self):
        return np.linalg.norm(self.vertices[self.faces]-self.centroids[:, None, :], axis=2).max(axis=1)

    @cached_property
    def bounds(self):
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    @cached_property
    def index(self):
        return TriangleIndex(self)


def _finish_mesh(gid, vertices, faces, ties=0, area_floor=0.0, **diagnostics):
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    if len(faces):
        triangles = vertices[faces]
        area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                       triangles[:, 2]-triangles[:, 0]), axis=1) * 0.5
        faces = faces[area > area_floor]
    if len(faces):
        used, inverse = np.unique(faces, return_inverse=True)
        vertices, faces = vertices[used], inverse.reshape(-1, 3).astype(np.int32)
    else:
        vertices = np.empty((0, 3), dtype=np.float64)
    return Mesh(gid, vertices, faces, ties, **diagnostics)


def dense_interface_field(cell: Cell, *, max_voxels=32_000_000):
    """Fill unknown *signs*, preserving all measured corner values verbatim.

    A missing vertex has the same class as all 26 neighbours, otherwise the
    recorder would retain it. Positive seeds therefore cannot propagate past
    the recorded negative layer. No unknown value can affect a crossing cube.
    """
    side = cell.edge + 2
    if side**3 > max_voxels:
        raise ValueError('cell exceeds the explicit dense-reconstruction limit')
    indices, values = cell.arrays()
    shape = (side, side, side)  # z,y,x; file indices are x-fastest
    known = np.zeros(shape, dtype=bool)
    negative = np.zeros(shape, dtype=bool)
    positive = np.zeros(shape, dtype=bool)
    known.ravel()[indices] = True
    negative.ravel()[indices] = values < np.float32(0.5)
    positive.ravel()[indices] = values >= np.float32(0.5)
    # The observer's padded shell is explicitly zero, not a measured free face.
    shell = np.zeros(shape, dtype=bool)
    shell[[0, -1], :, :] = True
    shell[:, [0, -1], :] = True
    shell[:, :, [0, -1]] = True
    if np.any(positive & shell):
        raise SurfaceFormatError('positive value in the declared zero shell')
    inside = ndimage.binary_propagation(positive, structure=np.ones((3, 3, 3)),
                                       mask=~(negative | shell))
    base = inside[:-1, :-1, :-1]
    mixed = np.zeros_like(base)
    complete = np.ones_like(base)
    for dz, dy, dx in product((0, 1), repeat=3):
        region = np.s_[dz:dz+side-1, dy:dy+side-1, dx:dx+side-1]
        mixed |= base != inside[region]
        complete &= known[region]
    if np.any(mixed & ~complete):
        raise SurfaceFormatError('interface reconstruction needs an unrecorded cube corner')
    dense = inside.astype(np.float32)
    dense.ravel()[indices] = values
    return dense


def _check_support(dense, cell, header):
    inside = dense >= np.float32(0.5)
    if (np.any(inside[:, :, 1]) or np.any(inside[:, :, -2])
            or np.any(inside[:, 1, :]) or np.any(inside[:, -2, :])):
        raise UnobservableInterface(f'cell {cell.id}: interface reaches a lateral support edge')
    for local, world in ((1, cell.origin[2]), (-2, cell.origin[2]+cell.edge-1)):
        if np.any(inside[local]):
            if header.boundary_flags == 7 or world not in (0, header.nz-1):
                raise UnobservableInterface(f'cell {cell.id}: interface reaches an interior z support edge')


def mesh_from_surface(cell: Cell, header: Header, *, max_voxels=32_000_000):
    # Version-1 CellState flags: bit 6 is a support-sizing hint, not lost data.
    # Other (including unknown) flags must not produce geometric events.
    if cell.flags & ~(1 << 6):
        raise UnobservableInterface(f'cell {cell.id}: recorded integrity flags {cell.flags:#x}')
    dense = dense_interface_field(cell, max_voxels=max_voxels)
    _check_support(dense, cell, header)
    ties = int(np.count_nonzero(dense == np.float32(0.5)))
    if dense.max() < 0.5:
        return _finish_mesh(cell.id, [], [], ties)
    # A binary64-ULP level shift includes exact float32 0.5 without altering
    # any recorded value or the classification of a smaller float32 sample.
    level = float(np.nextafter(0.5, 0.0)) if ties else 0.5
    vertices, faces, _, _ = marching_cubes(dense, level=level,
        spacing=(header.dz, header.dy, header.dx), method='lewiner', allow_degenerate=False)
    vertices = vertices[:, ::-1].astype(np.float64)
    cap_area = 0.0
    if header.boundary_flags != 7 and len(faces):
        z = vertices[faces, 2] / header.dz + cell.origin[2] - 1
        # Only triangles in computational zero-extension cubes are artificial.
        # Steric-wall padding contains real fields and is not clipped away.
        caps = np.all(z <= 0, axis=1) | np.all(z >= header.nz-1, axis=1)
        temporary = Mesh(cell.id, vertices, faces)
        cap_area = float(temporary.areas[caps].sum())
        faces = faces[~caps]
    periodic = (True, True, header.boundary_flags == 7)
    dimensions = (header.nx, header.ny, header.nz)
    origin = [value % size if wrap else value
              for value, size, wrap in zip(cell.origin, dimensions, periodic)]
    vertices += (np.asarray(origin, dtype=np.float64)-1) * (header.dx, header.dy, header.dz)
    if header.boundary_flags == 11:
        vertices[:, 2] -= (header.channel_padding-0.5) * header.dz
    floor = (min(header.dx, header.dy, header.dz) * 1e-6)**2
    mesh = _finish_mesh(cell.id, vertices, faces, ties, floor, removed_cap_area=cap_area)
    if header.boundary_flags == 11 and len(mesh.faces):
        z = mesh.vertices[mesh.faces, 2]
        height = header.channel_height * header.dz
        entirely = np.all(z < 0, axis=1) | np.all(z > height, axis=1)
        partly = np.any((z < 0) | (z > height), axis=1)
        mesh = Mesh(mesh.id, mesh.vertices, mesh.faces, ties, cap_area,
                    float(mesh.areas[entirely].sum()), float(mesh.areas[partly].sum()))
    return mesh


def point_triangle_distance(points, triangles):
    """Euclidean distances for corresponding point/triangle rows, including edges."""
    p = np.asarray(points, dtype=np.float64)
    a, b, c = np.moveaxis(np.asarray(triangles, dtype=np.float64), 1, 0)
    ab, ac, ap = b-a, c-a, p-a
    normal = np.cross(ab, ac)
    norm2 = np.einsum('ij,ij->i', normal, normal)
    if np.any(norm2 <= 0):
        raise ValueError('distance query contains a degenerate triangle')
    height = np.einsum('ij,ij->i', ap, normal)
    v = np.einsum('ij,ij->i', np.cross(ap, ac), normal) / norm2
    w = np.einsum('ij,ij->i', np.cross(ab, ap), normal) / norm2
    interior = (v >= 0) & (w >= 0) & (v+w <= 1)
    squared = np.where(interior, height*height/norm2, np.inf)
    for start, end in ((a, b), (b, c), (c, a)):
        edge, relative = end-start, p-start
        scale = np.einsum('ij,ij->i', relative, edge) / np.einsum('ij,ij->i', edge, edge)
        residual = relative-np.clip(scale, 0, 1)[:, None]*edge
        squared = np.minimum(squared, np.einsum('ij,ij->i', residual, residual))
    return np.sqrt(np.maximum(squared, 0))


class TriangleIndex:
    def __init__(self, mesh):
        if not len(mesh.faces):
            raise ValueError('cannot index an empty interface')
        self.mesh = mesh
        self.tree = cKDTree(mesh.centroids)
        self.max_radius = float(mesh.radii.max())

    def bounds(self, points):
        distances, ids = self.tree.query(points)
        upper = point_triangle_distance(points, self.mesh.vertices[self.mesh.faces[ids]])
        return np.maximum(0, distances-self.max_radius), upper

    def exact_within(self, points, limits, *, max_candidates=1_000_000):
        """Exact below each limit; infinity may represent an unneeded farther hit."""
        result = np.full(len(points), np.inf)
        for start in range(0, len(points), 256):
            p = points[start:start+256]
            bound = np.asarray(limits)[start:start+256]
            _, upper = self.bounds(p)
            neighbours = self.tree.query_ball_point(p, np.minimum(bound, upper)+self.max_radius+1e-10)
            counts = np.fromiter((len(row) for row in neighbours), dtype=np.int64, count=len(p))
            if counts.sum() > max_candidates:
                raise ValueError('triangle query exceeds memory limit; reduce the gap or analyze smaller batches')
            rows = np.repeat(np.arange(len(p)), counts)
            ids = np.fromiter((item for row in neighbours for item in row), dtype=np.int64)
            if len(ids):
                distances = point_triangle_distance(p[rows], self.mesh.vertices[self.mesh.faces[ids]])
                np.minimum.at(result[start:start+len(p)], rows, distances)
        return result


def image_shifts(a: Mesh, b: Mesh, header: Header, gap):
    """All potentially contacting images, not a centre-only image approximation."""
    choices = []
    amin, amax = a.bounds
    bmin, bmax = b.bounds
    lengths = np.array((header.nx*header.dx, header.ny*header.dy, header.nz*header.dz))
    for axis, periodic in enumerate((True, True, header.boundary_flags == 7)):
        if periodic:
            low = math.ceil((amin[axis]-gap-bmax[axis]) / lengths[axis])
            high = math.floor((amax[axis]+gap-bmin[axis]) / lengths[axis])
            choices.append(range(low, high+1))
        else:
            choices.append((0,))
    if math.prod(len(c) for c in choices) > 125:
        raise ValueError('too many periodic images for a local contact measurement')
    result = []
    for image in product(*choices):
        shift = np.asarray(image)*lengths
        separation = np.maximum(np.maximum(amin-bmax-shift, bmin+shift-amax), 0)
        if np.linalg.norm(separation) <= gap:
            result.append(shift)
    return result


@dataclass(frozen=True)
class AreaBounds:
    lower: float
    upper: float
    unresolved_triangles: int
    resource_limited: bool = False

    @property
    def estimate(self):
        return (self.lower+self.upper)*0.5


def proximity_area(source: Mesh, target: Mesh, shifts, gap, resolution, max_depth=2,
                   *, max_triangles=250_000):
    """Area bounds on this mesh from the 1-Lipschitz distance function.

    Images are queried as a union: a source patch is never counted twice.
    Unresolved leaf area widens the bounds instead of silently voting in/out.
    """
    validate_measurement(gap, resolution, max_depth)
    if not isinstance(max_triangles, int) or max_triangles <= 0:
        raise ValueError('max_triangles must be a positive integer')
    if not shifts or not len(source.faces) or not len(target.faces):
        return AreaBounds(0.0, 0.0, 0)
    # AABB lower bounds cheaply discard most non-facing surface triangles.
    possible = np.zeros(len(source.faces), dtype=bool)
    tmin, tmax = target.bounds
    for shift in shifts:
        delta = np.maximum(np.maximum(tmin+shift-source.centroids,
                                      source.centroids-tmax-shift), 0)
        possible |= np.linalg.norm(delta, axis=1) <= gap+source.radii
    triangles = source.vertices[source.faces[possible]]
    lower = unresolved_area = 0.0
    unresolved_count = 0
    resource_limited = False
    for depth in range(max_depth+1):
        if not len(triangles):
            break
        centres = triangles.mean(axis=1)
        radii = np.linalg.norm(triangles-centres[:, None, :], axis=2).max(axis=1)
        areas = 0.5*np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                           triangles[:, 2]-triangles[:, 0]), axis=1)
        lo, hi = np.full(len(triangles), np.inf), np.full(len(triangles), np.inf)
        for shift in shifts:
            a, b = target.index.bounds(centres-shift)
            lo, hi = np.minimum(lo, a), np.minimum(hi, b)
        scale = max(1.0, gap, float(np.abs(centres).max()))
        epsilon = 64 * np.finfo(np.float64).eps * scale
        near, far = hi+radii+epsilon <= gap, lo-radii-epsilon > gap
        undecided = ~(near | far)
        if np.any(undecided):
            distances = np.full(np.count_nonzero(undecided), np.inf)
            for shift in shifts:
                distances = np.minimum(distances, target.index.exact_within(
                    centres[undecided]-shift, gap+radii[undecided]+epsilon))
            near[undecided] = distances+radii[undecided]+epsilon <= gap
            far[undecided] = distances-radii[undecided]-epsilon > gap
        lower += float(areas[near].sum())
        remaining = ~(near | far)
        stop = remaining & ((radii <= resolution) | (depth == max_depth))
        unresolved_area += float(areas[stop].sum())
        unresolved_count += int(stop.sum())
        t = triangles[remaining & ~stop]
        if len(t)*4 > max_triangles:
            unresolved_area += float(areas[remaining & ~stop].sum())
            unresolved_count += len(t)
            resource_limited = True
            break
        ab, bc, ca = (t[:, 0]+t[:, 1])*0.5, (t[:, 1]+t[:, 2])*0.5, (t[:, 2]+t[:, 0])*0.5
        triangles = np.concatenate((np.stack((t[:, 0], ab, ca), axis=1),
            np.stack((ab, t[:, 1], bc), axis=1), np.stack((ca, bc, t[:, 2]), axis=1),
            np.stack((ab, bc, ca), axis=1)), axis=0)
    return AreaBounds(lower, lower+unresolved_area, unresolved_count, resource_limited)


@dataclass(frozen=True)
class Contact:
    a: int
    b: int
    a_near_b: AreaBounds
    b_near_a: AreaBounds

    @property
    def lower(self):
        return min(self.a_near_b.lower, self.b_near_a.lower)

    @property
    def upper(self):
        return min(self.a_near_b.upper, self.b_near_a.upper)


def validate_measurement(gap, resolution, max_depth):
    if (not math.isfinite(gap) or not math.isfinite(resolution)
            or gap <= 0 or resolution <= 0
            or not isinstance(max_depth, int) or isinstance(max_depth, bool)
            or not 0 <= max_depth <= 4):
        raise ValueError('gap/resolution must be finite and positive; max_depth must be in [0,4]')


def frame_contacts(meshes, header, *, gap, resolution=0.25, max_depth=2):
    meshes = sorted(meshes, key=lambda mesh: mesh.id)
    if len({m.id for m in meshes}) != len(meshes):
        raise ValueError('duplicated persistent cell ID')
    validate_measurement(gap, resolution, max_depth)
    if any(not len(m.faces) for m in meshes):
        raise ValueError('empty interface: contact changes involving an unobservable cell cannot be inferred')
    result = []
    for i, a in enumerate(meshes):
        for b in meshes[i+1:]:
            shifts = image_shifts(a, b, header, gap)
            if not shifts:
                continue
            ab = proximity_area(a, b, shifts, gap, resolution, max_depth)
            ba = proximity_area(b, a, [-shift for shift in shifts], gap, resolution, max_depth)
            if ab.upper > 0 and ba.upper > 0:
                result.append(Contact(a.id, b.id, ab, ba))
    return result
