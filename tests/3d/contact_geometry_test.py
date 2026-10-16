"""Analytic geometry and dense-field oracles for surface contact measurements."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np
from skimage.measure import marching_cubes

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import pf3d_contact as contact
import pf3d_surface as surface
from surface_contract_test import dense_oracle


def header(flags=7, size=96):
    return surface.Header(2, flags, 0, size, size, size if flags == 7 else 16,
        1., 1., 1., .01, 10000., .5, 10000, 10 if flags == 11 else 0,
        3 if flags == 11 else 0, 10. if flags == 11 else 0.,
        1. if flags == 11 else 0., 1.)


def cell_from_dense(phi, gid=11, origin=(0, 0, 0)):
    _, _, indices = dense_oracle(phi)
    data = b''.join(surface.VERTEX.pack(int(i), float(phi.ravel()[i])) for i in indices)
    return surface.Cell(gid, origin, len(phi)-2, 1., .01, 4.,
                        (1., 0., 0.), (0., 0., 0.), 0, 0, data)


def sphere_phi(edge=16, centre=(8.3, 8.1, 8.2), radius=4.):
    z, y, x = np.mgrid[:edge, :edge, :edge]
    distance = np.sqrt((x-centre[0])**2+(y-centre[1])**2+(z-centre[2])**2)
    return np.pad(np.clip(.5+.2*(radius-distance), 0, 1).astype(np.float32), 1)


def square(gid, *, z=0., x=0., width=2., y=0., height=2.):
    vertices = np.array(((x,y,z), (x+width,y,z),
                         (x+width,y+height,z), (x,y+height,z)))
    return contact.Mesh(gid, vertices, np.array(((0,1,2),(0,2,3)), dtype=np.int32))


class ContactGeometry(unittest.TestCase):
    def test_dense_reconstruction_preserves_every_crossing_cube(self):
        fields = [sphere_phi()]
        hollow = sphere_phi()
        hollow[7:10, 7:10, 7:10] = 0
        fields.append(hollow)
        fields.append(np.pad((np.indices((8,8,8)).sum(axis=0) % 2).astype(np.float32), 1))
        for phi in fields:
            cell = cell_from_dense(phi)
            restored = contact.dense_interface_field(cell)
            original, values, _ = dense_oracle(phi)
            lower, restored_values, _ = dense_oracle(restored)
            np.testing.assert_array_equal(original, lower)
            np.testing.assert_array_equal(values.view('u4'), restored_values.view('u4'))

    def test_surface_mesh_matches_direct_dense_lewiner(self):
        phi = sphere_phi()
        mesh = contact.mesh_from_surface(cell_from_dense(phi), header())
        vertices, faces, _, _ = marching_cubes(phi, .5, allow_degenerate=False)
        np.testing.assert_array_equal(mesh.vertices, vertices[:, ::-1]-1)
        np.testing.assert_array_equal(mesh.faces, faces)
        self.assertLess(abs(mesh.areas.sum()-4*np.pi*4**2)/(4*np.pi*4**2), .04)

    def test_periodic_translation_does_not_lose_large_integer_origins(self):
        cell = cell_from_dense(sphere_phi(), origin=(3,4,5))
        shifted = replace(cell, origin=tuple(v+96*2**54 for v in cell.origin))
        a = contact.mesh_from_surface(cell, header())
        b = contact.mesh_from_surface(shifted, header())
        np.testing.assert_array_equal(a.vertices, b.vertices)

    def test_substrate_cap_is_removed_without_creating_a_new_face(self):
        phi = sphere_phi(centre=(8.3,8.1,8.), radius=4.)
        origin = (10,10,-8)
        world_z = np.arange(18)-1+origin[2]
        phi[world_z < 0] = 0
        mesh = contact.mesh_from_surface(cell_from_dense(phi, origin=origin), header(3))
        self.assertTrue(np.all(mesh.vertices[:, 2] >= 0))
        flat = np.all(np.abs(mesh.vertices[mesh.faces, 2]) < 1e-10, axis=1)
        self.assertFalse(np.any(flat))
        self.assertLess(abs(mesh.areas.sum()-2*np.pi*4**2)/(2*np.pi*4**2), .08)

    def test_channel_keeps_real_fields_in_steric_wall_padding(self):
        phi = sphere_phi(centre=(8.3,8.1,3.), radius=4.)
        phi[1] = 0  # keep the synthetic support away from the array exterior
        mesh = contact.mesh_from_surface(cell_from_dense(phi), header(11))
        self.assertLess(mesh.vertices[:,2].min(), 0.)
        self.assertLessEqual(mesh.vertices[:,2].max(), 10.)
        self.assertFalse(np.any(np.all(mesh.vertices[mesh.faces,2] == 0, axis=1)))
        self.assertGreater(mesh.wall_area_lower, 0.)
        self.assertGreaterEqual(mesh.wall_area_upper, mesh.wall_area_lower)

    def test_upper_computational_cap_is_not_a_free_interface(self):
        phi = sphere_phi(centre=(8.,8.,14.),radius=4.)
        mesh = contact.mesh_from_surface(cell_from_dense(phi),header(3))
        self.assertGreater(mesh.removed_cap_area,0.)
        self.assertLessEqual(mesh.vertices[:,2].max(),15.)
        self.assertFalse(np.any(np.all(mesh.vertices[mesh.faces,2] == 15.,axis=1)))

    def test_support_truncation_is_not_reported_as_a_contact_face(self):
        phi = sphere_phi(centre=(0.,8.,8.), radius=4.)
        with self.assertRaisesRegex(contact.UnobservableInterface, 'support edge'):
            contact.mesh_from_surface(cell_from_dense(phi), header())

    def test_point_to_triangle_interior_edges_and_vertices(self):
        triangle = np.array(((0.,0.,0.),(1.,0.,0.),(0.,1.,0.)))
        points = np.array(((.2,.2,2.),(2.,0.,0.),(.5,.5,0.),(-1.,-1.,0.)))
        actual = contact.point_triangle_distance(points, np.repeat(triangle[None], 4, axis=0))
        np.testing.assert_allclose(actual, (2.,1.,0.,np.sqrt(2)), rtol=0, atol=1e-14)

    def test_parallel_patch_area_and_separation(self):
        a = square(1, z=.2)
        b = square(2, z=0., x=-5., y=-5., width=12., height=12.)
        measured = contact.proximity_area(a,b,[np.zeros(3)],.5,.01,4)
        self.assertAlmostEqual(measured.lower, 4.)
        self.assertAlmostEqual(measured.upper, 4.)
        outside = contact.proximity_area(a,b,[np.zeros(3)],.1,.01,4)
        self.assertAlmostEqual(outside.upper, 0.)

    def test_vertex_only_distance_would_miss_this_contact(self):
        target = square(2, width=100., height=100.)
        points = np.array(((40.,40.,.25),))
        distance = target.index.exact_within(points, np.array((1.,)))
        np.testing.assert_allclose(distance, (.25,), atol=1e-13)

    def test_periodic_seam_contact_and_no_z_wrapping(self):
        a, b = square(1,z=.25), square(2,z=95.75)
        periodic = contact.frame_contacts([a,b],header(),gap=.7,resolution=.01,max_depth=4)
        self.assertEqual(len(periodic),1)
        self.assertAlmostEqual(periodic[0].lower,4.)
        self.assertEqual(contact.frame_contacts([a,b],header(11),gap=.7),[])

    def test_images_are_a_union_not_repeated_area(self):
        a = square(1,width=3.,height=.2)
        b = square(2,z=.1,width=2.1,height=.2)
        h = header(size=2)
        shifts = contact.image_shifts(a,b,h,.5)
        self.assertGreater(len(shifts),1)
        area = contact.proximity_area(a,b,shifts,.5,.01,4)
        self.assertLessEqual(area.upper,.6+1e-10)
        self.assertGreater(area.lower,.55)

    def test_unresolved_area_is_reported_not_forced(self):
        a, b = square(1,z=.5),square(2)
        area = contact.proximity_area(a,b,[np.zeros(3)],.5,.25,2)
        self.assertEqual(area.lower,0.)
        self.assertAlmostEqual(area.upper,4.)
        self.assertGreater(area.unresolved_triangles,0)

    def test_exact_half_level_is_not_a_finite_area_isolated_cell(self):
        phi = np.zeros((10,10,10), dtype=np.float32)
        phi[5,5,5] = .5
        mesh = contact.mesh_from_surface(cell_from_dense(phi), header())
        self.assertEqual(mesh.half_level_ties, 1)
        self.assertEqual(len(mesh.faces), 0)

    def test_half_level_ties_use_inclusive_lewiner_topology(self):
        phi = np.zeros((10,10,10),dtype=np.float32)
        phi[4:6,4:6,4:6] = [[[0.,1.],[.5,.5]],[[.5,1.],[0.,1.]]]
        inclusive = marching_cubes(phi,float(np.nextafter(.5,0.)),allow_degenerate=False)
        exclusive = marching_cubes(phi,.5,allow_degenerate=False)
        self.assertNotEqual(len(inclusive[1]),len(exclusive[1]))
        mesh = contact.mesh_from_surface(cell_from_dense(phi),header())
        np.testing.assert_array_equal(mesh.vertices,inclusive[0][:,::-1]-1)
        np.testing.assert_array_equal(mesh.faces,inclusive[1])

    def test_unequal_axes_and_spacing_are_not_swapped(self):
        z,y,x = np.mgrid[:16,:16,:16]
        phi = np.pad(np.clip(.5+.2*(1-np.sqrt(((x-8)/5)**2+
                     ((y-8)/3)**2+((z-8)/2)**2)),0,1).astype('f4'),1)
        mesh = contact.mesh_from_surface(cell_from_dense(phi), replace(header(),dx=2.,dy=3.,dz=4.))
        np.testing.assert_allclose(np.ptp(mesh.vertices,axis=0),(20,18,16),atol=1e-5)

    def test_refinement_budget_widens_uncertainty_instead_of_allocating(self):
        a,b = square(1,z=.5),square(2)
        result = contact.proximity_area(a,b,[np.zeros(3)],.5,.001,4,max_triangles=1)
        self.assertTrue(result.resource_limited)
        self.assertEqual((result.lower,result.upper),(0.,4.))

    def test_invalid_options_are_rejected_even_without_candidate_pairs(self):
        for options in (dict(resolution=float('nan')),dict(max_depth=5),dict(gap=float('inf'))):
            args = dict(gap=1.)
            args.update(options)
            with self.assertRaises(ValueError):
                contact.frame_contacts([],header(),**args)

    def test_strip_area_is_bracketed_and_refinement_reduces_uncertainty(self):
        source = square(1)
        vertices = np.array(((0,-5,-5),(0,5,-5),(0,5,5),(0,-5,5)),dtype=float)
        target = contact.Mesh(2,vertices,np.array(((0,1,2),(0,2,3))))
        widths = []
        for depth in range(5):
            measured = contact.proximity_area(source,target,[np.zeros(3)],.75,.001,depth)
            self.assertLessEqual(measured.lower,1.5)
            self.assertGreaterEqual(measured.upper,1.5)
            widths.append(measured.upper-measured.lower)
        self.assertTrue(all(b <= a+1e-12 for a,b in zip(widths,widths[1:])))
        self.assertLess(widths[-1],widths[0]/4)


if __name__ == '__main__':
    unittest.main()
