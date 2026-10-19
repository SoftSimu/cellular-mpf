"""Committed surface legs through geometry, table I/O and persistent changes."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import pf3d_analyze_contacts as analysis
import pf3d_surface as surface
from contact_geometry_test import cell_from_dense, header, sphere_phi, square


def stream_bytes(frames, *, interval=10):
    count = len(frames[0][1])
    values = [b'PF3SUR1\0', 1, surface.FILE.size, count, 7, 0,
        surface.VERTEX.size, surface.CELL.size, surface.FRAME.size,
        96,96,96, 1.,1.,1., .01,10000.,.5, interval, 0,0, 0.,0.,1., 0,1,1,0]
    raw = bytearray(surface.FILE.pack(*values))
    struct.pack_into('<Q',raw,160,zlib.crc32(raw))
    for step,cells in frames:
        vertices = sum(cell.vertex_count for cell in cells)
        transaction = bytearray(surface.FRAME.pack(b'P3SFRM1\0',step,step*.01,count,vertices,0))
        for cell in cells:
            transaction += surface.CELL.pack(cell.id,*cell.origin,cell.edge,0,cell.vertex_count,
                cell.gamma,cell.active_speed,cell.radius,0.,*cell.polarity,*cell.velocity,cell.tumbles,cell.flags)
            if cell.data:
                transaction += surface.CHUNK.pack(b'P3SCHK1\0',cell.vertex_count,len(cell.data),zlib.crc32(cell.data))
                transaction += cell.data
        raw += transaction
        raw += surface.FOOTER.pack(b'P3SCMT1\0',step,vertices,zlib.crc32(transaction))
    raw += surface.FOOTER.pack(b'P3SEND1\0',len(frames),frames[-1][0],0)
    return raw


class ContactCLI(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='pf3d-contacts-')
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.a = cell_from_dense(sphere_phi(radius=3.),gid=2,origin=(10,10,10))
        self.near = replace(self.a,id=9,origin=(17,10,10))
        self.far = replace(self.near,origin=(40,10,10))

    def write_stream(self, name, frames, **kwargs):
        path = self.folder/name
        path.write_bytes(stream_bytes(frames,**kwargs))
        return path

    def run_cli(self, *args, expect=0):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
            code = analysis.main([str(arg) for arg in args])
        self.assertEqual(code,expect,errors.getvalue())
        return errors.getvalue()

    def measure(self, *paths, name='contacts.jsonl', extras=(), expect=0):
        out = self.folder/name
        self.run_cli('measure',*paths,'--output',out,'--gap',1.5,*extras,expect=expect)
        return out

    def track(self, source, *, name='changes.jsonl', extras=(), expect=0):
        out = self.folder/name
        self.run_cli('track',source,'--output',out,'--area-on',2.,'--area-off',1.,
            '--persistence-steps',10,'--max-frame-gap',20,'--exchange-window',20,*extras,expect=expect)
        return out

    def records(self, path):
        return list(analysis.read_records(path))

    def test_real_reader_to_geometry_to_persistent_gain(self):
        path = self.write_stream('a.pfs3d',[(0,(self.a,self.far))]+
                                 [(step,(self.a,self.near)) for step in (10,20,30,40)])
        table = self.measure(path)
        frames = self.records(table)[1:-1]
        self.assertFalse(frames[0]['contacts'])
        self.assertEqual(len(frames[1]['contacts']),1)
        self.assertTrue(all(cell['topology_usable'] for cell in frames[0]['cells']))
        changes = self.records(self.track(table))
        self.assertEqual(changes[-1]['counts']['contact_gained'],1)
        self.assertIsNone(changes[-1]['confirmed_3d_topological_events'])
        coarse = self.records(self.track(table,name='coarse.jsonl',extras=('--stride',2)))
        self.assertEqual(coarse[-1]['counts']['contact_gained'],1)

    def test_restart_endpoint_is_deduplicated_even_with_different_packing(self):
        chunks = [self.near.data[i:i+surface.VERTEX.size] for i in range(0,len(self.near.data),surface.VERTEX.size)]
        repacked = replace(self.near,data=b''.join(reversed(chunks)))
        a = self.write_stream('a.pfs3d',[(0,(self.a,self.near)),(10,(self.a,self.near))])
        b = self.write_stream('b.pfs3d',[(10,(repacked,self.a)),(20,(self.near,self.a))])
        result = self.records(self.measure(a,b))[-1]
        self.assertEqual((result['frames'],result['deduplicated_endpoints']),(3,1))

    def test_changed_endpoint_and_overlapping_legs_are_rejected(self):
        a = self.write_stream('a.pfs3d',[(0,(self.a,self.near)),(10,(self.a,self.near))])
        b = self.write_stream('b.pfs3d',[(10,(self.a,replace(self.near,tumbles=1)))])
        out = self.measure(a,b,expect=2)
        self.assertIn('restart endpoint',self.records(out)[-1]['message'])
        self.track(out,expect=2)
        self.measure(a,a,name='overlap.jsonl',expect=2)

    def test_identity_changes_and_geometry_contract_changes_are_rejected(self):
        a = self.write_stream('a.pfs3d',[(0,(self.a,self.near))])
        b = self.write_stream('b.pfs3d',[(10,(self.a,replace(self.near,id=7)))])
        self.measure(a,b,expect=2)
        damaged = bytearray(a.read_bytes())
        struct.pack_into('<d',damaged,88,2.)  # dt remains positive but differs
        struct.pack_into('<Q',damaged,160,0)
        struct.pack_into('<Q',damaged,160,zlib.crc32(damaged[:surface.FILE.size]))
        b.write_bytes(damaged)
        self.measure(a,b,name='different-contract.jsonl',expect=2)

    def test_truncated_tail_recovery_is_explicit_and_corruption_still_fails(self):
        path = self.write_stream('a.pfs3d',[(0,(self.a,self.near)),(10,(self.a,self.near))])
        raw = path.read_bytes()
        path.write_bytes(raw[:-5])
        self.measure(path,expect=2)
        recovered = self.measure(path,name='recovered.jsonl',extras=('--recover',))
        self.assertEqual(self.records(recovered)[-1]['status'],'recovered_prefix')
        tracked = self.track(recovered)
        self.assertEqual(self.records(tracked)[-1]['source_status'],'recovered_prefix')
        broken = bytearray(raw)
        broken[surface.FILE.size+surface.FRAME.size+surface.CELL.size+surface.CHUNK.size+5] ^= 1
        path.write_bytes(broken)
        self.measure(path,name='corrupt.jsonl',extras=('--recover',),expect=2)

    def test_gap_resets_and_unresolved_endpoints_are_censored(self):
        path = self.write_stream('a.pfs3d',[(0,(self.a,self.far)),(100,(self.a,self.near))])
        result = self.records(self.track(self.measure(path)))[-1]
        self.assertEqual(result['counts']['contact_gained'],0)
        self.assertEqual(result['counts']['observation_resets'],2)
        short = self.write_stream('b.pfs3d',[(0,(self.a,self.far)),(10,(self.a,self.near))])
        result = self.records(self.track(self.measure(short,name='short.jsonl'),name='pending.jsonl'))[-1]
        self.assertEqual(len(result['right_censored_pending']),1)

    def test_empty_interface_is_censored_not_a_measured_loss(self):
        empty = replace(self.near,data=b'')
        path = self.write_stream('a.pfs3d',[(0,(self.a,self.near)),(10,(self.a,empty)),
                                          (20,(self.a,self.far)),(30,(self.a,self.far))])
        table = self.measure(path)
        frame = self.records(table)[2]
        self.assertFalse(frame['cells'][1]['observable'])
        result = self.records(self.track(table))[-1]
        self.assertEqual(result['counts']['contact_lost'],0)

    def test_pending_establishment_and_gap_censoring_are_distinguished(self):
        empty = replace(self.near,data=b'')
        path = self.write_stream('a.pfs3d',[(0,(self.a,empty)),(10,(self.a,self.near))])
        result = self.records(self.track(self.measure(path)))[-1]
        self.assertEqual(result['right_censored_pending'][0]['kind'],'pending_state_establishment')
        path = self.write_stream('b.pfs3d',[(0,(self.a,self.far)),(10,(self.a,self.near)),(100,(self.a,self.near))])
        result = self.records(self.track(self.measure(path,name='gap.jsonl'),name='gap-changes.jsonl'))[-1]
        self.assertEqual(result['counts']['internally_censored_changes'],1)
        self.assertEqual(result['counts']['contact_gained'],0)

    def test_unexplained_mesh_hole_is_not_topologically_usable(self):
        args = analysis.parser().parse_args(['measure','unused','--gap','1','--output','unused'])
        frame = surface.Frame(0,0.,(self.a,),0)
        with patch('pf3d_contact.mesh_from_surface',return_value=square(self.a.id)):
            record = analysis.measure_frame(frame,header(),args)
        cell, = record['cells']
        self.assertTrue(cell['observable'])
        self.assertEqual(cell['boundary_edges'],4)
        self.assertFalse(cell['topology_usable'])

    def test_output_is_never_overwritten_and_prefix_is_not_complete(self):
        path = self.write_stream('a.pfs3d',[(0,(self.a,self.near)),(10,(self.a,self.near))])
        table = self.measure(path,extras=('--limit-frames',1))
        before = table.read_bytes()
        self.assertEqual(self.records(table)[-1]['status'],'bounded_prefix')
        self.measure(path,expect=2)
        self.assertEqual(table.read_bytes(),before)
        self.run_cli('measure',path,'--output',path,'--gap',1.5,expect=2)
        with surface.SurfaceReader(path) as reader:
            self.assertEqual(len(list(reader)),2)

    def test_parallel_measurement_is_ordered_and_equal_to_serial(self):
        path = self.write_stream('parallel.pfs3d', [(0, (self.a, self.far))] +
            [(step, (self.a, self.near)) for step in (10, 20, 30, 40)])
        serial = self.records(self.measure(path))
        parallel = self.records(self.measure(path, name='parallel.jsonl', extras=('--workers', 2)))
        for rows in (serial, parallel):
            rows[0].pop('workers')
            for row in rows[1:-1]:
                row.pop('analysis_seconds')
        self.assertEqual(serial, parallel)

    def test_parallel_prefix_does_not_schedule_or_report_extra_frames(self):
        path = self.write_stream('prefix.pfs3d', [(step, (self.a, self.near)) for step in (0, 10, 20)])
        rows = self.records(self.measure(path, extras=('--workers', 2, '--limit-frames', 1)))
        self.assertEqual(rows[-1]['frames'], 1)
        self.assertEqual(rows[-1]['status'], 'bounded_prefix')
        self.assertEqual(rows[-1]['streams'][0]['read_frames'], 1)

    def test_recorded_integrity_failure_is_censored_not_a_contact_loss(self):
        bad = replace(self.near, flags=1 << 3)
        path = self.write_stream('flagged.pfs3d', [(0,(self.a,self.near)),
            (10,(self.a,bad)), (20,(self.a,self.far)), (30,(self.a,self.far))])
        table = self.measure(path)
        self.assertFalse(self.records(table)[2]['cells'][1]['observable'])
        self.assertEqual(self.records(self.track(table))[-1]['counts']['contact_lost'], 0)
        hint = replace(self.near, flags=1 << 6)
        path = self.write_stream('hint.pfs3d', [(0,(self.a,hint))])
        self.assertTrue(self.records(self.measure(path,name='hint.jsonl'))[1]['cells'][1]['observable'])


if __name__ == '__main__':
    unittest.main()
