#!/usr/bin/env python3
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from stack_channel_centres import assemble


class Stacks(unittest.TestCase):
    def parents(self):
        return [dict(path=f'parent-{p}', side=200, height=98, radius=49,
                     cells=[dict(id=i,x=x,y=y,z=49) for i,(x,y) in
                            enumerate(((10,20),(100,120)))]) for p in range(2)]

    def test_reproducible_and_geometry(self):
        points, info = assemble(self.parents(), 1701, 98)
        self.assertEqual((points, info), assemble(self.parents(), 1701, 98))
        self.assertNotEqual(points, assemble(self.parents(), 1702, 98)[0])
        self.assertEqual([p[0] for p in points], list(range(4)))
        self.assertEqual([p[3] for p in points], [49,49,147,147])
        self.assertEqual(info['H'], 196)
        self.assertGreaterEqual(info['minimum_centre_distance'], 49)

    def test_wrong_box_and_duplicate(self):
        parents = self.parents()
        with self.assertRaises(ValueError): assemble([parents[0],parents[0]], 1, 98)
        parents[1]['side'] = 210
        with self.assertRaises(ValueError): assemble(parents, 1, 98)

    def test_pitch_and_nonsingle_layer(self):
        with self.assertRaises(ValueError): assemble(self.parents(), 1, float('nan'))
        parents = self.parents(); parents[1]['height'] = 196
        with self.assertRaises(ValueError): assemble(parents, 1, 98)
        self.assertEqual(assemble(self.parents(), 1, 90)[1]['H'], 188)


if __name__ == '__main__':
    unittest.main()
