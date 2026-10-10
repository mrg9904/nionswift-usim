"""Independently validate saved prism geometry, records and noninterpenetration.

Uses the separating-axis theorem on float32 STL vertices rather than the
linear-programming placement solver. Tolerance covers STL coordinate rounding.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from shapely.geometry import MultiPoint, Polygon, Point
from shapely import union_all
from create_lacey_carbon import STL_SAMPLES
from create_thousand_cathodes import SpatialGrid


def validate(root, count=1000):
    info = json.loads((root/f'{count}_cathodes.json').read_text(encoding='utf-8'))
    records = info['particles']
    data = np.load(root/f'{count}_cathodes.npz', allow_pickle=False)
    assert len(records) == count
    np.testing.assert_array_equal(data['id'], np.arange(1, count+1))
    vertices = np.array([r['vertices_lab_nm'] for r in records])
    normals, directions = [], []
    for i, record in enumerate(records):
        v = vertices[i]
        edges = np.roll(v[:6], -1, axis=0)-v[:6]
        np.testing.assert_allclose(edges[:3], -edges[3:], atol=1e-7)
        lengths = np.linalg.norm(edges, axis=1)
        assert np.all((lengths >= 50.) & (lengths <= 1000.))
        assert 50. <= record['height_nm'] <= 1000.
        assert v[:, 2].max() <= 5.+info['max_stack_height_nm']+1e-7
        np.testing.assert_allclose(lengths, record['base_edge_lengths_nm'], atol=1e-7)
        normal = np.array(record['base_normal_abc'], float)
        assert (normal >= 0).all() and (normal <= 100).all() and normal[:2].any()
        normal /= np.linalg.norm(normal)
        r = np.array(record['rotation_matrix_local_to_lab'])
        np.testing.assert_allclose(r.T@r, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(r[:, 2], normal, atol=1e-12)
        np.testing.assert_allclose(v[6:]-v[:6], np.tile(normal*record['height_nm'], (6, 1)), atol=1e-7)
        local = np.column_stack((record['base_vertices_local_nm'], np.zeros(6)))
        np.testing.assert_allclose(local@r.T+record['bottom_face_center_nm'], v[:6], atol=1e-7)
        np.testing.assert_allclose(data['center_nm'][i], v.mean(axis=0), atol=1e-8)
        np.testing.assert_allclose(data['rotation_matrix'][i], r, atol=1e-12)
        transform = np.asarray(record['transform_base_local_to_lab'])
        np.testing.assert_allclose(transform[:3, :3], r, atol=1e-12)
        np.testing.assert_allclose(transform[:3, 3], record['bottom_face_center_nm'], atol=1e-8)
        np.testing.assert_allclose(data['transform_base_local_to_lab'][i], transform, atol=1e-12)
        assert np.max(np.abs(v[:, :2])) <= 27000.+1e-7
        contact_id = record['contact_particle_id']
        assert 0 <= contact_id < record['id']
        if contact_id == 0:
            assert np.isclose(v[:, 2].min(), 5.)
        edge_axes = np.vstack((edges[:3], normal))
        edge_axes /= np.linalg.norm(edge_axes, axis=1)[:, None]
        directions.append(edge_axes)
        normals.append(np.vstack((np.cross(edge_axes[:3], normal), normal)))
    # Verify the actual binary STL's ordered particle triangles, not just JSON.
    path = root/info['stl_file']
    face_dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
    raw = np.fromfile(path, dtype=face_dtype, offset=84)
    support = raw['vertices'][:-count*20]
    film = union_all([Polygon(t[:, :2]) for t in support[np.all(np.isclose(support[:, :, 2], 5., atol=1e-6), axis=1)]])
    contact_region = film.buffer(.05)
    for record in records:
        if not record['stacked']:
            assert contact_region.covers(Point(record['contact_position_nm'][:2]))
    faces = []
    for i in range(1, 5):
        faces.extend(((0, i+1, i), (6, 6+i, 6+i+1)))
    for i in range(6):
        j = (i+1) % 6
        faces.extend(((i, j, 6+j), (i, 6+j, 6+i)))
    np.testing.assert_array_equal(raw['vertices'][-count*20:].reshape(count, 20, 3, 3),
                                  vertices[:, np.array(faces)].astype(np.float32))
    stored_vertices = vertices.astype(np.float32).astype(float)
    assert stored_vertices[:, :, 2].max() <= 5.+info['max_stack_height_nm']+.05
    grid, checked = SpatialGrid(), 0
    for i, v in enumerate(stored_vertices):
        bounds = MultiPoint(v[:, :2]).convex_hull.bounds
        for j in grid.query(bounds):
            other = stored_vertices[j]
            if np.any(v.max(axis=0) < other.min(axis=0)) or np.any(other.max(axis=0) < v.min(axis=0)):
                continue
            axes = np.vstack((normals[i], normals[j],
                              np.cross(directions[i][:, None, :], directions[j][None, :, :]).reshape(-1, 3)))
            norm = np.linalg.norm(axes, axis=1)
            axes = axes[norm > 1e-9]/norm[norm > 1e-9, None]
            first, second = v@axes.T, other@axes.T
            separation = np.maximum(first.min(axis=0)-second.max(axis=0), second.min(axis=0)-first.max(axis=0))
            if separation.max() < -.05:
                raise AssertionError(f'Particles {i+1} and {j+1} interpenetrate beyond STL rounding tolerance')
            checked += 1
        grid.add(i, bounds)
    print(f'PASS: {count} prisms; all dimensions, parallel sides, normals, rotations and STL/array correspondence valid; {checked} nearby 3D pairs pass separating-axis test (0.05 nm tolerance).')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=STL_SAMPLES)
    parser.add_argument('--count', type=int, default=1000)
    args = parser.parse_args()
    validate(args.root, args.count)
