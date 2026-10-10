"""Independently validate saved prism geometry, records and noninterpenetration.

Uses the separating-axis theorem on float32 STL vertices rather than the
linear-programming placement solver. Tolerance covers STL coordinate rounding.
"""
import argparse
import hashlib
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
    initial_rotation = np.asarray(info.get('initial_rotation_matrix', np.eye(3)))
    np.testing.assert_allclose(initial_rotation.T@initial_rotation, np.eye(3), atol=1e-12)
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
        initial_normal = np.array(record.get('base_normal_abc_initial', record['base_normal_abc']), float)
        assert (initial_normal >= 0).all() and (initial_normal <= 100).all() and initial_normal[:2].any()
        np.testing.assert_allclose(initial_normal, np.rint(initial_normal), atol=1e-12)
        lab_direction = initial_rotation @ initial_normal
        np.testing.assert_allclose(record['base_normal_abc'], lab_direction, atol=1e-10)
        np.testing.assert_allclose(data['normal_abc'][i], lab_direction, atol=1e-10)
        normal = lab_direction/np.linalg.norm(lab_direction)
        np.testing.assert_allclose(record['base_normal_unit_xyz'], normal, atol=1e-12)
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
        assert np.max(np.abs((v @ initial_rotation)[:, :2])) <= 27000.+1e-7
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
    assert hashlib.sha256(path.read_bytes()).hexdigest() == info['stl_sha256']
    face_dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
    raw = np.fromfile(path, dtype=face_dtype, offset=84)
    support = raw['vertices'][:-count*20]
    film = union_all([Polygon(t[:, :2]) for t in support[np.all(np.isclose(support[:, :, 2], 5., atol=1e-6), axis=1)]])
    contact_region = film.buffer(.05)
    if info.get('grid_tiles_per_axis', 1) > 1:
        tiles = info['grid_tiles_per_axis']
        copper = union_all([Polygon(t[:, :2]) for t in support[np.all(np.isclose(support[:, :, 2], 10000., atol=1e-6), axis=1)]])
        assert copper.geom_type == 'Polygon' and len(copper.interiors) == tiles*tiles
        np.testing.assert_allclose(copper.area, tiles*tiles*(85000.**2-54000.**2), rtol=1e-6)
        for local, lab in zip(info['grid_tile_centers_local_nm'], info['grid_tile_centers_lab_nm']):
            np.testing.assert_allclose(np.asarray(local) @ initial_rotation.T, lab, atol=1e-8)
            assert not copper.covers(Point(lab[:2]))
            if np.linalg.norm(local) > 0:
                assert film.distance(Point(lab[:2])) > 30000.
        unrotated_support = support @ initial_rotation
        np.testing.assert_allclose(np.ptp(unrotated_support[:, :, :2].reshape(-1, 2), axis=0),
                                   [tiles*85000., tiles*85000.], atol=.05)
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
