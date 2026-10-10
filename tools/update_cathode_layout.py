"""Update the existing particles' layout without resampling sizes or contacts."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import trimesh
from shapely.geometry import Polygon
from create_lacey_carbon import STL_SAMPLES, SAMPLES, verify
from create_thousand_cathodes import save_size_distribution
from cathode_layout import apply_layout, transform_records, save_arrays, save_layout_preview
from nion.usim_device import SampleGeometry


def update(root=STL_SAMPLES, output=SAMPLES/'thousand_cathodes', degrees=60., tiles=3):
    info = json.loads((root/'1000_cathodes.json').read_text(encoding='utf-8'))
    records = info['particles']
    source = root/info['source_stl']
    if hashlib.sha256(source.read_bytes()).hexdigest() != info['source_sha256']:
        raise ValueError('Original support changed; regenerate the specimen explicitly')
    mesh = trimesh.load_mesh(source)
    single = json.loads((root/'single_cathode.json').read_text(encoding='utf-8'))
    support = mesh.submesh([np.flatnonzero(~SampleGeometry.cathode_face_mask(mesh, single))], append=True)
    old_rotation = np.asarray(info.get('initial_rotation_matrix', np.eye(3)))
    transform_records(records, old_rotation.T)
    for record in records:
        if 'base_normal_abc_initial' in record:
            record['base_normal_abc'] = record['base_normal_abc_initial'].copy()
    combined, layout = apply_layout(support, records, degrees, tiles)
    crystal_volume = sum(Polygon(r['base_vertices_local_nm']).area*r['height_nm'] for r in records)
    expected_volume = support.volume+(tiles*tiles-1)*(85000.**2-54000.**2)*10000.+crystal_volume
    verify(combined, expected_volume)
    stl = combined.export(file_type='stl')
    loaded = trimesh.load_mesh(io.BytesIO(stl), file_type='stl')
    verify(loaded, expected_volume)
    info.update(layout)
    info.update(bounds_nm=loaded.bounds.tolist(), faces=len(loaded.faces), watertight=bool(loaded.is_watertight),
                stl_sha256=hashlib.sha256(stl).hexdigest())
    # The previously registered sample reads these same three artifacts.
    (root/info['stl_file']).write_bytes(stl)
    (root/'1000_cathodes.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    save_arrays(info, root)
    output.mkdir(parents=True, exist_ok=True)
    save_layout_preview(info, support, output)
    save_size_distribution(info, output)
    summary = {k: v for k, v in info.items() if k != 'particles'}
    (output/'model_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=STL_SAMPLES)
    parser.add_argument('--output', type=Path, default=SAMPLES/'thousand_cathodes')
    parser.add_argument('--rotation-deg', type=float, default=60.)
    parser.add_argument('--grid-tiles', type=int, default=3)
    args = parser.parse_args()
    update(args.root, args.output, args.rotation_deg, args.grid_tiles)
