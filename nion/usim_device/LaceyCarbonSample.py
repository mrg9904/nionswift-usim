"""Smooth 5 nm amorphous carbon support and opaque 10 um copper grid.

The combined STL is in nm. Its 54 um opening identifies carbon versus copper;
material parameters are phenomenological, as in the existing EELS simulator.
"""
import numpy as np
import json
from pathlib import Path
from nion.usim_device import SampleGeometry
from nion.usim_device import EELSModel, HAADFFocusModel, STLDepthSample, SimulationSettings
from nion.utils import Geometry


class LaceyCarbonSample(STLDepthSample.STLDepthSample):
    def __init__(self, stage_size_nm):
        root = Path(__file__).with_name('samples')
        metadata = json.loads((root/'single_cathode.json').read_text(encoding='utf-8'))
        def support_only(mesh):
            faces = np.flatnonzero(~SampleGeometry.cathode_face_mask(mesh, metadata))
            return mesh.submesh([faces], append=True)
        super().__init__(stage_size_nm, file_name="hexagonal_prism_on_lacey_carbon_with_copper_grid.stl",
                         shift_nm=(0., 0.), mesh_filter=support_only)

    @property
    def title(self):
        return "LaceyCarbon"

    @property
    def initial_view(self):
        return Geometry.FloatPoint(), 200.

    def eels_layers_at(self, position_m):
        layers = super().eels_layers_at(position_m)
        if not layers:
            return []
        carbon = max(abs(position_m.x), abs(position_m.y)) < 27000e-9
        material = EELSModel.EELSMaterial(
            edges=((284., 15.),) if carbon else ((75., 20.), (932., 30.), (952., 30.)),
            plasmon_eV=24. if carbon else 19., mean_free_path_nm=150. if carbon else 100.)
        return [EELSModel.EELSLayer(layers[0].thickness_nm, material)]

    def thickness_projection(self, offset_m, fov_size_nm, extra_nm, center_nm, used_size):
        data = np.zeros(tuple(used_size), np.float32)
        super().plot_features(data, offset_m, fov_size_nm, extra_nm, center_nm, used_size)
        return data * SimulationSettings.STL_REFERENCE_THICKNESS_NM

    def plot_features(self, data, offset_m, fov_size_nm, extra_nm, center_nm, used_size):
        thickness = self.thickness_projection(offset_m, fov_size_nm, extra_nm, center_nm, used_size)
        # Transmission stays nonnegative even in the opaque copper frame.
        data += -100 * np.expm1(-thickness / SimulationSettings.RONCHIGRAM_TRANSMISSION_LENGTH_NM)

    def generate_depth_planes(self, offset_m, fov_size_nm, extra_nm, center_nm, used_size, slice_thickness_nm):
        if slice_thickness_nm <= 0:
            raise ValueError("slice_thickness_nm must be positive")
        thickness = self.thickness_projection(offset_m, fov_size_nm, extra_nm, center_nm, used_size)
        # The thin film is one slab; opaque copper uses entrance-surface contrast.
        # Do not slice a 10 um opaque grid into thousands of defocus planes.
        carbon = (thickness > 0) & (thickness < 10)
        copper = thickness >= 10
        planes = [(2.5, carbon.astype(np.float32) * .25), (0., copper.astype(np.float32))]
        return HAADFFocusModel.PreparedDepthPlanes(planes)
