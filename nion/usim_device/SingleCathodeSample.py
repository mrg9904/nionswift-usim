"""Tilted crystalline hexagonal prism on an amorphous lacey carbon support.

Material intervals are separate: the air gap under a tilted prism is vacuum.
STL extrema across the entire assembly must never be used as its thickness.
"""
import json
import logging
from pathlib import Path
import numpy as np
from nion.utils import Geometry
from nion.usim_device import EELSModel, HAADFFocusModel, LaceyCarbonSample, STLDepthSample, SurfaceRasterizer
from nion.usim_device import SampleGeometry
from nion.usim_device import TiltedSurfaceRasterizer
from nion.usim_device import SimulationSettings


class SingleCathodeSample(LaceyCarbonSample.LaceyCarbonSample):
    def __init__(self, stage_size_nm):
        root = Path(__file__).with_name('samples')
        self.model_info = json.loads((root/'single_cathode.json').read_text(encoding='utf-8'))
        STLDepthSample.STLDepthSample.__init__(self, stage_size_nm,
            file_name='hexagonal_prism_on_lacey_carbon_with_copper_grid.stl', shift_nm=(0., 0.))
        crystal_faces = SampleGeometry.cathode_face_mask(self._mesh, self.model_info)
        self._support_triangles = self._mesh.triangles[~crystal_faces].copy()
        self._crystal_triangles = self._mesh.triangles[crystal_faces].copy()
        self.center_nm = np.unique(self._crystal_triangles.reshape(-1, 3), axis=0).mean(axis=0)
        self._support = TiltedSurfaceRasterizer.TiltedSurfaceRasterizer(self._support_triangles, self.center_nm)
        self._crystal = TiltedSurfaceRasterizer.TiltedSurfaceRasterizer(self._crystal_triangles, self.center_nm)
        self.crystal_cif_path = str(root/'NNMTO_pristine.cif')
        self.zone_axis = (0, 0, 1)
        n = np.asarray(self.model_info['base_normal_unit_xyz'])
        u = np.cross([0., 0., 1.], n)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        spin = np.radians(self.model_info['spin_about_prism_axis_deg'])
        self.crystal_rotation = np.column_stack((np.cos(spin)*u+np.sin(spin)*v,
            -np.sin(spin)*u+np.cos(spin)*v, n))
        self._projection_key = self._projection = self._planes_key = self._planes = None
        self._gpu_layers_enabled = True

    def set_stage_tilt(self, tilt):
        if tilt == self._stage_tilt_rad:
            return
        # Tilt about the cathode center (the selected stage/eucentric point).
        self._support.set_tilt(tilt)
        self._crystal.set_tilt(tilt)
        self._stage_tilt_rad = tilt
        self._projection_key = self._projection = self._planes_key = self._planes = None

    def _carbon_mask(self, x, y, lower, upper):
        # Classify the support in its untilted frame, rather than treating a
        # foreshortened carbon film as copper once its beam thickness exceeds 10 nm.
        z = (lower+upper)*.5
        axis = SampleGeometry.stage_rotation(self.stage_tilt_rad)[:, 2]
        original_z = ((x[None, :]-self.center_nm[0])*axis[0]
            +(y[:, None]-self.center_nm[1])*axis[1]+(z-self.center_nm[2])*axis[2]+self.center_nm[2])
        return original_z < 5.01

    @property
    def title(self):
        return 'SingleCathodeOnCarbon'

    @property
    def initial_view(self):
        # uSim's sample coordinate is scan_center - stage_offset.
        return Geometry.FloatPoint(x=-self.center_nm[0]*1e-9, y=-self.center_nm[1]*1e-9), 200.

    @staticmethod
    def _thickness(lower, upper):
        return np.where(np.isfinite(lower) & np.isfinite(upper), np.maximum(upper-lower, 0.), 0.)

    def crystal_thickness_at(self, position_m):
        lower, upper = self._crystal.surface_maps(np.array([position_m.x*1e9]), np.array([position_m.y*1e9]))
        return float(self._thickness(lower, upper)[0, 0])

    def eels_layers_at(self, position_m):
        x, y = np.array([position_m.x*1e9]), np.array([position_m.y*1e9])
        lo, hi = self._support.surface_maps(x, y)
        support = float(self._thickness(lo, hi)[0, 0])
        carbon = bool(self._carbon_mask(x, y, lo, hi)[0, 0])
        material = EELSModel.EELSMaterial(
            edges=((284., 15.),) if carbon else ((75., 20.), (932., 30.), (952., 30.)),
            plasmon_eV=24. if carbon else 19., mean_free_path_nm=150. if carbon else 100.)
        layers = [EELSModel.EELSLayer(support, material)] if support > 0 else []
        crystal = self.crystal_thickness_at(position_m)
        if crystal > 0:
            layers.append(EELSModel.EELSLayer(crystal, EELSModel.EELSMaterial()))
        return layers

    def _maps(self, offset_m, fov_size_nm, extra_nm, center_nm, used_size):
        key = (offset_m.as_tuple(), fov_size_nm.as_tuple(), extra_nm.as_tuple(), center_nm.as_tuple(), used_size.as_tuple())
        if key != self._projection_key:
            width, height = fov_size_nm.width+extra_nm.x, fov_size_nm.height+extra_nm.y
            x = center_nm.x-offset_m.x*1e9 + ((np.arange(used_size.width)+.5)/used_size.width-.5)*width
            y = center_nm.y-offset_m.y*1e9 + ((np.arange(used_size.height)+.5)/used_size.height-.5)*height
            self._projection = (*self._support.surface_maps(x, y), *self._crystal.surface_maps(x, y))
            self._projection_axes = (x, y)
            self._projection_key = key
        return self._projection

    def thickness_projection(self, *args):
        slo, shi, clo, chi = self._maps(*args)
        return self._thickness(slo, shi)+self._thickness(clo, chi)

    def plot_crystal_features(self, data, *args):
        from nion.usim_device import SimulationSettings
        _, _, lower, upper = self._maps(*args)
        data += -100*np.expm1(-self._thickness(lower, upper)/SimulationSettings.RONCHIGRAM_TRANSMISSION_LENGTH_NM)

    def generate_depth_planes(self, offset_m, fov_size_nm, extra_nm, center_nm, used_size, slice_thickness_nm):
        if slice_thickness_nm <= 0:
            raise ValueError('slice_thickness_nm must be positive')
        maps = self._maps(offset_m, fov_size_nm, extra_nm, center_nm, used_size)
        key = (self._projection_key, slice_thickness_nm)
        if key == self._planes_key:
            return self._planes
        slo, shi, lower, upper = maps
        support = self._thickness(slo, shi)
        carbon = self._carbon_mask(*self._projection_axes, slo, shi) & (support > 0)
        copper = (support > 0) & ~carbon
        intervals = []
        for lo, hi, valid in ((slo, shi, carbon), (lower, upper, np.isfinite(lower) & np.isfinite(upper) & (upper > lower))):
            if valid.any():
                low, high = float(lo[valid].min()), float(hi[valid].max())
                dz = max(slice_thickness_nm, (high-low)/64)
                intervals.append((lo, hi, valid, low, high, dz))
        use_gpu = (self._gpu_layers_enabled and SimulationSettings.STL_SURFACE_BACKEND != 'cpu' and
            (SimulationSettings.STL_SURFACE_BACKEND == 'gpu' or used_size.width*used_size.height >= SimulationSettings.STL_GPU_MINIMUM_PIXELS))
        if use_gpu:
            try:
                from nion.usim_device.GPUHAADF import GPUPreparedLayerPlanes, GPUCompositeDepthPlanes
                layers = [GPUPreparedLayerPlanes(np.where(valid, lo, np.nan), np.where(valid, hi, np.nan), dz, low, high)
                    for lo, hi, valid, low, high, dz in intervals]
                if copper.any():
                    z = float(slo[copper].min())
                    lo = np.where(copper, z-.5, np.nan)
                    layers.append(GPUPreparedLayerPlanes(lo, lo+1., 1., z-.5, z+.5, normalize=True))
                if not layers:
                    empty = np.full(tuple(used_size), np.nan, np.float32)
                    layers.append(GPUPreparedLayerPlanes(empty, empty, 1., -.5, .5))
                self._planes = GPUCompositeDepthPlanes(layers)
                self._planes_key = key
                return self._planes
            except Exception:
                self._gpu_layers_enabled = False
                logging.warning('uSim cathode GPU depth preparation failed; using CPU', exc_info=True)
        planes = [(float(slo[copper].min()) if copper.any() else 0., copper.astype(np.float32))]
        for lo, hi, valid, low, high, dz in intervals:
            for bottom in np.arange(low, high, dz):
                top = min(bottom+dz, high)
                overlap = np.where(valid, np.maximum(np.minimum(hi, top)-np.maximum(lo, bottom), 0.), 0.)
                planes.append(((bottom+top)/2, (overlap/20).astype(np.float32)))
        self._planes = HAADFFocusModel.PreparedDepthPlanes(planes)
        self._planes_key = key
        return self._planes
