"""Spatially culled, independently oriented cathodes on the shared support."""
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np
from nion.utils import Geometry
from nion.usim_device import SingleCathodeSample, STLDepthSample, TiltedSurfaceRasterizer
from nion.usim_device import SampleGeometry, HAADFFocusModel, SimulationSettings


class _Crystal(SingleCathodeSample.SingleCathodeSample):
    """Lightweight crystal view used by the existing diffraction renderer."""
    def __init__(self, owner, index):
        self.owner, self.index = owner, index
        self.crystal_cif_path, self.zone_axis = owner.crystal_cif_path, owner.zone_axis
        self.crystal_rotation = owner.rotations[index]
        self._crystal = TiltedSurfaceRasterizer.TiltedSurfaceRasterizer(
            owner._particle_triangles[index], owner.center_nm)
        self._stage_tilt_rad = Geometry.FloatPoint()
        self._projection_key = self._projection = None

    def set_stage_tilt(self, tilt):
        self._crystal.set_tilt(tilt)
        self._stage_tilt_rad = tilt

    def plot_crystal_features(self, data, *args):
        from nion.usim_device import SimulationSettings
        x, y = self.owner.axes(*args)
        bounds = self.owner._bounds[self.index]
        xi = np.flatnonzero((x >= bounds[0, 0]) & (x <= bounds[1, 0]))
        yi = np.flatnonzero((y >= bounds[0, 1]) & (y <= bounds[1, 1]))
        if len(xi) and len(yi):
            region = slice(yi[0], yi[-1]+1), slice(xi[0], xi[-1]+1)
            lo, hi = self._crystal.surface_maps(x[region[1]], y[region[0]])
            data[region] += -100*np.expm1(-self._thickness(lo, hi)/SimulationSettings.RONCHIGRAM_TRANSMISSION_LENGTH_NM)

    def crystal_thickness_at(self, position_m):
        bounds = self.owner._bounds[self.index]
        if not (bounds[0, 0] <= position_m.x*1e9 <= bounds[1, 0] and
                bounds[0, 1] <= position_m.y*1e9 <= bounds[1, 1]):
            return 0.
        return super().crystal_thickness_at(position_m)


class ThousandCathodeSample(SingleCathodeSample.SingleCathodeSample):
    def __init__(self, stage_size_nm):
        root = Path(__file__).with_name('samples')
        self.model_info = json.loads((root/'1000_cathodes.json').read_text(encoding='utf-8'))
        STLDepthSample.STLDepthSample.__init__(self, stage_size_nm,
            file_name=self.model_info['stl_file'], shift_nm=(0., 0.))
        with np.load(root/'1000_cathodes.npz', allow_pickle=False) as records:
            self.rotations = records['rotation_matrix'].copy()
            self.vertices = records['vertices_lab_nm'].astype(np.float32).astype(float)
        count = len(self.vertices)
        if count != self.model_info['particle_count'] or len(self._mesh.triangles) != self.model_info['faces']:
            raise ValueError('Cathode geometry and orientation records do not match')
        # Exporter appends 20 triangles per prism after the unchanged support.
        self._particle_triangles = self._mesh.triangles[-count*20:].reshape(count, 20, 3, 3).copy()
        for vertices, triangles in zip(self.vertices, self._particle_triangles):
            if not np.all(np.min(np.linalg.norm(triangles.reshape(-1, 1, 3)-vertices[None], axis=2), axis=1) < .01):
                raise ValueError('STL particle order does not match orientation records')
        self.center_nm = self.vertices[0].mean(axis=0)
        self._support = TiltedSurfaceRasterizer.TiltedSurfaceRasterizer(self._mesh.triangles[:-count*20], self.center_nm)
        self.crystal_cif_path, self.zone_axis = str(root/'NNMTO_pristine.cif'), (0, 0, 1)
        self._crystals = OrderedDict()
        self._gpu_layers_enabled = True
        self._gpu_particle_rasterizer = None
        self._gpu_particle_failed = False
        self._projection_key = self._projection = self._planes_key = self._planes = None
        self._update_bounds()

    @property
    def title(self):
        return '1000CathodeParticleOnCarbon'

    @property
    def initial_view(self):
        return Geometry.FloatPoint(x=-self.center_nm[0]*1e-9, y=-self.center_nm[1]*1e-9), 2000.

    def _update_bounds(self):
        vertices = (self.vertices-self.center_nm) @ SampleGeometry.stage_rotation(self.stage_tilt_rad).T+self.center_nm
        self._bounds = np.stack((vertices.min(axis=1), vertices.max(axis=1)), axis=1)

    def set_stage_tilt(self, tilt):
        if tilt == self.stage_tilt_rad:
            return
        self._stage_tilt_rad = tilt
        self._support.set_tilt(tilt)
        for crystal in self._crystals.values():
            crystal.set_stage_tilt(tilt)
        self._update_bounds()
        self._projection_key = self._projection = self._planes_key = self._planes = None

    @staticmethod
    def axes(offset_m, fov_size_nm, extra_nm, center_nm, used_size):
        x = center_nm.x-offset_m.x*1e9+((np.arange(used_size.width)+.5)/used_size.width-.5)*(fov_size_nm.width+extra_nm.x)
        y = center_nm.y-offset_m.y*1e9+((np.arange(used_size.height)+.5)/used_size.height-.5)*(fov_size_nm.height+extra_nm.y)
        return x, y

    def visible_indices(self, x, y):
        bounds = self._bounds
        return np.flatnonzero((bounds[:, 0, 0] <= max(x)) & (bounds[:, 1, 0] >= min(x)) &
            (bounds[:, 0, 1] <= max(y)) & (bounds[:, 1, 1] >= min(y)))

    def crystal(self, index):
        crystal = self._crystals.pop(int(index), None)
        if crystal is None:
            crystal = _Crystal(self, int(index))
            crystal.set_stage_tilt(self.stage_tilt_rad)
        self._crystals[int(index)] = crystal
        # Each prism holds only twenty faces and four tiny rasterizers, not
        # full projection images. Retain the finite specimen's geometry.
        while len(self._crystals) > len(self.vertices):
            self._crystals.popitem(last=False)
        return crystal

    def _maps(self, *args):
        key = tuple(arg.as_tuple() for arg in args)
        if key != self._projection_key:
            x, y = self.axes(*args)
            lo, hi = self._support.surface_maps(x, y)
            pieces = []
            self.last_visible_particle_ids = tuple(int(i)+1 for i in self.visible_indices(x, y))
            regions, indices = [], []
            for identifier in self.last_visible_particle_ids:
                i = identifier-1
                bound = self._bounds[i]
                xi = np.flatnonzero((x >= bound[0, 0]) & (x <= bound[1, 0]))
                yi = np.flatnonzero((y >= bound[0, 1]) & (y <= bound[1, 1]))
                if not len(xi) or not len(yi):
                    continue
                region = (slice(yi[0], yi[-1]+1), slice(xi[0], xi[-1]+1))
                indices.append(i)
                regions.append(region)
            projections = None
            if indices and not self._gpu_particle_failed and SimulationSettings.STL_SURFACE_BACKEND != 'cpu' and (
                    SimulationSettings.STL_SURFACE_BACKEND == 'gpu' or len(indices) >= 32 or
                    args[-1].width*args[-1].height >= SimulationSettings.STL_GPU_MINIMUM_PIXELS):
                try:
                    from nion.usim_device.GPUParticleRasterizer import GPUParticleRasterizer
                    if self._gpu_particle_rasterizer is None:
                        self._gpu_particle_rasterizer = GPUParticleRasterizer()
                    triangles = (self._particle_triangles[indices]-self.center_nm) @ SampleGeometry.stage_rotation(self.stage_tilt_rad).T+self.center_nm
                    rectangles = np.asarray([[r[1].start, r[1].stop, r[0].start, r[0].stop] for r in regions], np.int32)
                    projections = self._gpu_particle_rasterizer.project(triangles, rectangles, x, y)
                except Exception as error:
                    import logging
                    logging.getLogger(__name__).warning('Batched particle CUDA unavailable; using CPU: %s', error)
                    self._gpu_particle_failed = True
            if projections is None:
                projections = [self.crystal(i)._crystal.surface_maps(x[region[1]], y[region[0]]) for i, region in zip(indices, regions)]
            pieces = [(i, region, lower, upper) for i, region, (lower, upper) in zip(indices, regions, projections)]
            self._projection_axes = x, y
            self._projection = lo, hi, pieces
            self._projection_key = key
        return self._projection

    def thickness_projection(self, *args):
        lo, hi, pieces = self._maps(*args)
        thickness = self._thickness(lo, hi)
        for _, region, lower, upper in pieces:
            thickness[region] += self._thickness(lower, upper)
        return thickness

    def crystal_thickness_at(self, position_m):
        x, y = [position_m.x*1e9], [position_m.y*1e9]
        return sum(self.crystal(i).crystal_thickness_at(position_m) for i in self.visible_indices(x, y))

    def plot_crystal_features(self, data, *args):
        from nion.usim_device import SimulationSettings
        _, _, pieces = self._maps(*args)
        thickness = np.zeros(data.shape, np.float32)
        for _, region, lower, upper in pieces:
            thickness[region] += self._thickness(lower, upper)
        data += -100*np.expm1(-thickness/SimulationSettings.RONCHIGRAM_TRANSMISSION_LENGTH_NM)

    def generate_depth_planes(self, offset_m, fov_size_nm, extra_nm, center_nm, used_size, slice_thickness_nm):
        if slice_thickness_nm <= 0:
            raise ValueError('slice_thickness_nm must be positive')
        lo, hi, pieces = self._maps(offset_m, fov_size_nm, extra_nm, center_nm, used_size)
        key = self._projection_key, slice_thickness_nm
        if key == self._planes_key:
            return self._planes
        support = self._thickness(lo, hi)
        carbon = self._carbon_mask(*self._projection_axes, lo, hi) & (support > 0)
        copper = (support > 0) & ~carbon
        intervals = [(slice(None), slice(None), np.where(carbon, lo, np.nan), np.where(carbon, hi, np.nan))]
        intervals.extend((region[0], region[1], lower, upper) for _, region, lower, upper in pieces)
        valid_intervals = [(ys, xs, lower, upper) for ys, xs, lower, upper in intervals if np.isfinite(lower).any()]
        planes = [(float(lo[copper].min()) if copper.any() else 0., copper.astype(np.float32))]
        depth_limits = []
        if valid_intervals:
            bottom = min(float(np.nanmin(lower)) for _, _, lower, _ in valid_intervals)
            top = max(float(np.nanmax(upper)) for _, _, _, upper in valid_intervals)
            dz = max(slice_thickness_nm, (top-bottom)/64.)
            depth_limits = [(z, min(z+dz, top)) for z in np.arange(bottom, top, dz)]
        if self._gpu_layers_enabled and self._support.uses_gpu(tuple(used_size)):
            try:
                from nion.usim_device.GPUHAADF import GPUSparseDepthPlanes
                self._planes = GPUSparseDepthPlanes(tuple(used_size), valid_intervals, depth_limits, planes[0])
                self._planes_key = key
                return self._planes
            except Exception:
                import logging
                logging.warning('uSim sparse HAADF CUDA unavailable; using CPU', exc_info=True)
                self._gpu_layers_enabled = False
        stack = np.zeros((len(depth_limits), *tuple(used_size)), np.float32)
        for ys, xs, lower, upper in valid_intervals:
            low, high = float(np.nanmin(lower)), float(np.nanmax(upper))
            for index, (z, end) in enumerate(depth_limits):
                if end <= low or z >= high:
                    continue
                stack[index, ys, xs] += np.nan_to_num(np.maximum(np.minimum(upper, end)-np.maximum(lower, z), 0.)/20.).astype(np.float32)
        planes.extend(((z+end)/2., plane) for (z, end), plane in zip(depth_limits, stack) if plane.any())
        self._planes = HAADFFocusModel.PreparedDepthPlanes(planes)
        self._planes_key = key
        return self._planes
