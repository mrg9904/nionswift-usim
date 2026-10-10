from __future__ import annotations

# standard libraries
import typing
import math
import logging
import numpy
import numpy.typing
import scipy.ndimage
import scipy.stats

from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.device_kit import InstrumentDevice
from nion.instrumentation import stem_controller
from nion.usim_device import CameraSimulator
from nion.usim_device import Noise
from nion.usim_device import EELSModel
from nion.usim_device import KikuchiModel
from nion.usim_device import RonchigramContrast
from nion.usim_device import SampleSimulator
from nion.usim_device import SampleGeometry
from nion.usim_device import LaceyCarbonSample
from nion.usim_device import SimulationSettings
from nion.usim_device import InstrumentDevice as InstrumentDevice_
from nion.utils import Geometry


_NDArray = numpy.typing.NDArray[typing.Any]


class AberrationsController:
    """Track aberrations and apply them.

    All values are SI.

    Derived from code by Juan-Carlos Idrobo and Andy Lupini.
    """
    coefficient_names = (
        "c0a", "c0b", "c10", "c12a", "c12b", "c21a", "c21b", "c23a", "c23b", "c30", "c32a", "c32b", "c34a", "c34b",
        "c41a", "c41b", "c43a", "c43b", "c45a", "c45b", "c50", "c52a", "c52b", "c54a", "c54b", "c56a", "c56b", "c70"
    )

    def __init__(self, height: int, width: int, theta: float, max_defocus: float, defocus: float) -> None:
        self.__height = height
        self.__width = width
        self.__theta = theta
        self.__max_defocus = max_defocus
        self.__coefficients: typing.Dict[str, float] = dict()
        self.__intermediates: typing.Dict[str, _NDArray] = dict()  # functions of height/width/theta
        self.__chis: typing.Dict[str, _NDArray] = dict()  # chi's, functions of intermediate and coefficients
        self.__coefficients["c10"] = defocus
        self.__chi: typing.Optional[_NDArray] = None
        self.__c: typing.Optional[typing.List[float]] = None
        self.__gpu_mapper = None
        self.__gpu_failed = False

    def apply(self, aberrations: typing.Mapping[str, typing.Union[int, float]], data: numpy.typing.NDArray[numpy.float32]) -> _NDArray:
        height = int(aberrations["height"])
        width = int(aberrations["width"])
        theta = aberrations["theta"]
        max_defocus = self.__max_defocus * float(aberrations.get("source_scale", 1.0))
        if getattr(self, "_source_scale", None) != max_defocus:
            self._source_scale = max_defocus
            self.__c = None

        if theta != self.__theta or width != self.__width or height != self.__height:
            self.__width = width
            self.__height = height
            self.__theta = theta
            self.__intermediates = dict()
            self.__chis = dict()
            self.__chi = None
            self.__c = None

        for coefficient_name in self.coefficient_names:
            if self.__coefficients.get(coefficient_name) != aberrations.get(coefficient_name):
                # print(f"changed {coefficient_name}")
                self.__coefficients[coefficient_name] = aberrations[coefficient_name]
                if coefficient_name in ("c0a", "c0b"):
                    # Beam displacement contributes a constant phase gradient.
                    # Keep the nonlinear mapping when only the probe moves.
                    continue
                self.__chis.pop(coefficient_name, None)
                self.__chi = None
                self.__c = None

        # below: the tedious part...

        def get_i0ab() -> typing.Tuple[_NDArray, _NDArray]:
            i0a = self.__intermediates.get("c0a")
            i0b = self.__intermediates.get("c0b")
            if i0a is None or i0b is None:
                i0a, i0b = numpy.meshgrid(numpy.linspace(-theta, theta, width), numpy.linspace(-theta, theta, height))
                self.__intermediates["c0a"] = i0a
                self.__intermediates["c0b"] = i0b
            return i0a, i0b

        def get_i0a() -> _NDArray:
            return get_i0ab()[0]

        def get_i0b() -> _NDArray:
            return get_i0ab()[1]

        def get_i0a_squared() -> _NDArray:
            i0a_squared = self.__intermediates.get("c0a_squared")
            if i0a_squared is None:
                i0a_squared = get_i0a() ** 2
                self.__intermediates["c0a_squared"] = i0a_squared
            return i0a_squared

        def get_i0b_squared() -> _NDArray:
            i0b_squared = self.__intermediates.get("c0b_squared")
            if i0b_squared is None:
                i0b_squared = get_i0b() ** 2
                self.__intermediates["c0b_squared"] = i0b_squared
            return i0b_squared

        def get_iradius() -> _NDArray:
            ir = self.__intermediates.get("ir")
            if ir is None:
                ir = get_i0a_squared() + get_i0b_squared()
                self.__intermediates["ir"] = ir
            return ir

        def get_idiff_sq() -> _NDArray:
            ids = self.__intermediates.get("ids")
            if ids is None:
                ids = get_i0a_squared() - get_i0b_squared()
                self.__intermediates["ids"] = ids
            return ids

        def get_intermediate(coefficient_name: str) -> _NDArray:
            intermediate = self.__intermediates.get(coefficient_name)
            if intermediate is None:
                if coefficient_name == "c0a":
                    intermediate = get_i0a()
                elif coefficient_name == "c0b":
                    intermediate = get_i0b()
                elif coefficient_name == "c10":
                    intermediate = get_iradius() / 2
                elif coefficient_name == "c12a":
                    intermediate = get_idiff_sq() / 2
                elif coefficient_name == "c12b":
                    intermediate = get_i0a() * get_i0b()
                elif coefficient_name == "c21a":
                    intermediate = get_i0a() * (get_i0a_squared() + get_i0b_squared()) / 3
                elif coefficient_name == "c21b":
                    intermediate = get_i0b() * (get_i0a_squared() + get_i0b_squared()) / 3
                elif coefficient_name == "c23a":
                    intermediate = get_i0a() * (get_i0a_squared() - 3 * get_i0b_squared()) / 3
                elif coefficient_name == "c23b":
                    intermediate = get_i0b() * (3 * get_i0a_squared() - get_i0b_squared()) / 3
                elif coefficient_name == "c30":
                    intermediate = get_intermediate("c10") ** 2
                elif coefficient_name == "c32a":
                    intermediate = get_intermediate("c10") * get_intermediate("c12a")
                elif coefficient_name == "c32b":
                    intermediate = get_intermediate("c10") * get_intermediate("c12b")
                elif coefficient_name == "c34a":
                    intermediate = (get_i0a_squared() ** 2 - 6 * get_i0a_squared() * get_i0b_squared() + get_i0b_squared() ** 2) / 4
                elif coefficient_name == "c34b":
                    intermediate = get_i0a() ** 3 * get_i0b() - get_i0a() * get_i0b() ** 3
                elif coefficient_name == "c41a":
                    intermediate = 4 * get_i0a() * get_intermediate("c10") ** 2 / 5
                elif coefficient_name == "c41b":
                    intermediate = 4 * get_i0b() * get_intermediate("c10") ** 2 / 5
                elif coefficient_name == "c43a":
                    intermediate = get_iradius() * (get_i0a() * get_idiff_sq() - 2 * get_i0a() * get_i0b() ** 2) / 5
                elif coefficient_name == "c43b":
                    intermediate = get_iradius() * (get_i0b() * get_idiff_sq() + 2 * get_i0b() * get_i0a() ** 2) / 5
                elif coefficient_name == "c45a":
                    intermediate = (get_i0a() * get_idiff_sq() ** 2 - 4 * get_i0a() * get_idiff_sq() * get_i0b() ** 2 - 4 * get_i0a() ** 3 * get_i0b() ** 2) / 5
                elif coefficient_name == "c45b":
                    # this type: ignore is inexplicably necessary; seems like bug in numpy.typing or mypy. try removing to see if it passes typing tests.
                    intermediate = (get_i0b() * get_idiff_sq() ** 2 + 4 * get_i0b() * get_idiff_sq() * get_i0a() ** 2 - 4 * get_i0a() ** 2 * get_i0b() ** 3) / 5  # type: ignore
                elif coefficient_name == "c50":
                    intermediate = 8 * get_intermediate("c10") ** 3 / 6
                elif coefficient_name == "c52a":
                    intermediate = get_iradius() ** 2 * get_idiff_sq() / 6
                elif coefficient_name == "c52b":
                    intermediate = get_iradius() ** 2 * get_intermediate("c12b") / 3
                elif coefficient_name == "c54a":
                    intermediate = get_iradius() * (get_idiff_sq() ** 2 - 4 * get_intermediate("c12b") ** 2) / 6
                elif coefficient_name == "c54b":
                    intermediate = 2 * get_iradius() * get_idiff_sq() * get_intermediate("c12b") / 3
                elif coefficient_name == "c56a":
                    intermediate = get_idiff_sq() ** 3 / 6 - 2 * get_i0a() ** 2 * get_i0b() ** 2 * get_idiff_sq()
                elif coefficient_name == "c56b":
                    intermediate = get_intermediate("c12b") * get_idiff_sq() ** 2 - (4 * get_i0a() ** 3 * get_i0b() ** 3) / 3
                elif coefficient_name == "c70":
                    intermediate = 2 * get_intermediate("c10") ** 4
            assert intermediate is not None
            return intermediate

        def get_chi(coefficient_name: str) -> typing.Optional[_NDArray]:
            chi = self.__chis.get(coefficient_name)
            if chi is None:
                coefficient = self.__coefficients.get(coefficient_name, 0.0)
                if coefficient != 0.0:
                    chi = coefficient * get_intermediate(coefficient_name)
                if chi is not None:
                    self.__chis[coefficient_name] = chi
                else:
                    self.__chis.pop(coefficient_name, None)
            return chi

        if self.__chi is None:
            # print("recalculating chi")
            for coefficient_name in self.coefficient_names:
                if coefficient_name in ("c0a", "c0b"):
                    continue
                partial_chi = get_chi(coefficient_name)
                if partial_chi is not None:
                    if self.__chi is None:
                        # print(f"0 {coefficient_name}")
                        self.__chi = numpy.copy(partial_chi)
                    else:
                        # print(f"+ {coefficient_name}")
                        self.__chi += partial_chi
            self.__c = None

        if self.__chi is None:
            # An aberration-free focused beam still transmits electrons.
            self.__chi = numpy.zeros((height, width))

        if self.__c is None and self.__chi is not None:
            # print("recalculating grad chi")
            grad_chi = numpy.gradient(self.__chi)
            max_chi0 = max_defocus * theta * theta
            max_chi1 = max_defocus * theta * theta * ((1 - 1 / width) * (1 - 1 / width) + (1 - 1 / height) * (1 - 1 / height)) / 2
            max_chi = max_chi0 - max_chi1
            scale_y = height / 2 / max_chi
            scale_x = width / 2 / max_chi
            self.__c = [scale_y * grad_chi[0] + height/2, scale_x * grad_chi[1] + width/2]

        # note, the scaling factor of 2pi/wavelength has been removed from chi since it cancels out.

        if self.__c is not None:
            # scale the offsets so that at max defocus, the coordinates cover the entire area of data.
            max_chi = max_defocus * theta * theta * (1 - ((1 - 1 / width)**2 + (1 - 1 / height)**2)/2)
            dy = height/2/max_chi * (2*theta/(height-1)) * self.__coefficients.get("c0b", 0.)
            dx = width/2/max_chi * (2*theta/(width-1)) * self.__coefficients.get("c0a", 0.)
            backend = SimulationSettings.RONCHIGRAM_BACKEND
            if not self.__gpu_failed and (backend == "gpu" or (backend == "auto" and data.size >= SimulationSettings.RONCHIGRAM_GPU_MINIMUM_PIXELS)):
                try:
                    if self.__gpu_mapper is None:
                        self.__gpu_mapper = RonchigramContrast.GPUMapper()
                    return self.__gpu_mapper.apply(data, self.__c, dy, dx)
                except Exception as error:
                    logging.getLogger(__name__).warning("Ronchigram GPU mapping failed; using CPU: %s", error)
                    self.__gpu_failed = True
                    self.__gpu_mapper = None
            coordinates = [self.__c[0] + dy, self.__c[1] + dx]
            return scipy.ndimage.map_coordinates(data, coordinates, order=1)  # type: ignore

        return numpy.zeros((height, width))

    def ray_coordinates(self):
        """Reuse the exact ray map of the last apply, without evaluating rays again."""
        height, width = self.__height, self.__width
        theta, max_defocus = self.__theta, self._source_scale
        max_chi = max_defocus*theta*theta*(1-((1-1/width)**2+(1-1/height)**2)/2)
        dy = height/2/max_chi*(2*theta/(height-1))*self.__coefficients.get('c0b', 0.)
        dx = width/2/max_chi*(2*theta/(width-1))*self.__coefficients.get('c0a', 0.)
        return self.__c[0]+dy, self.__c[1]+dx

    def source_displacement(self, position, image_shape):
        """Relative source coordinates of a displayed pixel and image centre.

        Use the actual nonlinear aberration mapping, including defocus sign.
        The constant beam/probe displacement cancels between the two points.
        """
        coordinates = self.__c
        if coordinates is None:
            return None
        height, width = coordinates[0].shape
        row = (position.y+.5)/image_shape[0]*height-.5
        column = (position.x+.5)/image_shape[1]*width-.5
        points = [[row, (height-1)/2], [column, (width-1)/2]]
        y = scipy.ndimage.map_coordinates(coordinates[0], points, order=1, mode="nearest")
        x = scipy.ndimage.map_coordinates(coordinates[1], points, order=1, mode="nearest")
        return Geometry.FloatPoint(y=float(y[0]-y[1])/height, x=float(x[0]-x[1])/width)


def ellipse_radius(polar_angle: typing.Union[float, _NDArray], a: float, b: float, rotation: float) -> typing.Union[float, _NDArray]:
    """
    Returns the radius of a point lying on an ellipse with the given parameters. The ellipse is described in polar
    coordinates here, which makes it easy to incorporate a rotation.

    Parameters
    -----------
    polar_angle : float or _NDArray
                  Polar angle of a point to which the corresponding radius should be calculated (rad).
    a : float
        Length of the major half-axis of the ellipse.
    b : float
        Length of the minor half-axis of the ellipse.
    rotation : Rotation of the ellipse with respect to the x-axis (rad). Counter-clockwise is positive.

    Returns
    --------
    radius : float or _NDArray
             Radius of a point lying on an ellipse with the given parameters.
    """

    return a * b / numpy.sqrt((b * numpy.cos(polar_angle + rotation)) ** 2 + (a * numpy.sin(polar_angle + rotation)) ** 2)  # type: ignore


def draw_ellipse(image: _NDArray, ellipse: typing.Tuple[float, float, float, float, float], *, color: typing.Any = 1.0) -> None:
    """
    Draws an ellipse on a 2D-array.

    Parameters
    ----------
    image : array
            The array on which the ellipse will be drawn. Note that the data will be modified in place.
    ellipse : tuple
              A tuple describing an ellipse with the same moments as the aperture. The values must be (in this order):
              [0] The y-coordinate of the center.
              [1] The x-coordinate of the center.
              [2] The length of the major half-axis
              [3] The length of the minor half-axis
              [4] The rotation of the ellipse in rad.
    color : optional
            The color to which the pixels inside the given ellipse will be set. Note that `color` will be cast to the
            type of `image` automatically. If this is not possible, an exception will be raised. The default is 1.0.

    Returns
    --------
    None
    """
    shape = image.shape
    assert len(shape) == 2, 'Can only draw an ellipse on a 2D-array.'
    # coords = np.mgrid[-shape[0]/2:shape[0]/2:shape[0]*1j, -shape[1]/2:shape[1]/2:shape[1]*1j]
    top = max(int(ellipse[0] - ellipse[2]), 0)
    left = max(int(ellipse[1] - ellipse[2]), 0)
    bottom = min(int(ellipse[0] + ellipse[2]) + 1, shape[0])
    right = min(int(ellipse[1] + ellipse[2]) + 1, shape[1])
    coords = numpy.mgrid[top - ellipse[0]:bottom - ellipse[0], left - ellipse[1]:right - ellipse[1]]  # type: ignore
    # coords[0] -= ellipse[0]
    # coords[1] -= ellipse[1]
    radii = numpy.sqrt(numpy.sum(coords**2, axis=0))
    polar_angles = numpy.arctan2(coords[0], coords[1])
    ellipse_radii = ellipse_radius(polar_angles, *ellipse[2:])
    image[top:bottom, left:right][radii < ellipse_radii] = color


class RonchigramCameraSimulator(CameraSimulator.CameraSimulator):
    depends_on = ["C10Control", "C12Control", "C21Control", "C23Control", "C30Control", "C32Control", "C34Control",
                  "C34Control", "stage_position_m", "probe_state", "probe_position", "features",
                  "beam_shift_m", "is_blanked", "BeamCurrent", "CAperture", "ApertureRound", "S_VOA", "S_MOA",
                  "ConvergenceAngle", "stage_tilt_rad", "stage_z_m", "EHT"]

    def __init__(self, instrument: InstrumentDevice_.Instrument, ronchigram_shape: Geometry.IntSize, counts_per_electron: int, stage_size_nm: float) -> None:
        super().__init__(instrument, "ronchigram", ronchigram_shape, counts_per_electron)
        self.__cached_frame: typing.Optional[DataAndMetadata.DataAndMetadata] = None
        max_defocus = instrument.max_defocus
        self.__stage_size_nm = stage_size_nm
        self.__source_fov_nm = stage_size_nm
        self.__data_scale = 1.0
        self.__aperture_ellipse: typing.Optional[typing.Tuple[float, float, float, float, float]] = None
        self.__aperture_mask = None
        theta = self._tv_pixel_angle * ronchigram_shape.height / 2  # half angle on camera
        defocus_m = instrument.defocus_m
        self.__aberrations_controller = AberrationsController(ronchigram_shape.height, ronchigram_shape.width, theta, max_defocus, defocus_m)
        self.noise = Noise.ElectronCountingNoise(counts_per_electron)
        self.__kikuchi_cache_key = None
        self.__kikuchi_pattern = None
        self.__crystal_settings = None
        self.__contrast_composer = None
        self.__source_key = None
        self.__source_data = None
        self.__line_renderer = None
        self.__line_renderer_failed = False

    def close(self) -> None:
        if hasattr(self, '_multi_crystal_cache'):
            self._multi_crystal_cache.clear()
        if hasattr(self, '_multi_bands_cache'):
            self._multi_bands_cache.clear()
        self.__line_renderer = None
        self.__contrast_composer = None
        self.__kikuchi_pattern = None
        self.__source_data = None
        self.__cached_frame = None
        self.__aberrations_controller = None
        self.noise.clear_gpu_cache()
        super().close()

    def _source_image(self, readout_area, binning_shape, *, crystalline=False, sample=None):
        """Cache the sphere projection independently of probe/tilt/defocus."""
        sample = sample if sample is not None else self.instrument.scan_data_generator.sample
        SampleGeometry.prepare_sample(sample, self.instrument)
        offset_m = self.instrument.stage_position_m
        sphere = isinstance(sample, SampleSimulator.SphericalParticleSample)
        lacey = isinstance(sample, LaceyCarbonSample.LaceyCarbonSample)
        key = None
        if sphere:
            key = (id(sample), offset_m.y, offset_m.x,
                   readout_area.top, readout_area.left, readout_area.height, readout_area.width,
                   binning_shape.height, binning_shape.width,
                   SimulationSettings.RONCHIGRAM_TRANSMISSION_LENGTH_NM,
                   tuple((f.position_m.y, f.position_m.x, f.radius_nm, f.reference_thickness_nm)
                         for f in sample.features))
            if key == self.__source_key and self.__source_data is not None:
                return self.__source_data
        elif lacey and not crystalline:
            key = (id(sample), sample.stage_tilt_rad.as_tuple(), offset_m.y, offset_m.x, self.__source_fov_nm, readout_area.as_tuple(),
                   binning_shape.as_tuple(), SimulationSettings.RONCHIGRAM_TRANSMISSION_LENGTH_NM)
            if key == self.__source_key and self.__source_data is not None:
                return self.__source_data
        height, width = readout_area.height, readout_area.width
        full_fov_nm = self.__source_fov_nm
        fov_size_nm = Geometry.FloatSize(full_fov_nm*height/self._sensor_dimensions.height,
                                        full_fov_nm*width/self._sensor_dimensions.width)
        center_nm = Geometry.FloatPoint(full_fov_nm*(readout_area.center.y/self._sensor_dimensions.height-.5),
                                       full_fov_nm*(readout_area.center.x/self._sensor_dimensions.width-.5))
        data = numpy.zeros((height, width), numpy.float32)
        plot = sample.plot_crystal_features if crystalline else sample.plot_features
        plot(data, offset_m, fov_size_nm, Geometry.FloatPoint(), center_nm, Geometry.IntSize(height, width))
        if sphere:
            data = 100*KikuchiModel.transmission(data*sample.features[0].reference_thickness_nm)
        else:
            data = 100-data
        data = self._get_binned_data(data, binning_shape)
        if not crystalline:
            self.__source_key = key
            self.__source_data = data if sphere or lacey else None
        return data

    def stage_displacement_for_pixel(self, position, image_shape):
        """Convert a Ronchigram double click into specimen displacement in m."""
        settings = self._last_frame_settings
        if settings is None or self._needs_recalculation or self.__cached_frame is None:
            return None
        if self.instrument.value_manager.is_blanked:
            return None
        # Undo the display rotation before converting an image pixel to a ray.
        raw_position = Geometry.FloatPoint(image_shape[0]-1-position.y, image_shape[1]-1-position.x)
        displacement = self.__aberrations_controller.source_displacement(raw_position, image_shape)
        if displacement is None:
            return None
        area = settings.readout_area
        return Geometry.FloatPoint(
            y=displacement.y*self.__source_fov_nm*area.height/self._sensor_dimensions.height*1e-9,
            x=displacement.x*self.__source_fov_nm*area.width/self._sensor_dimensions.width*1e-9)

    def _apply_kikuchi(self, data, readout_area, binning_shape, frame_settings, scan_context, aberrations=None, sample=None, crystal_data=None, band_area=None, halo=None):
        sample = sample if sample is not None else self.instrument.scan_data_generator.sample
        if hasattr(sample, 'visible_indices'):
            return self._apply_multiple_crystals(data, readout_area, binning_shape, frame_settings, scan_context, aberrations, sample)
        if not getattr(sample, 'crystal_cif_path', None):
            return data, {}
        manager = typing.cast(InstrumentDevice_.ValueManager, self.instrument.value_manager)
        position = EELSModel.probe_sample_position(manager.actual_offset_m,
            scan_context.fov_size_nm or Geometry.FloatSize(), scan_context.center_nm or Geometry.FloatPoint(),
            frame_settings.current_probe_position or Geometry.FloatPoint(.5, .5), scan_context.rotation_rad)
        composite = hasattr(sample, 'plot_crystal_features')
        thickness_nm = (sample.crystal_thickness_at(position) if composite else
            sum(feature.thickness_at(position) for feature in sample.features))
        supplied_crystal = crystal_data is not None
        if not supplied_crystal:
            crystal_data = data
        if composite and not supplied_crystal:
            source = self._source_image(readout_area, binning_shape, crystalline=True, sample=sample)
            crystal_data = self.__aberrations_controller.apply(aberrations, source)
        sample_rotation = getattr(sample, 'crystal_rotation', None)
        rotation_key = tuple(numpy.asarray(sample_rotation).ravel()) if sample_rotation is not None else ()
        tilt = self.instrument.GetVal2D('stage_tilt_rad')
        voltage = self.instrument.GetVal('EHT')
        convergence = self.instrument.GetVal('ConvergenceAngle')
        metadata = {'model': 'geometric_kikuchi_v2', 'cif': sample.crystal_cif_path,
            'zone_axis': list(sample.zone_axis), 'thickness_nm': thickness_nm,
            'voltage_v': voltage, 'convergence_semiangle_rad': convergence,
            'stage_tilt_rad': {'tx': tilt.x, 'ty': tilt.y},
            'probe_position_sample_m': {'x': position.x, 'y': position.y}, 'band_count': 0}
        metadata['d_min_angstrom'] = SimulationSettings.KIKUCHI_D_MIN_ANGSTROM
        if composite:
            metadata['crystal_rotation_matrix'] = numpy.asarray(sample_rotation).tolist()
            metadata['crystal_001_direction_xyz'] = numpy.asarray(sample_rotation)[:, 2].tolist()
        # The parked position is the beam centre, not the whole illuminated
        # region. Defocused/aberrated rays can cross the crystal even when
        # this centre is in vacuum. Use the projected transmission footprint.
        vacuum_level = 100*binning_shape.height*binning_shape.width
        if not bool(((crystal_data > 0) & (crystal_data < vacuum_level-1e-4)).any()):
            return data, metadata
        calibrations = self._raw_dimensional_calibrations(readout_area, binning_shape)
        y_rad = calibrations[0].offset + numpy.arange(data.shape[0])*calibrations[0].scale
        x_rad = calibrations[1].offset + numpy.arange(data.shape[1])*calibrations[1].scale
        angular_y, angular_x = y_rad, x_rad
        if band_area is not None:
            full_calibrations = self._raw_dimensional_calibrations(band_area, binning_shape)
            angular_y = full_calibrations[0].offset+numpy.arange(band_area.height//binning_shape.height)*full_calibrations[0].scale
            angular_x = full_calibrations[1].offset+numpy.arange(band_area.width//binning_shape.width)*full_calibrations[1].scale
        max_angle = math.atan(math.hypot(math.tan(max(abs(angular_x))), math.tan(max(abs(angular_y)))))
        key = (sample.crystal_cif_path, tuple(sample.zone_axis), rotation_key, voltage, tilt.x, tilt.y,
               tuple(data.shape), tuple((c.offset, c.scale) for c in calibrations),
               SimulationSettings.KIKUCHI_D_MIN_ANGSTROM, SimulationSettings.KIKUCHI_MAX_BANDS,
               SimulationSettings.RONCHIGRAM_BACKEND,
               SimulationSettings.KIKUCHI_BROADENING_RAD_AT_100_NM,
               SimulationSettings.RONCHIGRAM_PATTERN_CACHE_BYTES, max_angle)
        if key != self.__kikuchi_cache_key:
            crystal = KikuchiModel.load_crystal(sample.crystal_cif_path)
            band_key = (sample.crystal_cif_path, tuple(sample.zone_axis), voltage, tilt.x, tilt.y, max_angle,
                        SimulationSettings.KIKUCHI_D_MIN_ANGSTROM, SimulationSettings.KIKUCHI_MAX_BANDS)
            visible = None
            if supplied_crystal:
                # Pixel bounds change while focusing, crystal orientation does
                # not. Reuse reflectors independently of each cropped image.
                if getattr(self, '_multi_bands_key', None) != band_key:
                    self._multi_bands_key = band_key
                    self._multi_bands_cache = {}
                visible = self._multi_bands_cache.get(rotation_key)
            if visible is None:
                visible = KikuchiModel.bands(crystal, voltage, tuple(sample.zone_axis), tilt.x, tilt.y,
                    max_angle_rad=max_angle, d_min_angstrom=SimulationSettings.KIKUCHI_D_MIN_ANGSTROM,
                    max_bands=SimulationSettings.KIKUCHI_MAX_BANDS, sample_rotation=sample_rotation)
                if supplied_crystal:
                    if len(self._multi_bands_cache) >= 1000:
                        self._multi_bands_cache.clear()
                    self._multi_bands_cache[rotation_key] = visible
            # Bound interactive raster work. Width is >=0.35 mrad; a 512 grid
            # resolves it at the default camera angle, then interpolate to readout.
            height, width = min(512, data.shape[0]), min(512, data.shape[1])
            grid_x, grid_y = numpy.linspace(x_rad[0], x_rad[-1], width), numpy.linspace(y_rad[0], y_rad[-1], height)
            pattern = None
            use_gpu = (SimulationSettings.RONCHIGRAM_BACKEND == 'gpu' or
                (SimulationSettings.RONCHIGRAM_BACKEND == 'auto' and data.size >= SimulationSettings.RONCHIGRAM_GPU_MINIMUM_PIXELS))
            if use_gpu and not self.__line_renderer_failed:
                try:
                    if self.__line_renderer is None:
                        self.__line_renderer = RonchigramContrast.GPULineRenderer()
                    pattern = self.__line_renderer.render(visible, grid_x, grid_y, output_shape=data.shape)
                except Exception as error:
                    self.__line_renderer_failed = True
                    logging.warning('Ronchigram CUDA line renderer unavailable; using CPU: %s', error)
            if pattern is None:
                pattern = KikuchiModel.render_lines(visible, grid_x, grid_y)
            if pattern.shape != data.shape:
                yy, xx = numpy.meshgrid(numpy.linspace(0, height-1, data.shape[0]),
                    numpy.linspace(0, width-1, data.shape[1]), indexing='ij')
                pattern = scipy.ndimage.map_coordinates(pattern, [yy, xx], order=1)
            # Do not apply a virtual aperture here. Physical VOA/MOA masks
            # are applied below only when their insertion controls are on.
            self.__kikuchi_pattern = (pattern, len(visible))
            self.__contrast_composer = RonchigramContrast.ContrastComposer(pattern,
                (calibrations[0].scale, calibrations[1].scale), SimulationSettings.RONCHIGRAM_BACKEND)
            self.__kikuchi_cache_key = key
        pattern, metadata['band_count'] = self.__kikuchi_pattern
        composed, contrast_metadata = self.__contrast_composer.compose(crystal_data, vacuum_level, halo=halo)
        if composite:
            xp = self.__contrast_composer.xp
            real = xp.asarray(data) if xp is not numpy or isinstance(data, numpy.ndarray) else data.get()
            crystal_real = xp.asarray(crystal_data) if xp is not numpy or isinstance(crystal_data, numpy.ndarray) else crystal_data.get()
            # Only the crystal produces Kikuchi bands. Carbon/copper still
            # attenuate rays, but must not masquerade as crystalline material.
            data = xp.asarray(composed) * xp.where(crystal_real > 0, real/xp.maximum(crystal_real, 1e-12), 0.)
        else:
            data = composed
        metadata.update(contrast_metadata)
        metadata['probe_diffusion_sigma_rad'] = float(KikuchiModel.diffusion_sigma_rad(thickness_nm))
        return data, metadata

    def _apply_multiple_crystals(self, data, area, binning, settings, context, aberrations, sample):
        # Cull in the physical source footprint, independent of scan FoV.
        # Auto uses GPU for the shared large ray map, then CPU for tiny local
        # compositions where hundreds of CUDA launches would cost more.
        if SimulationSettings.RONCHIGRAM_BACKEND != 'gpu' and not isinstance(data, numpy.ndarray):
            data = data.get()
        fov = self.__source_fov_nm
        size = Geometry.FloatSize(fov*area.height/self._sensor_dimensions.height,
                                  fov*area.width/self._sensor_dimensions.width)
        center = Geometry.FloatPoint(fov*(area.center.y/self._sensor_dimensions.height-.5),
                                     fov*(area.center.x/self._sensor_dimensions.width-.5))
        x, y = sample.axes(self.instrument.stage_position_m, size, Geometry.FloatPoint(), center,
                           Geometry.IntSize(area.height, area.width))
        from nion.usim_device import ParticleProjection
        _, _, pieces = sample._maps(self.instrument.stage_position_m, size, Geometry.FloatPoint(), center,
                                   Geometry.IntSize(area.height, area.width))
        mapped, inside, mapping_backend = ParticleProjection.mapped_particles(pieces, (area.height, area.width),
            self.__aberrations_controller.ray_coordinates(), binning.as_tuple())
        result = data.copy()
        records = []
        cache = getattr(self, '_multi_crystal_cache', None)
        if cache is None:
            from collections import OrderedDict
            cache = self._multi_crystal_cache = OrderedDict()
        def cache_bytes(entry):
            pattern, composer = entry[1][0], entry[2]
            return pattern.nbytes+composer.pattern.nbytes+(composer.bank.nbytes if composer.bank is not None else 0)
        cached_bytes = sum(cache_bytes(entry) for entry in cache.values())
        vacuum = 100*binning.height*binning.width
        calibrations = self._raw_dimensional_calibrations(area, binning)
        jobs = []
        for index, (pixels, attenuation) in mapped.items():
            # Only pixels traversing this crystal change. Include a Gaussian
            # halo so local composition retains the full-frame boundary model.
            thickness_max = -SimulationSettings.RONCHIGRAM_TRANSMISSION_LENGTH_NM*numpy.log(
                max(1-attenuation.max()/vacuum, 1e-12))
            blur_level = 100.*2**max(0, math.ceil(math.log2(max(thickness_max/100., 1.))))
            sigma_rad = max(SimulationSettings.RONCHIGRAM_DIFFUSE_SIGMA_RAD,
                SimulationSettings.KIKUCHI_BROADENING_RAD_AT_100_NM*math.sqrt(blur_level/100.))
            pad_y = int(math.ceil(4*sigma_rad/abs(calibrations[0].scale)))+2
            pad_x = int(math.ceil(4*sigma_rad/abs(calibrations[1].scale)))+2
            rows, columns = numpy.unravel_index(pixels, data.shape)
            top, bottom = max(0, int(rows.min())-pad_y), min(data.shape[0], int(rows.max())+pad_y+1)
            left, right = max(0, int(columns.min())-pad_x), min(data.shape[1], int(columns.max())+pad_x+1)
            # A one-pixel focus shift should not rebuild every pattern and
            # blur bank. Expand to stable tiles while keeping exact ray data.
            tile = 32
            top, left = top//tile*tile, left//tile*tile
            bottom = min(data.shape[0], (bottom+tile-1)//tile*tile)
            right = min(data.shape[1], (right+tile-1)//tile*tile)
            region = slice(top, bottom), slice(left, right)
            projected = numpy.where(inside[region], vacuum, 0.).astype(numpy.float32)
            projected[rows-top, columns-left] -= attenuation
            cropped_area = Geometry.IntRect(origin=Geometry.IntPoint(area.top+top*binning.height, area.left+left*binning.width),
                size=Geometry.IntSize((bottom-top)*binning.height, (right-left)*binning.width))
            jobs.append((index, region, projected, cropped_area))
        halos = (RonchigramContrast.batched_halos([job[2] for job in jobs],
                 (calibrations[0].scale, calibrations[1].scale), vacuum) if mapping_backend == 'gpu'
                 else [None]*len(jobs))
        for (index, region, projected, cropped_area), halo in zip(jobs, halos):
            crystal = sample.crystal(index)
            cache_id = (id(sample), int(index))
            previous = cache.pop(cache_id, None)
            if previous is None:
                self.__kikuchi_cache_key = None
            else:
                cached_bytes -= cache_bytes(previous)
                self.__kikuchi_cache_key, self.__kikuchi_pattern, self.__contrast_composer = previous
            composed, metadata = self._apply_kikuchi(data[region], cropped_area, binning, settings, context, aberrations,
                                                    sample=crystal, crystal_data=projected, band_area=area, halo=halo)
            if metadata.get('band_count', 0):
                if isinstance(result, numpy.ndarray) and not isinstance(composed, numpy.ndarray):
                    composed = composed.get()
                elif not isinstance(result, numpy.ndarray) and isinstance(composed, numpy.ndarray):
                    import cupy as cp
                    composed = cp.asarray(composed)
                result[region] += composed-data[region]
                metadata['particle_id'] = int(index)+1
                records.append(metadata)
                cache[cache_id] = (self.__kikuchi_cache_key, self.__kikuchi_pattern, self.__contrast_composer)
                cached_bytes += cache_bytes(cache[cache_id])
            # One shared budget instead of four entries that thrash at every
            # large-view frame. Count both host patterns and device blur banks.
            while len(cache) > 1 and cached_bytes > SimulationSettings.RONCHIGRAM_PATTERN_CACHE_BYTES:
                cached_bytes -= cache_bytes(cache.popitem(last=False)[1])
        # The single-crystal cache must not retain the last particle's key.
        self.__kikuchi_cache_key = None
        return result.clip(0), {'model': 'geometric_kikuchi_multi_crystal', 'crystals': records,
            'visible_particle_ids': [int(i)+1 for i in sample.visible_indices(x, y)],
            'mapped_particle_count': len(mapped),
            'mapping_backend': mapping_backend,
            'diffusion_backend': 'gpu' if halos and all(halo is not None for halo in halos) else 'cpu',
            'band_count': sum(record['band_count'] for record in records)}

    @property
    def _tv_pixel_angle(self) -> float:
        instrument = self.instrument
        return math.asin(instrument.stage_size_nm / (instrument.max_defocus * 1E9)) / self._camera_shape.height

    @property
    def _max_defocus(self) -> float:
        return self.instrument.max_defocus

    def _draw_aperture(self, frame_data: _NDArray, binning_shape: Geometry.IntSize, enlarge_by: float = 0.0) -> None:
        # TODO handle asymmetric binning
        binning = binning_shape[0]
        position = self.instrument.GetVal2D("CAperture")
        aperture_round = self.instrument.GetVal2D("ApertureRound")
        shape = frame_data.shape
        ellipse_center = 0.5 * shape[0] + position.y / self._tv_pixel_angle / binning, 0.5 * shape[1] + position.x / self._tv_pixel_angle / binning
        excentricity = math.sqrt(aperture_round[0]**2 + aperture_round[1]**2)
        # adapt excentricity so that control behaves linearly and is defined for all values
        # this is the modified inverse function of the calculation of the major half-axis a
        excentricity = math.sqrt(1-1/(1+abs(excentricity))**4)
        direction = numpy.arctan2(aperture_round[0], aperture_round[1])
        # Calculate a and b (the ellipse half-axes) from excentricity. Keep ellipse area constant
        convergence_angle = self.instrument.GetVal("ConvergenceAngle") * (1 + enlarge_by)
        convergence_angle_pixels = convergence_angle / self._tv_pixel_angle / binning
        a = math.sqrt(convergence_angle_pixels**2 / math.sqrt(1 - excentricity**2))
        b = convergence_angle_pixels**2 / a
        self.__aperture_ellipse = ellipse_center + (a, b, direction)
        aperture_mask = numpy.zeros_like(frame_data)
        draw_ellipse(aperture_mask, self.__aperture_ellipse)
        frame_data *= aperture_mask

    def get_frame_data(self, readout_area: Geometry.IntRect, binning_shape: Geometry.IntSize, exposure_s: float, scan_context: stem_controller.ScanContext, parked_probe_position: typing.Optional[Geometry.FloatPoint]) -> DataAndMetadata.DataAndMetadata:
        frame_settings = self._get_frame_settings(readout_area, binning_shape, exposure_s, scan_context, parked_probe_position)
        sample = self.instrument.scan_data_generator.sample
        rotation = getattr(sample, 'crystal_rotation', None)
        crystal_settings = (getattr(sample, 'crystal_cif_path', None), tuple(getattr(sample, 'zone_axis', ())),
                            tuple(numpy.asarray(rotation).ravel()) if rotation is not None else ())
        if crystal_settings != self.__crystal_settings:
            self._needs_recalculation = True
            self.__crystal_settings = crystal_settings
        if frame_settings != self._last_frame_settings:
            self._needs_recalculation = True
            self._last_frame_settings = frame_settings

        if self._needs_recalculation or self.__cached_frame is None:
            # print("recalculating frame")
            thickness_param = 100
            value_manager = typing.cast(InstrumentDevice_.ValueManager, self.instrument.value_manager)
            metadata = {}
            effective_defocus = self.instrument.GetVal("C10Control") - self.instrument.GetVal("stage_z_m")
            # Scan FoV defines a probe's physical position, not the camera's
            # angular field or specimen magnification. Only its actual offset
            # enters the source extent and aberration mapping.
            scan_offset = EELSModel.probe_sample_position(Geometry.FloatPoint(),
                scan_context.fov_size_nm or Geometry.FloatSize(), scan_context.center_nm or Geometry.FloatPoint(),
                frame_settings.current_probe_position or Geometry.FloatPoint(.5, .5), scan_context.rotation_rad)
            # Large support films must not be clipped to the historical 1 um
            # source. Enclose ray displacements, with headroom for aberrations;
            # powers of two retain the projection cache across small focus steps.
            self.__source_fov_nm = self.__stage_size_nm
            if isinstance(sample, LaceyCarbonSample.LaceyCarbonSample):
                half_angle = self._tv_pixel_angle * self._sensor_dimensions.height / 2
                beam = self.instrument.GetVal2D("beam_shift_m")
                required = (4 * abs(effective_defocus) * half_angle * 1e9
                            + 2 * max(abs(beam.x+scan_offset.x), abs(beam.y+scan_offset.y)) * 1e9)
                self.__source_fov_nm *= 2 ** max(0, math.ceil(math.log2(max(required / self.__stage_size_nm, 1))))
            if not value_manager.is_blanked:
                data = self._source_image(readout_area, binning_shape)
            else:
                data = numpy.zeros((readout_area.height//binning_shape.height,
                                    readout_area.width//binning_shape.width), numpy.float32)

            if not value_manager.is_blanked:
                theta = self._tv_pixel_angle * self._sensor_dimensions.height / 2  # half angle on camera
                aberrations: typing.Dict[str, typing.Union[float, int]] = dict()
                aberrations["height"] = data.shape[0]
                aberrations["width"] = data.shape[1]
                aberrations["theta"] = theta
                aberrations["source_scale"] = self.__source_fov_nm / self.__stage_size_nm
                aberrations["c0a"] = self.instrument.GetVal2D("beam_shift_m").x + scan_offset[1]
                aberrations["c0b"] = self.instrument.GetVal2D("beam_shift_m").y + scan_offset[0]
                aberrations["c10"] = effective_defocus
                aberrations["c12a"] = self.instrument.GetVal2D("C12Control").x
                aberrations["c12b"] = self.instrument.GetVal2D("C12Control").y
                aberrations["c21a"] = self.instrument.GetVal2D("C21Control").x
                aberrations["c21b"] = self.instrument.GetVal2D("C21Control").y
                aberrations["c23a"] = self.instrument.GetVal2D("C23Control").x
                aberrations["c23b"] = self.instrument.GetVal2D("C23Control").y
                aberrations["c30"] = self.instrument.GetVal("C30Control")
                aberrations["c32a"] = self.instrument.GetVal2D("C32Control").x
                aberrations["c32b"] = self.instrument.GetVal2D("C32Control").y
                aberrations["c34a"] = self.instrument.GetVal2D("C34Control").x
                aberrations["c34b"] = self.instrument.GetVal2D("C34Control").y
                data = self.__aberrations_controller.apply(aberrations, data)
                data, metadata = self._apply_kikuchi(data, readout_area, binning_shape, frame_settings, scan_context, aberrations)
                if not isinstance(data, numpy.ndarray):
                    # Vacuum/non-crystalline frames bypass the compositor.
                    data = data.get()
                if self.instrument.GetVal("S_VOA") > 0:
                    self._draw_aperture(data, binning_shape)
                elif self.instrument.GetVal("S_MOA") > 0:
                    self._draw_aperture(data, binning_shape, enlarge_by=0.1)

            # Align the camera display orientation with HAADF. Rotate the
            # complete image, including real-space contrast and diffraction.
            data = data[::-1, ::-1]
            if metadata:
                metadata['display_rotation_deg'] = 180
            intensity_calibration = Calibration.Calibration(units="counts")
            dimensional_calibrations = self.get_dimensional_calibrations(readout_area, binning_shape)

            self.__cached_frame = DataAndMetadata.new_data_and_metadata(data.astype(numpy.float32), intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations,
                metadata={'kikuchi_simulation': metadata} if metadata else {})
            self.__data_scale = self.get_total_counts(exposure_s) / (data.shape[0] * data.shape[1] * thickness_param)
            self._needs_recalculation = False

        assert self.__cached_frame
        use_gpu = (SimulationSettings.RONCHIGRAM_BACKEND != "cpu"
                   and self.__contrast_composer is not None and self.__contrast_composer.backend == "gpu")
        result = (self.noise.apply_gpu(self.__cached_frame, self.__data_scale) if use_gpu
                  else self.noise.apply(self.__cached_frame * self.__data_scale))
        return DataAndMetadata.new_data_and_metadata(result.data,
            intensity_calibration=result.intensity_calibration, dimensional_calibrations=result.dimensional_calibrations,
            metadata=self.__cached_frame.metadata, timestamp=result.timestamp)

    def get_dimensional_calibrations(self, readout_area: typing.Optional[Geometry.IntRect], binning_shape: typing.Optional[Geometry.IntSize]) -> typing.Sequence[Calibration.Calibration]:
        area = readout_area or Geometry.IntRect(origin=Geometry.IntPoint(), size=self._sensor_dimensions)
        bins = binning_shape or Geometry.IntSize(1, 1)
        raw = self._raw_dimensional_calibrations(area, bins)
        return [Calibration.Calibration(offset=c.offset+(length-1)*c.scale, scale=-c.scale, units=c.units)
                for c, length in zip(raw, (area.height//bins.height, area.width//bins.width))]

    def _raw_dimensional_calibrations(self, readout_area, binning_shape):
        area = readout_area or Geometry.IntRect(origin=Geometry.IntPoint(), size=self._sensor_dimensions)
        bins = binning_shape or Geometry.IntSize(1, 1)
        scale_y = self._tv_pixel_angle * bins.height
        scale_x = self._tv_pixel_angle * bins.width
        offset_y = self._tv_pixel_angle * (area.top + (bins.height-1)/2 - self._sensor_dimensions.height/2)
        offset_x = self._tv_pixel_angle * (area.left + (bins.width-1)/2 - self._sensor_dimensions.width/2)
        dimensional_calibrations = [
            Calibration.Calibration(offset=offset_y, scale=scale_y, units="rad"),
            Calibration.Calibration(offset=offset_x, scale=scale_x, units="rad")
        ]
        return dimensional_calibrations
