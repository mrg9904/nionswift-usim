# standard libraries
import abc
import math

from nion.usim_device import EELSModel
import gettext
import logging

import numpy
import numpy.typing
import random
import scipy.ndimage
import typing

from nion.data import Image
from nion.utils import Geometry

_NDArray = numpy.typing.NDArray[typing.Any]


_ = gettext.gettext


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


class Feature:
    def __init__(self, position_m: Geometry.FloatPoint, size_m: Geometry.FloatSize, edges: typing.Sequence[typing.Tuple[int, int]], plasmon_eV: float, plurality: int) -> None:
        self.position_m = position_m
        self.size_m = size_m
        self.edges = edges
        self.plasmon_eV = plasmon_eV
        self.plurality = plurality

    def get_scan_rect_m(self, offset_m: Geometry.FloatPoint, fov_nm: Geometry.FloatSize, center_nm: Geometry.FloatPoint) -> Geometry.FloatRect:
        scan_size_m = Geometry.FloatSize(height=fov_nm.height, width=fov_nm.width) / 1E9
        scan_rect_m = Geometry.FloatRect.from_center_and_size(Geometry.FloatPoint.make(center_nm) / 1E9, scan_size_m)
        scan_rect_m -= offset_m
        return scan_rect_m

    def get_feature_rect_m(self) -> Geometry.FloatRect:
        return Geometry.FloatRect.from_center_and_size(self.position_m, self.size_m)

    def intersects(self, offset_m: Geometry.FloatPoint, fov_nm: Geometry.FloatSize, center_nm: Geometry.FloatPoint, probe_position: Geometry.FloatPoint) -> bool:
        scan_rect_m = self.get_scan_rect_m(offset_m, fov_nm, center_nm)
        feature_rect_m = self.get_feature_rect_m()
        probe_position_m = Geometry.FloatPoint(y=probe_position.y * scan_rect_m.height + scan_rect_m.top, x=probe_position.x * scan_rect_m.width + scan_rect_m.left)
        return scan_rect_m.intersects_rect(feature_rect_m) and feature_rect_m.contains_point(probe_position_m)

    def thickness_at(self, position_m: Geometry.FloatPoint) -> float:
        """Local beam-path length; legacy flat features default to 30 nm."""
        if not self.get_feature_rect_m().contains_point(position_m):
            return 0.0
        return float(getattr(self, "thickness_nm", 30.0))

    @property
    def eels_material(self) -> EELSModel.EELSMaterial:
        return EELSModel.EELSMaterial(edges=tuple(self.edges), plasmon_eV=self.plasmon_eV)

    def plot(self, data: _NDArray, offset_m: Geometry.FloatPoint, fov_nm: Geometry.FloatSize, center_nm: Geometry.FloatPoint, shape: Geometry.IntSize) -> int:
        raise NotImplementedError()


class FlakeFeature(Feature):

    def __init__(self, position_m: Geometry.FloatPoint, size_m: Geometry.FloatSize, edges: typing.Sequence[typing.Tuple[int, int]], plasmon_eV: float, plurality: int) -> None:
        super().__init__(position_m, size_m, edges, plasmon_eV, plurality)

    def plot(self, data: _NDArray, offset_m: Geometry.FloatPoint, fov_nm: Geometry.FloatSize, center_nm: Geometry.FloatPoint, shape: Geometry.IntSize) -> int:
        # TODO: how does center_nm interact with stage position?
        # TODO: take into account feature angle
        # TODO: take into account frame parameters angle
        # TODO: expand features to other shapes than rectangle
        scan_rect_m = self.get_scan_rect_m(offset_m, fov_nm, center_nm)
        feature_rect_m = self.get_feature_rect_m()
        sum = 0
        if scan_rect_m.intersects_rect(feature_rect_m):
            feature_rect_top_px = int(shape[0] * (feature_rect_m.top - scan_rect_m.top) / scan_rect_m.height)
            feature_rect_left_px = int(shape[1] * (feature_rect_m.left - scan_rect_m.left) / scan_rect_m.width)
            feature_rect_height_px = int(shape[0] * feature_rect_m.height / scan_rect_m.height)
            feature_rect_width_px = int(shape[1] * feature_rect_m.width / scan_rect_m.width)
            if feature_rect_top_px < 0:
                feature_rect_height_px += feature_rect_top_px
                feature_rect_top_px = 0
            if feature_rect_left_px < 0:
                feature_rect_width_px += feature_rect_left_px
                feature_rect_left_px = 0
            if feature_rect_top_px + feature_rect_height_px > shape[0]:
                feature_rect_height_px = shape[0] - feature_rect_top_px
            if feature_rect_left_px + feature_rect_width_px > shape[1]:
                feature_rect_width_px = shape[1] - feature_rect_left_px
            feature_rect_origin_px = Geometry.IntPoint(y=feature_rect_top_px, x=feature_rect_left_px)
            feature_rect_size_px = Geometry.IntSize(height=feature_rect_height_px, width=feature_rect_width_px)
            feature_rect_px = Geometry.IntRect(feature_rect_origin_px, feature_rect_size_px)
            data[feature_rect_px.top:feature_rect_px.bottom, feature_rect_px.left:feature_rect_px.right] += 1.0
            sum += (feature_rect_px.bottom - feature_rect_px.top) * (feature_rect_px.right - feature_rect_px.left)
        return sum


class AmorphousBackground(Feature):

    def __init__(self, position_m: Geometry.FloatPoint, size_m: Geometry.FloatSize, edges: typing.Sequence[typing.Tuple[int, int]], plasmon_eV: float, plurality: int) -> None:
        super().__init__(position_m, size_m, edges, plasmon_eV, plurality)
        self.__amorphous = typing.cast(_NDArray, numpy.random.RandomState(1).randn(2048, 2048) * 2 + 1)

    def plot(self, data: _NDArray, offset_m: Geometry.FloatPoint, fov_nm: Geometry.FloatSize, center_nm: Geometry.FloatPoint, shape: Geometry.IntSize) -> int:
        half_size_nm = self.size_m * 1e9 * 0.5

        # calculate destination bounds in nm
        left_nm = -offset_m.x * 1E9 - fov_nm.width / 2
        top_nm = -offset_m.y * 1E9 - fov_nm.height / 2
        right_nm = left_nm + fov_nm.width
        bottom_nm = top_nm + fov_nm.height

        intersection_left_nm = max(left_nm, -half_size_nm.width)
        intersection_top_nm = max(top_nm, -half_size_nm.height)
        intersection_right_nm = min(right_nm, half_size_nm.width)
        intersection_bottom_nm = min(bottom_nm, half_size_nm.height)

        if intersection_left_nm < intersection_right_nm and intersection_top_nm < intersection_bottom_nm:
            src_left = int(self.__amorphous.shape[1] * max((intersection_left_nm + half_size_nm.width) / (half_size_nm.width * 2), 0))
            src_top = int(self.__amorphous.shape[0] * max((intersection_top_nm + half_size_nm.height) / (half_size_nm.height * 2), 0))
            src_right = int(self.__amorphous.shape[1] * min((intersection_right_nm + half_size_nm.width) / (half_size_nm.width * 2), 1))
            src_bottom = int(self.__amorphous.shape[0] * min((intersection_bottom_nm + half_size_nm.height) / (half_size_nm.height * 2), 1))
            dst_left = int(data.shape[1] * max((intersection_left_nm - left_nm) / (right_nm - left_nm), 0))
            dst_top = int(data.shape[0] * max((intersection_top_nm - top_nm) / (bottom_nm - top_nm), 0))
            dst_right = int(data.shape[1] * min((intersection_right_nm - left_nm) / (right_nm - left_nm), 1))
            dst_bottom = int(data.shape[0] * min((intersection_bottom_nm - top_nm) / (bottom_nm - top_nm), 1))

            src = self.__amorphous[src_top:src_bottom, src_left:src_right]
            range_ = numpy.ptp(src)
            if range_ > 0:
                src = 4 * (src - numpy.amin(src)) / range_
                data[dst_top:dst_bottom, dst_left:dst_right] += Image.scaled(src, (dst_bottom - dst_top, dst_right - dst_left))

        return 0


class GoldBallFeature(Feature):

    def __init__(self, position_m: Geometry.FloatPoint, size_m: Geometry.FloatSize, edges: typing.Sequence[typing.Tuple[int, int]], plasmon_eV: float, plurality: int, orientation: float) -> None:
        super().__init__(position_m, size_m, edges, plasmon_eV, plurality)
        self.orientation = orientation

    def plot(self, data: _NDArray, offset_m: Geometry.FloatPoint, fov_nm: Geometry.FloatSize, center_nm: Geometry.FloatPoint, shape: Geometry.IntSize) -> int:
        scan_rect_m = self.get_scan_rect_m(offset_m, fov_nm, center_nm)
        feature_rect_m = self.get_feature_rect_m()
        feature_rect_aspect_ratio = min(feature_rect_m.height, feature_rect_m.width) / max(feature_rect_m.height, feature_rect_m.width)
        feature_rect_m = Geometry.FloatRect.from_center_and_size(feature_rect_m.center, (max(feature_rect_m.size), max(feature_rect_m.size)))
        thickness = min(self.size_m) * 2.5e9
        sum = 0
        if scan_rect_m.intersects_rect(feature_rect_m):
            feature_rect_top_px = int(shape[0] * (feature_rect_m.top - scan_rect_m.top) / scan_rect_m.height)
            feature_rect_left_px = int(shape[1] * (feature_rect_m.left - scan_rect_m.left) / scan_rect_m.width)
            feature_rect_height_px = int(shape[0] * feature_rect_m.height / scan_rect_m.height)
            feature_rect_width_px = int(shape[1] * feature_rect_m.width / scan_rect_m.width)
            if feature_rect_top_px < 0:
                feature_rect_height_px += feature_rect_top_px
                feature_rect_top_px = 0
            if feature_rect_left_px < 0:
                feature_rect_width_px += feature_rect_left_px
                feature_rect_left_px = 0
            if feature_rect_top_px + feature_rect_height_px > shape[0]:
                feature_rect_height_px = shape[0] - feature_rect_top_px
            if feature_rect_left_px + feature_rect_width_px > shape[1]:
                feature_rect_width_px = shape[1] - feature_rect_left_px
            feature_rect_origin_px = Geometry.IntPoint(y=feature_rect_top_px, x=feature_rect_left_px)
            feature_rect_size_px = Geometry.IntSize(height=feature_rect_height_px, width=feature_rect_width_px)
            feature_rect_px = Geometry.IntRect(feature_rect_origin_px, feature_rect_size_px)
            feature_data = data[feature_rect_px.top:feature_rect_px.bottom, feature_rect_px.left:feature_rect_px.right]
            ellipse_array = numpy.zeros_like(feature_data)
            draw_ellipse(ellipse_array, (ellipse_array.shape[0] / 2, ellipse_array.shape[1] / 2, ellipse_array.shape[0] / 3,
                                         feature_rect_aspect_ratio * ellipse_array.shape[1] / 3, self.orientation), color=thickness)
            feature_data += scipy.ndimage.gaussian_filter(ellipse_array, 2.0)
            sum += (feature_rect_px.bottom - feature_rect_px.top) * (feature_rect_px.right - feature_rect_px.left)
        return sum


class Sample(abc.ABC):

    @property
    @abc.abstractmethod
    def title(self) -> str: ...

    @property
    @abc.abstractmethod
    def features(self) -> typing.List[Feature]: ...

    @property
    def initial_view(self) -> typing.Tuple[Geometry.FloatPoint, float]:
        """Stage position (meters) and initial FoV (nm) for this specimen."""
        return Geometry.FloatPoint(), 200.0

    def eels_layers_at(self, position_m: Geometry.FloatPoint) -> typing.List[EELSModel.EELSLayer]:
        """Query specimen geometry in absolute sample coordinates (meters)."""
        layers = []
        for feature in self.features:
            thickness = feature.thickness_at(position_m)
            if thickness > 0:
                layers.append(EELSModel.EELSLayer(thickness, feature.eels_material))
        return layers

    @abc.abstractmethod
    def plot_features(self, data: _NDArray, offset_m: Geometry.FloatPoint, fov_size_nm: Geometry.FloatSize, extra_nm: Geometry.FloatPoint, center_nm: Geometry.FloatPoint, used_size: Geometry.IntSize) -> None: ...


class RectangleFlakeSample(Sample):

    def __init__(self, stage_size_nm: float):
        self.__features: typing.List[Feature] = list()
        sample_size_m = Geometry.FloatSize(height=20 * stage_size_nm / 100, width=20 * stage_size_nm / 100) / 1E9
        feature_percentage = 0.3
        random_state = random.getstate()
        random.seed(1)
        energies = [[(68, 30), (855, 50), (872, 50)], [(29, 15), (1217, 50), (1248, 50)], [(1839, 5), (99, 50)]]  # Ni, Ge, Si
        plasmons = [20, 16.2, 16.8]
        for i in range(100):
            position_m = Geometry.FloatPoint(y=(2 * random.random() - 1.0) * sample_size_m.height, x=(2 * random.random() - 1.0) * sample_size_m.width)
            size_m = feature_percentage * Geometry.FloatSize(height=random.random() * sample_size_m.height, width=random.random() * sample_size_m.width)
            self.__features.append(FlakeFeature(position_m, size_m, energies[i%len(energies)], plasmons[i%len(plasmons)], 4))
        random.setstate(random_state)

    @property
    def title(self) -> str:
        return _("Flake")

    @property
    def features(self) -> typing.List[Feature]:
        return self.__features

    def plot_features(self, data: _NDArray, offset_m: Geometry.FloatPoint, fov_size_nm: Geometry.FloatSize, extra_nm: Geometry.FloatPoint, center_nm: Geometry.FloatPoint, used_size: Geometry.IntSize) -> None:
        for feature in self.__features:
            feature.plot(data, offset_m, fov_size_nm + extra_nm, center_nm, used_size)


class AmorphousSample(Sample):

    def __init__(self, stage_size_nm: float) -> None:
        self.__amorphous = typing.cast(_NDArray, numpy.random.RandomState(1).randn(2048, 2048)) * 2 + 1
        sample_size_m = Geometry.FloatSize(height=stage_size_nm, width=stage_size_nm) / 1E9
        self.__features: typing.List[Feature] = list()
        self.__features.append(AmorphousBackground(Geometry.FloatPoint(), sample_size_m, [(284, 30)], 25., 4))

    @property
    def title(self) -> str:
        return "Amorphous"

    @property
    def features(self) -> typing.List[Feature]:
        return self.__features

    def plot_features(self, data: _NDArray, offset_m: Geometry.FloatPoint, fov_size_nm: Geometry.FloatSize, extra_nm: Geometry.FloatPoint, center_nm: Geometry.FloatPoint, used_size: Geometry.IntSize) -> None:
        for feature in self.__features:
            feature.plot(data, offset_m, fov_size_nm + extra_nm, center_nm, used_size)


class CombinedTestSample(Sample):
    def __init__(self, stage_size_nm: float):
        self.__features: typing.List[Feature] = list()
        self.__last_plot_settings: typing.Any = None
        self.__last_plot: typing.Optional[_NDArray] = None
        sample_size_m = Geometry.FloatSize(height=stage_size_nm, width=stage_size_nm) / 1E9
        feature_max_size_m = 20e-9
        feature_min_size_m = 5e-9
        feature_aspect_ratio = 0.7
        random_state = random.getstate()
        random.seed(1)
        counter = 0
        # We map positions to a rather small grid to reduce the number of overlapping gold balls. This does not take into
        # account the gold ball sizes, but essentially gives each one a certain amount of space.
        position_map: _NDArray = numpy.zeros((120, 120), dtype=numpy.uint8)
        while (i := len(self.__features)) < 10000:
            if counter > 50000:
                logging.warning(f'Placing all features on the CTS sample took to long. Stopping after {i} features.')
                break
            counter += 1
            position_m = Geometry.FloatPoint(y=(2 * random.random() - 1.0) * sample_size_m.height, x=(2 * random.random() - 1.0) * sample_size_m.width)
            position_map_position = (int((position_m.y / sample_size_m.height + 1.0) * 0.5 * position_map.shape[0]),
                                     int((position_m.x / sample_size_m.width + 1.0) * 0.5 * position_map.shape[1]))
            if position_map[position_map_position] != 0:
                continue
            long_dimension_size = random.random() * (feature_max_size_m - feature_min_size_m) + feature_min_size_m
            # Aspect ratio is the maximum we allow, so add 1-maximum to the maximum and randomize the first part which
            # gives random nuzmbers between aspect_ratio and 1
            aspect_ratio = random.random() * (1 - feature_aspect_ratio) + feature_aspect_ratio
            short_dimension_size = aspect_ratio * long_dimension_size
            size_m = Geometry.FloatSize(height=long_dimension_size, width=short_dimension_size)
            # Random rotation between 0 and 180 degrees
            rotation = random.random() * numpy.pi
            new_feature = GoldBallFeature(position_m, size_m, [(54, 30), (83, 40), (2206, 300), (2291, 300)], 15., 4, rotation)
            self.__features.append(new_feature)
            position_map[position_map_position] = 1
        self.__features.append(AmorphousBackground(Geometry.FloatPoint(), sample_size_m, [(284, 30)], 25., 4))
        random.setstate(random_state)

    @property
    def title(self) -> str:
        return _("CTS")

    @property
    def features(self) -> typing.List[Feature]:
        return self.__features

    def plot_features(self, data: _NDArray, offset_m: Geometry.FloatPoint, fov_size_nm: Geometry.FloatSize, extra_nm: Geometry.FloatPoint, center_nm: Geometry.FloatPoint, used_size: Geometry.IntSize) -> None:
        plot_settings = (data.shape, offset_m.as_tuple(), fov_size_nm.as_tuple(), extra_nm.as_tuple(), center_nm.as_tuple(), used_size.as_tuple())
        if self.__last_plot is None or plot_settings != self.__last_plot_settings:
            for feature in self.__features:
                feature.plot(data, offset_m, fov_size_nm + extra_nm, center_nm, used_size)
            self.__last_plot_settings = plot_settings
            self.__last_plot = data.copy()
        else:
            data[:] = self.__last_plot

class HeightBlockFeature(FlakeFeature):
    """A rectangular sample feature with a defined axial height."""

    def __init__(
        self,
        position_m: Geometry.FloatPoint,
        size_m: Geometry.FloatSize,
        height_nm: float,
    ) -> None:
        super().__init__(
            position_m,
            size_m,
            [(68, 30), (855, 50), (872, 50)],
            20.0,
            4,
        )

        self.height_nm = height_nm

    def plot_height(
        self,
        height_map_nm: _NDArray,
        offset_m: Geometry.FloatPoint,
        fov_nm: Geometry.FloatSize,
        center_nm: Geometry.FloatPoint,
        shape: Geometry.IntSize,
    ) -> None:
        """Draw this feature into the sample height map."""

        block_mask = numpy.zeros_like(
            height_map_nm,
            dtype=numpy.float32,
        )

        # Use the existing FlakeFeature plotting method so that the
        # HAADF projection and height map use identical coordinates.
        self.plot(
            block_mask,
            offset_m,
            fov_nm,
            center_nm,
            shape,
        )

        inside_block = block_mask > 0

        # If two blocks overlap, retain the uppermost surface.
        height_map_nm[inside_block] = numpy.maximum(
            height_map_nm[inside_block],
            self.height_nm,
        )

class ThreeHeightBlocksSample(Sample):
    """A sample containing three blocks at different axial heights."""

    def __init__(self, stage_size_nm: float) -> None:
        self.__features: typing.List[Feature] = list()

        # Retain the standard Sample constructor interface.
        _ = stage_size_nm

        # All three blocks are 40 nm by 40 nm.
        block_size_m = Geometry.FloatSize(
            height=40e-9,
            width=40e-9,
        )

        # Each tuple contains:
        # (horizontal position in nm, height in nm)
        block_definitions = (
            (-60.0, 0.0),
            (0.0, 50.0),
            (60.0, 100.0),
        )

        for x_position_nm, height_nm in block_definitions:
            position_m = Geometry.FloatPoint(
                y=0.0,
                x=x_position_nm * 1e-9,
            )

            self.__features.append(
                HeightBlockFeature(
                    position_m=position_m,
                    size_m=block_size_m,
                    height_nm=height_nm,
                )
            )

    @property
    def title(self) -> str:
        return _("Three Height Blocks")

    @property
    def features(self) -> typing.List[Feature]:
        return self.__features

    def plot_features(
        self,
        data: _NDArray,
        offset_m: Geometry.FloatPoint,
        fov_size_nm: Geometry.FloatSize,
        extra_nm: Geometry.FloatPoint,
        center_nm: Geometry.FloatPoint,
        used_size: Geometry.IntSize,
    ) -> None:
        """Generate the ideal two-dimensional HAADF projection."""

        for feature in self.__features:
            feature.plot(
                data,
                offset_m,
                fov_size_nm + extra_nm,
                center_nm,
                used_size,
            )

    def plot_height_map(
        self,
        height_map_nm: _NDArray,
        offset_m: Geometry.FloatPoint,
        fov_size_nm: Geometry.FloatSize,
        extra_nm: Geometry.FloatPoint,
        center_nm: Geometry.FloatPoint,
        used_size: Geometry.IntSize,
    ) -> None:
        """Generate the height map for the three blocks."""

        height_map_nm.fill(0.0)

        for feature in self.__features:
            assert isinstance(feature, HeightBlockFeature)

            feature.plot_height(
                height_map_nm,
                offset_m,
                fov_size_nm + extra_nm,
                center_nm,
                used_size,
            )

class ThicknessBlockFeature(FlakeFeature):
    """A rectangular feature with a defined specimen thickness."""

    def __init__(
        self,
        position_m: Geometry.FloatPoint,
        size_m: Geometry.FloatSize,
        thickness_nm: float,
    ) -> None:
        super().__init__(
            position_m,
            size_m,
            [(68, 30), (855, 50), (872, 50)],
            20.0,
            4,
        )

        self.thickness_nm = thickness_nm

    def create_mask(
        self,
        offset_m: Geometry.FloatPoint,
        fov_nm: Geometry.FloatSize,
        center_nm: Geometry.FloatPoint,
        shape: Geometry.IntSize,
    ) -> _NDArray:
        """Create a binary mask using the existing FlakeFeature geometry."""

        mask = numpy.zeros(
            (shape.height, shape.width),
            dtype=numpy.float32,
        )

        super().plot(
            mask,
            offset_m,
            fov_nm,
            center_nm,
            shape,
        )

        return typing.cast(_NDArray, mask)

class ThreeThicknessBlocksSample(Sample):
    """Three blocks sharing one base plane but having different thicknesses."""

    def __init__(self, stage_size_nm: float) -> None:
        self.__features: typing.List[Feature] = list()

        _ = stage_size_nm

        # Keep the same projected size so that only thickness changes.
        block_size_m = Geometry.FloatSize(
            height=40e-9,
            width=40e-9,
        )

        # Each tuple contains:
        # (horizontal position in nm, thickness in nm)
        block_definitions = (
            (-60.0, 20.0),
            (0.0, 50.0),
            (60.0, 100.0),
        )

        for x_position_nm, thickness_nm in block_definitions:
            position_m = Geometry.FloatPoint(
                y=0.0,
                x=x_position_nm * 1e-9,
            )

            self.__features.append(
                ThicknessBlockFeature(
                    position_m=position_m,
                    size_m=block_size_m,
                    thickness_nm=thickness_nm,
                )
            )

    @property
    def title(self) -> str:
        return _("Three Thickness Blocks")

    @property
    def features(self) -> typing.List[Feature]:
        return self.__features

    def plot_features(
        self,
        data: _NDArray,
        offset_m: Geometry.FloatPoint,
        fov_size_nm: Geometry.FloatSize,
        extra_nm: Geometry.FloatPoint,
        center_nm: Geometry.FloatPoint,
        used_size: Geometry.IntSize,
    ) -> None:
        """Generate a thickness-dependent ideal HAADF projection."""

        reference_thickness_nm = 20.0

        for feature in self.__features:
            assert isinstance(
                feature,
                ThicknessBlockFeature,
            )

            mask = feature.create_mask(
                offset_m,
                fov_size_nm + extra_nm,
                center_nm,
                used_size,
            )

            # Thin-specimen approximation:
            # HAADF intensity is proportional to specimen thickness.
            data += (
                mask
                * feature.thickness_nm
                / reference_thickness_nm
            )

    def generate_depth_planes(
        self,
        offset_m: Geometry.FloatPoint,
        fov_size_nm: Geometry.FloatSize,
        extra_nm: Geometry.FloatPoint,
        center_nm: Geometry.FloatPoint,
        used_size: Geometry.IntSize,
        slice_thickness_nm: float,
    ) -> typing.List[typing.Tuple[float, _NDArray]]:
        """Divide the sample thickness into discrete axial slices.

        Returns a list of:
            (depth_nm, HAADF contribution at that depth)
        """
        if slice_thickness_nm <= 0.0:
            raise ValueError(
                "slice_thickness_nm must be greater than zero"
            )

        reference_thickness_nm = 20.0

        depth_plane_dictionary: typing.Dict[
            float,
            _NDArray,
        ] = dict()

        for feature in self.__features:
            assert isinstance(
                feature,
                ThicknessBlockFeature,
            )

            mask = feature.create_mask(
                offset_m,
                fov_size_nm + extra_nm,
                center_nm,
                used_size,
            )

            lower_depth_nm = 0.0

            while lower_depth_nm < feature.thickness_nm:
                current_slice_nm = min(
                    slice_thickness_nm,
                    feature.thickness_nm - lower_depth_nm,
                )

                center_depth_nm = (
                    lower_depth_nm
                    + current_slice_nm * 0.5
                )

                # Rounded key avoids tiny floating-point differences.
                depth_key_nm = round(center_depth_nm, 9)

                if depth_key_nm not in depth_plane_dictionary:
                    depth_plane_dictionary[depth_key_nm] = (
                        numpy.zeros(
                            (
                                used_size.height,
                                used_size.width,
                            ),
                            dtype=numpy.float32,
                        )
                    )

                # Each slice contributes according to its own thickness.
                depth_plane_dictionary[depth_key_nm] += (
                    mask
                    * current_slice_nm
                    / reference_thickness_nm
                )

                lower_depth_nm += current_slice_nm

        depth_planes = [
            (
                depth_nm,
                depth_plane_dictionary[depth_nm],
            )
            for depth_nm in sorted(
                depth_plane_dictionary.keys()
            )
        ]

        return depth_planes

class SphericalParticleFeature(Feature):
    """A spherical particle resting on the reference sample plane."""

    def __init__(
        self,
        position_m: Geometry.FloatPoint,
        radius_nm: float,
        base_height_nm: float = 0.0,
    ) -> None:
        diameter_m = 2.0 * radius_nm * 1e-9

        super().__init__(
            position_m=position_m,
            size_m=Geometry.FloatSize(
                height=diameter_m,
                width=diameter_m,
            ),
            edges=[
                (68, 30),
                (855, 50),
                (872, 50),
            ],
            plasmon_eV=20.0,
            plurality=4,
        )

        self.radius_nm = radius_nm
        self.base_height_nm = base_height_nm

        # A thickness of 20 nm corresponds to an HAADF intensity of 1.
        self.reference_thickness_nm = 20.0

    def thickness_at(self, position_m: Geometry.FloatPoint) -> float:
        radial_squared = ((position_m.x - self.position_m.x) * 1e9)**2 + ((position_m.y - self.position_m.y) * 1e9)**2
        return 2 * math.sqrt(max(0.0, self.radius_nm**2 - radial_squared))

    def _radial_distance_squared_nm(
        self,
        offset_m: Geometry.FloatPoint,
        fov_nm: Geometry.FloatSize,
        center_nm: Geometry.FloatPoint,
        shape: Geometry.IntSize,
    ) -> _NDArray:
        """Calculate the squared radial distance from the sphere center."""

        scan_rect_m = self.get_scan_rect_m(
            offset_m,
            fov_nm,
            center_nm,
        )

        y_coordinates_m = (
            scan_rect_m.top
            + (
                numpy.arange(
                    shape.height,
                    dtype=numpy.float64,
                )
                + 0.5
            )
            * scan_rect_m.height
            / shape.height
        )

        x_coordinates_m = (
            scan_rect_m.left
            + (
                numpy.arange(
                    shape.width,
                    dtype=numpy.float64,
                )
                + 0.5
            )
            * scan_rect_m.width
            / shape.width
        )

        y_distance_nm = (
            y_coordinates_m - self.position_m.y
        ) * 1e9

        x_distance_nm = (
            x_coordinates_m - self.position_m.x
        ) * 1e9

        radial_distance_squared_nm = (
            y_distance_nm[:, numpy.newaxis] ** 2
            + x_distance_nm[numpy.newaxis, :] ** 2
        )

        return typing.cast(
            _NDArray,
            radial_distance_squared_nm,
        )

    def plot(
        self,
        data: _NDArray,
        offset_m: Geometry.FloatPoint,
        fov_nm: Geometry.FloatSize,
        center_nm: Geometry.FloatPoint,
        shape: Geometry.IntSize,
    ) -> int:
        """Generate the ideal projected HAADF intensity of the sphere."""

        radial_distance_squared_nm = (
            self._radial_distance_squared_nm(
                offset_m,
                fov_nm,
                center_nm,
                shape,
            )
        )

        inside_sphere = (
            radial_distance_squared_nm
            <= self.radius_nm**2
        )

        thickness_map_nm = numpy.zeros(
            (shape.height, shape.width),
            dtype=numpy.float32,
        )

        thickness_map_nm[inside_sphere] = (
            2.0
            * numpy.sqrt(
                self.radius_nm**2
                - radial_distance_squared_nm[inside_sphere]
            )
        )

        # Thin-specimen approximation: HAADF intensity is proportional
        # to the projected specimen thickness.
        data += (
            thickness_map_nm
            / self.reference_thickness_nm
        )

        return int(numpy.count_nonzero(inside_sphere))

    def generate_depth_planes(
        self,
        offset_m: Geometry.FloatPoint,
        fov_nm: Geometry.FloatSize,
        center_nm: Geometry.FloatPoint,
        shape: Geometry.IntSize,
        slice_thickness_nm: float,
    ) -> typing.List[typing.Tuple[float, _NDArray]]:
        """Divide the spherical particle into axial slices."""

        if slice_thickness_nm <= 0.0:
            raise ValueError(
                "slice_thickness_nm must be greater than zero"
            )

        radial_distance_squared_nm = (
            self._radial_distance_squared_nm(
                offset_m,
                fov_nm,
                center_nm,
                shape,
            )
        )

        depth_planes: typing.List[
            typing.Tuple[float, _NDArray]
        ] = list()

        sphere_diameter_nm = 2.0 * self.radius_nm
        lower_local_depth_nm = 0.0

        while lower_local_depth_nm < sphere_diameter_nm:
            current_slice_nm = min(
                slice_thickness_nm,
                sphere_diameter_nm - lower_local_depth_nm,
            )

            center_local_depth_nm = (
                lower_local_depth_nm
                + current_slice_nm * 0.5
            )

            # The sphere center is located one radius above its base.
            distance_from_sphere_center_nm = (
                center_local_depth_nm
                - self.radius_nm
            )

            cross_section_radius_squared_nm = (
                self.radius_nm**2
                - distance_from_sphere_center_nm**2
            )

            # Numerical roundoff could produce a very small negative value.
            cross_section_radius_squared_nm = max(
                cross_section_radius_squared_nm,
                0.0,
            )

            inside_cross_section = (
                radial_distance_squared_nm
                <= cross_section_radius_squared_nm
            )

            plane_data = numpy.zeros(
                (shape.height, shape.width),
                dtype=numpy.float32,
            )

            plane_data[inside_cross_section] = (
                current_slice_nm
                / self.reference_thickness_nm
            )

            absolute_depth_nm = (
                self.base_height_nm
                + center_local_depth_nm
            )

            depth_planes.append(
                (
                    absolute_depth_nm,
                    plane_data,
                )
            )

            lower_local_depth_nm += current_slice_nm

        return depth_planes


class SphericalParticleSample(Sample):
    """A single spherical particle with a spherical thickness profile."""

    def __init__(self, stage_size_nm: float) -> None:
        _ = stage_size_nm

        self.__features: typing.List[Feature] = [
            SphericalParticleFeature(
                position_m=Geometry.FloatPoint(
                    y=0.0,
                    x=0.0,
                ),
                radius_nm=50.0,
                base_height_nm=0.0,
            )
        ]

    @property
    def title(self) -> str:
        return _("Spherical Particle")

    @property
    def features(self) -> typing.List[Feature]:
        return self.__features

    def plot_features(
        self,
        data: _NDArray,
        offset_m: Geometry.FloatPoint,
        fov_size_nm: Geometry.FloatSize,
        extra_nm: Geometry.FloatPoint,
        center_nm: Geometry.FloatPoint,
        used_size: Geometry.IntSize,
    ) -> None:
        """Generate the ideal projected HAADF image."""

        for feature in self.__features:
            feature.plot(
                data,
                offset_m,
                fov_size_nm + extra_nm,
                center_nm,
                used_size,
            )

    def generate_depth_planes(
        self,
        offset_m: Geometry.FloatPoint,
        fov_size_nm: Geometry.FloatSize,
        extra_nm: Geometry.FloatPoint,
        center_nm: Geometry.FloatPoint,
        used_size: Geometry.IntSize,
        slice_thickness_nm: float,
    ) -> typing.List[typing.Tuple[float, _NDArray]]:
        """Generate the depth-resolved spherical particle."""

        feature = self.__features[0]

        assert isinstance(
            feature,
            SphericalParticleFeature,
        )

        return feature.generate_depth_planes(
            offset_m,
            fov_size_nm + extra_nm,
            center_nm,
            used_size,
            slice_thickness_nm=slice_thickness_nm,
        )