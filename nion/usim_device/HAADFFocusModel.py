from __future__ import annotations

import math
import typing

import numpy
import numpy.typing
import scipy.ndimage

from nion.usim_device import SimulationSettings

_NDArray = numpy.typing.NDArray[numpy.float32]


def _make_odd_filter_size(value: float) -> int:
    """Convert a calculated box-filter width to a positive odd integer."""

    filter_size = max(
        1,
        int(round(value)),
    )

    if filter_size % 2 == 0:
        filter_size += 1

    return filter_size


def _filter_key(sigma_y_px: float, sigma_x_px: float) -> typing.Tuple[typing.Any, ...]:
    """Describe the actual filter, allowing equal depth filters to share work."""
    if (
        SimulationSettings.USE_FAST_BOX_FILTER
        and max(sigma_y_px, sigma_x_px) >= SimulationSettings.FAST_BOX_FILTER_THRESHOLD_PX
    ):
        passes = max(1, SimulationSettings.FAST_BOX_FILTER_PASSES)
        return (
            "box",
            _make_odd_filter_size(math.sqrt(12.0 * sigma_y_px**2 / passes + 1.0)),
            _make_odd_filter_size(math.sqrt(12.0 * sigma_x_px**2 / passes + 1.0)),
            passes,
        )
    return ("gaussian", sigma_y_px, sigma_x_px)


def _probe_sigma(
    focus_error_nm: float,
    convergence_angle_rad: float,
    pixel_size_y_nm: float,
    pixel_size_x_nm: float,
    minimum_sigma_px: float,
    maximum_sigma_px: float,
) -> typing.Tuple[float, float]:
    blur_nm = abs(convergence_angle_rad * focus_error_nm)
    return (
        min(math.sqrt(minimum_sigma_px**2 + (blur_nm / max(pixel_size_y_nm, 1e-12)) ** 2), maximum_sigma_px),
        min(math.sqrt(minimum_sigma_px**2 + (blur_nm / max(pixel_size_x_nm, 1e-12)) ** 2), maximum_sigma_px),
    )


def _apply_probe_blur(
    data: _NDArray,
    sigma_y_px: float,
    sigma_x_px: float,
) -> _NDArray:
    """Apply either an exact Gaussian or a fast box-filter approximation."""

    maximum_sigma_px = max(
        sigma_y_px,
        sigma_x_px,
    )

    use_fast_filter = (
        SimulationSettings.USE_FAST_BOX_FILTER
        and maximum_sigma_px
        >= SimulationSettings.FAST_BOX_FILTER_THRESHOLD_PX
    )

    if not use_fast_filter:
        filtered_data = scipy.ndimage.gaussian_filter(
            data,
            sigma=(sigma_y_px, sigma_x_px),
            mode="reflect",
            truncate=SimulationSettings.GAUSSIAN_TRUNCATE,
        )

        return filtered_data.astype(
            numpy.float32,
            copy=False,
        )

    number_of_passes = max(
        1,
        SimulationSettings.FAST_BOX_FILTER_PASSES,
    )

    # The variance of n repeated box filters is matched to the
    # requested Gaussian variance.
    box_height = _make_odd_filter_size(
        math.sqrt(
            12.0 * sigma_y_px**2 / number_of_passes
            + 1.0
        )
    )

    box_width = _make_odd_filter_size(
        math.sqrt(
            12.0 * sigma_x_px**2 / number_of_passes
            + 1.0
        )
    )

    filtered_data = data.astype(
        numpy.float32,
        copy=False,
    )

    for _ in range(number_of_passes):
        filtered_data = scipy.ndimage.uniform_filter(
            filtered_data,
            size=(box_height, box_width),
            mode="reflect",
        )

    return filtered_data.astype(
        numpy.float32,
        copy=False,
    )


def apply_defocus(
    data: _NDArray,
    *,
    defocus_m: float,
    best_focus_m: float,
    convergence_angle_rad: float,
    pixel_size_y_nm: float,
    pixel_size_x_nm: float,
    minimum_sigma_px: typing.Optional[float] = None,
    maximum_sigma_px: typing.Optional[float] = None,
) -> _NDArray:
    """Apply a fast defocus response to a simulated HAADF image.

    The geometrical probe broadening is approximated by

        r = alpha * abs(defocus - best_focus)

    This model is intended for autofocus/control development. It is not
    an atomic-resolution multislice image-formation calculation.
    """
    if minimum_sigma_px is None:
        minimum_sigma_px = (
            SimulationSettings.MINIMUM_SIGMA_PX
        )

    if maximum_sigma_px is None:
        maximum_sigma_px = (
            SimulationSettings.MAXIMUM_SIGMA_PX
        )
    sigma_y_px, sigma_x_px = _probe_sigma(
        (defocus_m - best_focus_m) * 1e9,
        convergence_angle_rad,
        pixel_size_y_nm,
        pixel_size_x_nm,
        minimum_sigma_px,
        maximum_sigma_px,
    )

    focused_data = _apply_probe_blur(
        data,
        sigma_y_px,
        sigma_x_px,
    )

    return focused_data.astype(numpy.float32, copy=False)

def apply_height_dependent_defocus(
    data: _NDArray,
    height_map_nm: _NDArray,
    *,
    defocus_m: float,
    best_focus_m: float,
    convergence_angle_rad: float,
    pixel_size_y_nm: float,
    pixel_size_x_nm: float,
) -> _NDArray:
    """Apply a separate defocus response to each sample height."""

    if data.shape != height_map_nm.shape:
        raise ValueError(
            "data and height_map_nm must have the same shape"
        )

    output = numpy.zeros_like(
        data,
        dtype=numpy.float32,
    )

    # For this test sample, the result is [0, 50, 100].
    height_planes_nm = numpy.unique(height_map_nm)

    for height_nm in height_planes_nm:
        plane_mask = height_map_nm == height_nm

        # Retain only the image intensity belonging to this height.
        plane_data = numpy.where(
            plane_mask,
            data,
            0.0,
        ).astype(
            numpy.float32,
            copy=False,
        )

        # A feature at a higher axial position has a correspondingly
        # shifted best-focus value.
        plane_best_focus_m = (
            best_focus_m
            + float(height_nm) * 1e-9
        )

        focused_plane = apply_defocus(
            plane_data,
            defocus_m=defocus_m,
            best_focus_m=plane_best_focus_m,
            convergence_angle_rad=convergence_angle_rad,
            pixel_size_y_nm=pixel_size_y_nm,
            pixel_size_x_nm=pixel_size_x_nm,
        )

        output += focused_plane

    return output

def apply_depth_planes_defocus(
    depth_planes: typing.Sequence[
        typing.Tuple[float, _NDArray]
    ],
    *,
    defocus_m: float,
    best_focus_m: float,
    convergence_angle_rad: float,
    pixel_size_y_nm: float,
    pixel_size_x_nm: float,
) -> _NDArray:
    """Focus and sum HAADF contributions from multiple sample depths."""

    if not depth_planes:
        raise ValueError(
            "depth_planes must contain at least one depth plane"
        )

    first_plane = depth_planes[0][1]

    output = numpy.zeros_like(
        first_plane,
        dtype=numpy.float32,
    )

    groups: typing.Dict[typing.Tuple[typing.Any, ...], typing.Tuple[_NDArray, float, float]] = {}
    for depth_nm, plane_data in depth_planes:
        # The reference focus corresponds to the common base plane z = 0.
        plane_best_focus_m = (
            best_focus_m
            + depth_nm * 1e-9
        )

        sigma_y, sigma_x = _probe_sigma(
            (defocus_m - plane_best_focus_m) * 1e9,
            convergence_angle_rad,
            pixel_size_y_nm,
            pixel_size_x_nm,
            SimulationSettings.MINIMUM_SIGMA_PX,
            SimulationSettings.MAXIMUM_SIGMA_PX,
        )
        key = _filter_key(sigma_y, sigma_x)
        group = groups.get(key)
        if group is None:
            groups[key] = (plane_data.copy(), sigma_y, sigma_x)
        else:
            group[0][:] += plane_data

    # Gaussian/box filtering is linear. Sum planes with the exact same
    # discrete filter before filtering; do not quantize Gaussian sigmas.
    # Cached sample planes must remain unchanged.
    for plane_data, sigma_y, sigma_x in groups.values():
        output += _apply_probe_blur(plane_data, sigma_y, sigma_x)

    return output

def fast_gaussian_filter(
    data: _NDArray,
    sigma_y_px: float,
    sigma_x_px: float,
) -> _NDArray:
    """Apply a fast Gaussian approximation using three box filters."""

    if max(sigma_y_px, sigma_x_px) <= 3.0:
        return scipy.ndimage.gaussian_filter(
            data,
            sigma=(sigma_y_px, sigma_x_px),
            mode="reflect",
            truncate=3.0,
        )

    box_height = max(
        1,
        int(round(math.sqrt(4.0 * sigma_y_px**2 + 1.0))),
    )

    box_width = max(
        1,
        int(round(math.sqrt(4.0 * sigma_x_px**2 + 1.0))),
    )

    # Use odd filter sizes.
    if box_height % 2 == 0:
        box_height += 1

    if box_width % 2 == 0:
        box_width += 1

    filtered_data = data.astype(
        numpy.float32,
        copy=False,
    )

    # Three box filters approximate one Gaussian filter.
    for _ in range(3):
        filtered_data = scipy.ndimage.uniform_filter(
            filtered_data,
            size=(box_height, box_width),
            mode="reflect",
        )

    return filtered_data
