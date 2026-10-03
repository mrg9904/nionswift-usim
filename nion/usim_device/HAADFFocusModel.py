from __future__ import annotations

import math

import numpy
import numpy.typing
import scipy.ndimage


_NDArray = numpy.typing.NDArray[numpy.float32]


def apply_defocus(
    data: _NDArray,
    *,
    defocus_m: float,
    best_focus_m: float,
    convergence_angle_rad: float,
    pixel_size_y_nm: float,
    pixel_size_x_nm: float,
    minimum_sigma_px: float = 0.25,
    maximum_sigma_px: float = 40.0,
) -> _NDArray:
    """Apply a fast defocus response to a simulated HAADF image.

    The geometrical probe broadening is approximated by

        r = alpha * abs(defocus - best_focus)

    This model is intended for autofocus/control development. It is not
    an atomic-resolution multislice image-formation calculation.
    """

    focus_error_nm = (defocus_m - best_focus_m) * 1e9

    # alpha is dimensionless (rad), so the result remains in nm.
    blur_nm = abs(convergence_angle_rad * focus_error_nm)

    sigma_y_px = math.sqrt(
        minimum_sigma_px**2
        + (blur_nm / max(pixel_size_y_nm, 1e-12)) ** 2
    )

    sigma_x_px = math.sqrt(
        minimum_sigma_px**2
        + (blur_nm / max(pixel_size_x_nm, 1e-12)) ** 2
    )

    # Avoid excessively expensive filtering at extreme defocus values.
    sigma_y_px = min(sigma_y_px, maximum_sigma_px)
    sigma_x_px = min(sigma_x_px, maximum_sigma_px)

    focused_data = scipy.ndimage.gaussian_filter(
        data,
        sigma=(sigma_y_px, sigma_x_px),
        mode="reflect",
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