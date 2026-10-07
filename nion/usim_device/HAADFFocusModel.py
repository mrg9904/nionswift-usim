from __future__ import annotations

import math
import typing
import functools

import numpy
import numpy.typing
import scipy.ndimage
import scipy.fft

from nion.usim_device import SimulationSettings

_NDArray = numpy.typing.NDArray[numpy.float32]


@functools.lru_cache(maxsize=16)
def _frequency_axes(shape: typing.Tuple[int, int]) -> typing.Tuple[
    numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]
]:
    # DCT-II represents reflection at half-pixel boundaries, matching the
    # spatial filter's reflect boundary convention without periodic wrap.
    return (
        (numpy.pi * numpy.arange(shape[0], dtype=numpy.float64) / shape[0]) ** 2,
        (numpy.pi * numpy.arange(shape[1], dtype=numpy.float64) / shape[1]) ** 2,
    )


def _spectral_shape(shape: typing.Tuple[int, int], sigma_y: float, sigma_x: float) -> typing.Tuple[int, int]:
    # Only evaluate frequencies with a non-negligible Gaussian transfer.
    # The tighter support at larger defocus makes repeated focus changes
    # cheaper rather than allocating a full weight image for every slice.
    limit = math.sqrt(-2.0 * math.log(SimulationSettings.FOURIER_TRANSFER_CUTOFF))
    return (
        min(shape[0], max(1, int(math.floor(limit * shape[0] / (numpy.pi * sigma_y))) + 1)) if sigma_y > 0 else shape[0],
        min(shape[1], max(1, int(math.floor(limit * shape[1] / (numpy.pi * sigma_x))) + 1)) if sigma_x > 0 else shape[1],
    )


def _spectral_weight(
    shape: typing.Tuple[int, int], sigma_y: float, sigma_x: float,
    support: typing.Optional[typing.Tuple[int, int]] = None,
) -> _NDArray:
    frequency_y, frequency_x = _frequency_axes(shape)
    if support is not None:
        frequency_y = frequency_y[:support[0]]
        frequency_x = frequency_x[:support[1]]
    # Separable weights avoid an exponential calculation at every pixel.
    weight_y = numpy.exp(-0.5 * (sigma_y * numpy.sqrt(frequency_y)) ** 2)
    weight_x = numpy.exp(-0.5 * (sigma_x * numpy.sqrt(frequency_x)) ** 2)
    return (weight_y[:, None] * weight_x[None, :]).astype(numpy.float32)


class PreparedDepthPlanes(typing.Sequence[typing.Tuple[float, _NDArray]]):
    """Immutable STL planes with a bounded, reusable DCT cache.

    The sample constructs this only after normalization. New geometry or
    slice spacing creates a new collection, invalidating its spectra too.
    Ordinary mutable sequences passed to autofocus are never cached.
    """

    def __init__(self, planes: typing.Sequence[typing.Tuple[float, _NDArray]]) -> None:
        self._planes = tuple(planes)
        for _, plane in self._planes:
            plane.setflags(write=False)
        self._spectra: typing.Dict[int, _NDArray] = {}
        self._cached_bytes = 0

    def __len__(self) -> int:
        return len(self._planes)

    @typing.overload
    def __getitem__(self, index: int) -> typing.Tuple[float, _NDArray]: ...

    @typing.overload
    def __getitem__(self, index: slice) -> typing.Sequence[typing.Tuple[float, _NDArray]]: ...

    def __getitem__(self, index: typing.Union[int, slice]) -> typing.Any:
        return self._planes[index]

    def spectrum(self, index: int) -> _NDArray:
        cached = self._spectra.get(index)
        if cached is not None:
            return cached
        spectrum = scipy.fft.dctn(self._planes[index][1], norm="ortho")
        if self._cached_bytes + spectrum.nbytes <= SimulationSettings.DEPTH_SPECTRUM_CACHE_BYTES:
            spectrum.setflags(write=False)
            self._spectra[index] = spectrum
            self._cached_bytes += spectrum.nbytes
        return spectrum


def _filter_key(sigma_y_px: float, sigma_x_px: float) -> typing.Tuple[typing.Any, ...]:
    """Describe the actual filter, allowing equal depth filters to share work."""
    return ("gaussian", sigma_y_px, sigma_x_px)


def _probe_sigma(
    focus_error_nm: float,
    convergence_angle_rad: float,
    pixel_size_y_nm: float,
    pixel_size_x_nm: float,
    minimum_sigma_px: float,
    maximum_sigma_px: typing.Optional[float] = None,
) -> typing.Tuple[float, float]:
    blur_nm = abs(convergence_angle_rad * focus_error_nm)
    sigmas = (
        math.hypot(minimum_sigma_px, blur_nm / max(pixel_size_y_nm, 1e-12)),
        math.hypot(minimum_sigma_px, blur_nm / max(pixel_size_x_nm, 1e-12)),
    )
    # Preserve an explicitly requested API limit, but impose no default cap.
    if maximum_sigma_px is not None:
        return min(sigmas[0], maximum_sigma_px), min(sigmas[1], maximum_sigma_px)
    return sigmas


def _apply_probe_blur(
    data: _NDArray,
    sigma_y_px: float,
    sigma_x_px: float,
) -> _NDArray:
    """Use spatial filtering near focus and constant-cost spectral blur beyond it."""
    if max(sigma_y_px, sigma_x_px) >= SimulationSettings.FOURIER_BLUR_THRESHOLD_PX:
        spectrum = scipy.fft.dctn(data, norm="ortho")
        spectrum *= _spectral_weight(data.shape, sigma_y_px, sigma_x_px)
        return scipy.fft.idctn(spectrum, norm="ortho").astype(numpy.float32, copy=False)
    return scipy.ndimage.gaussian_filter(
        data, sigma=(sigma_y_px, sigma_x_px), mode="reflect",
        truncate=SimulationSettings.GAUSSIAN_TRUNCATE,
    ).astype(numpy.float32, copy=False)


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
    return_device: bool = False,
) -> typing.Any:
    """Focus and sum HAADF contributions from multiple sample depths."""

    if not depth_planes:
        raise ValueError(
            "depth_planes must contain at least one depth plane"
        )

    if getattr(depth_planes, "_on_gpu", False):
        return depth_planes.render(
            defocus_m=defocus_m, best_focus_m=best_focus_m,
            convergence_angle_rad=convergence_angle_rad,
            pixel_size_y_nm=pixel_size_y_nm, pixel_size_x_nm=pixel_size_x_nm,
            return_device=return_device,
        )

    first_plane = depth_planes[0][1]

    output = numpy.zeros_like(
        first_plane,
        dtype=numpy.float32,
    )

    groups: typing.Dict[typing.Tuple[typing.Any, ...], typing.Tuple[_NDArray, float, float]] = {}
    combined_spectrum: typing.Optional[_NDArray] = None
    for index, (depth_nm, plane_data) in enumerate(depth_planes):
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
        )

        if max(sigma_y, sigma_x) >= SimulationSettings.FOURIER_BLUR_THRESHOLD_PX:
            spectrum = (
                depth_planes.spectrum(index)
                if isinstance(depth_planes, PreparedDepthPlanes)
                else scipy.fft.dctn(plane_data, norm="ortho")
            )
            support = _spectral_shape(plane_data.shape, sigma_y, sigma_x)
            if combined_spectrum is None:
                combined_spectrum = numpy.zeros_like(spectrum)
            combined_spectrum[:support[0], :support[1]] += (
                spectrum[:support[0], :support[1]]
                * _spectral_weight(plane_data.shape, sigma_y, sigma_x, support)
            )
            continue
        key = _filter_key(sigma_y, sigma_x)
        group = groups.get(key)
        if group is None:
            groups[key] = (plane_data.copy(), sigma_y, sigma_x)
        else:
            group[0][:] += plane_data

    # Filtering is linear. Sum large-blur spectra and perform just one
    # inverse transform per image; retain exact sigmas for every depth.
    # Cached sample planes must remain unchanged.
    for plane_data, sigma_y, sigma_x in groups.values():
        output += _apply_probe_blur(plane_data, sigma_y, sigma_x)
    if combined_spectrum is not None:
        output += scipy.fft.idctn(combined_spectrum, norm="ortho")

    return output

def fast_gaussian_filter(
    data: _NDArray,
    sigma_y_px: float,
    sigma_x_px: float,
) -> _NDArray:
    """Compatibility entry point for continuous, radius-independent blur."""
    return _apply_probe_blur(data, sigma_y_px, sigma_x_px)
