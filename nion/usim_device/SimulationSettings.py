"""Performance and accuracy settings for the uSim HAADF model."""

# ----------------------------------------------------------------------
# Depth slicing
# ----------------------------------------------------------------------

# Calculate the depth-slice thickness from the current scan FOV:
#
#     slice thickness = FOV / DEPTH_SLICE_FOV_DIVISOR
#
# Examples:
#     FOV = 50 nm  -> slice = 0.5 nm
#     FOV = 100 nm -> slice = 1.0 nm
#     FOV = 200 nm -> slice = 2.0 nm
# Calculate the requested slice thickness as:
DEPTH_SLICE_FOV_DIVISOR = 100.0

# Prevent excessively small slices from creating too many depth planes.
MINIMUM_DEPTH_SLICE_THICKNESS_NM = 0.1

# Prevent excessively large slices from producing a coarse depth model.
MAXIMUM_DEPTH_SLICE_THICKNESS_NM = 10.0


def calculate_depth_slice_thickness_nm(
    fov_height_nm: float,
    fov_width_nm: float,
) -> float:
    """Calculate and constrain slice thickness from the current FOV."""

    if DEPTH_SLICE_FOV_DIVISOR <= 0.0:
        raise ValueError(
            "DEPTH_SLICE_FOV_DIVISOR must be greater than zero"
        )

    if fov_height_nm <= 0.0 or fov_width_nm <= 0.0:
        raise ValueError(
            "The scan FOV dimensions must be greater than zero"
        )

    if MINIMUM_DEPTH_SLICE_THICKNESS_NM <= 0.0:
        raise ValueError(
            "MINIMUM_DEPTH_SLICE_THICKNESS_NM must be greater than zero"
        )

    if (
        MAXIMUM_DEPTH_SLICE_THICKNESS_NM
        < MINIMUM_DEPTH_SLICE_THICKNESS_NM
    ):
        raise ValueError(
            "MAXIMUM_DEPTH_SLICE_THICKNESS_NM must be greater "
            "than or equal to MINIMUM_DEPTH_SLICE_THICKNESS_NM"
        )

    effective_fov_nm = min(
        fov_height_nm,
        fov_width_nm,
    )

    requested_slice_thickness_nm = (
        effective_fov_nm
        / DEPTH_SLICE_FOV_DIVISOR
    )

    return min(
        max(
            requested_slice_thickness_nm,
            MINIMUM_DEPTH_SLICE_THICKNESS_NM,
        ),
        MAXIMUM_DEPTH_SLICE_THICKNESS_NM,
    )

# ----------------------------------------------------------------------
# STL depth-resolved sample
# ----------------------------------------------------------------------

# STL coordinates are interpreted directly as nanometers.
STL_SAMPLE_FILE_NAME = "ten_random_cuboids_5nm.stl"

# Translate the first cuboid to the scan origin.
# Set both values to 0.0 to retain the original STL coordinates.
STL_SAMPLE_SHIFT_X_NM = -1405.8069605737685
STL_SAMPLE_SHIFT_Y_NM = 40.71017630395872

# Match the intensity definition used by the existing thickness samples:
# 20 nm material thickness produces an ideal HAADF intensity of 1.0.
STL_REFERENCE_THICKNESS_NM = 20.0

# Maximum number of rays processed by trimesh in one batch.
STL_RAY_CHUNK_SIZE = 131072

# ----------------------------------------------------------------------
# STL intensity diagnostics
# ----------------------------------------------------------------------

# If True, normalize the summed depth-plane intensity at every occupied
# XY pixel. This removes projected-thickness contrast while preserving
# the relative distribution of intensity along Z.
#
# Use True only for debugging STL geometry and depth slicing.
STL_NORMALIZE_COLUMN_INTENSITY = True

# Total ideal intensity assigned to every occupied XY pixel after
# column normalization.
STL_NORMALIZED_COLUMN_INTENSITY = 1.0

# ----------------------------------------------------------------------
# Defocus blur
# ----------------------------------------------------------------------

# Minimum probe width in pixels, including the nominal in-focus probe.
MINIMUM_SIGMA_PX = 0.25

# Maximum permitted defocus blur.
#
# This is the most important parameter for preventing very large
# convolution kernels at large defocus.
MAXIMUM_SIGMA_PX = 10.0

# The Gaussian kernel is truncated at:
#
#     radius = GAUSSIAN_TRUNCATE * sigma
#
# 4.0 is more accurate but slower.
# 2.5-3.0 is usually sufficient for autofocus development.
GAUSSIAN_TRUNCATE = 3.0


# ----------------------------------------------------------------------
# Fast large-blur approximation
# ----------------------------------------------------------------------

# If True, replace large Gaussian filters with repeated box filters.
# This substantially improves performance at large defocus.
USE_FAST_BOX_FILTER = True

# Use the fast approximation when either sigma exceeds this value.
FAST_BOX_FILTER_THRESHOLD_PX = 3.0

# Three box-filter passes provide a reasonable Gaussian approximation.
FAST_BOX_FILTER_PASSES = 3

# ----------------------------------------------------------------------
# HAADF noise
# ----------------------------------------------------------------------

# Standard deviation of the additive Gaussian noise applied to the
# simulated HAADF image before multiplication by pixel_time_us.
#
# Suggested values:
#     0.0  -> no noise, useful for geometry debugging
#     0.01 -> very low noise
#     0.05 -> moderate noise
#     0.3  -> original uSim noise level
HAADF_NOISE_FACTOR = 0.0