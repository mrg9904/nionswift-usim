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
MINIMUM_DEPTH_SLICE_THICKNESS_NM = 0.5

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