"""Performance and accuracy settings for the uSim HAADF model."""

# ----------------------------------------------------------------------
# Depth slicing
# ----------------------------------------------------------------------

# Thickness of each axial sample slice.
#
# Smaller values:
#     More accurate depth integration, but more image filters per frame.
#
# For a 100 nm particle:
#     1.0 nm -> 100 depth planes
#     2.0 nm -> 50 depth planes
#     5.0 nm -> 20 depth planes
DEPTH_SLICE_THICKNESS_NM = 5.0


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