"""Settings for interactive control of the uSim microscope."""

# Master switch for all interactive controls.
INTERACTIVE_CONTROLS_ENABLED = True


# ----------------------------------------------------------------------
# Mouse-wheel focus control
# ----------------------------------------------------------------------

ENABLE_MOUSE_WHEEL_FOCUS = True

# Focus change for one mouse-wheel event. uSim stores C10Control in meters,
# but this setting is expressed in nanometers for convenience.
FOCUS_STEP_NM = 1.0

# Change this to -1.0 if the wheel direction feels reversed on your system.
FOCUS_WHEEL_DIRECTION = 1.0

# Safety limits for interactive focus adjustment.
MINIMUM_FOCUS_NM = -1000.0
MAXIMUM_FOCUS_NM = 1000.0


# ----------------------------------------------------------------------
# Double-click stage movement
# ----------------------------------------------------------------------

ENABLE_DOUBLE_CLICK_STAGE_MOVE = True

# Only move the stage while the standard pointer tool is active. This avoids
# interfering with Nion Swift's zoom and graphic-creation tools.
DOUBLE_CLICK_REQUIRES_POINTER_TOOL = True

# Ignore modified double-clicks so that Ctrl/Shift/Alt remain available for
# other Nion Swift operations.
DOUBLE_CLICK_REQUIRES_NO_MODIFIERS = True

# Set either value to -1.0 if an axis moves in the wrong direction on the
# local Nion Swift installation.
STAGE_X_DIRECTION = 1.0
STAGE_Y_DIRECTION = 1.0


# ----------------------------------------------------------------------
# Keyboard microscope controls
# ----------------------------------------------------------------------

ENABLE_KEYBOARD_CONTROLS = True

# Keys are case-insensitive; only tilt keys accept Shift to reverse direction.
FOV_DECREASE_KEY = "r"
FOV_INCREASE_KEY = "e"
DEFOCUS_DECREASE_KEY = "d"
DEFOCUS_INCREASE_KEY = "f"

# R multiplies the FoV by 0.8; E multiplies it by 1.25.
# These values are reciprocal, so one E followed by one R restores the
# original FoV, apart from floating-point rounding.
FOV_ZOOM_IN_FACTOR = 0.8
FOV_ZOOM_OUT_FACTOR = 1.25

MINIMUM_FOV_NM = 5.0

# D decreases C10 and F increases it by this amount. C10 is the value shown
# in the uSim Instrument panel and feeds the effective C10Control used by the
# HAADF focus model. A 1 nm change is usually too small to see, so use 10 nm.
DEFOCUS_STEP_NM = 10.0

# T changes TX, Y changes TY. Shift reverses either direction.
TILT_X_KEY = "t"
TILT_Y_KEY = "y"
TILT_STEP_DEG = 0.1
