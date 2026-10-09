"""Interactive mouse and keyboard controls for uSim scan and Ronchigram displays.

Controls
--------
Double click
    Move the clicked specimen position to the image center.
R / E
    Decrease / increase the current scan field of view.
D / F
    Decrease / increase the instrument defocus.
T / Y (Shift reverses direction)
    Increase stage TX / TY by the configured degree step.

Nion Swift 16.16.2 exposes display-panel key events, but it does not expose
image mouse-wheel or image double-click events to plug-ins. This module hooks
the two ImageCanvasItem methods while uSim is running and restores the exact
original methods when uSim stops.
"""

from __future__ import annotations

import logging
import math
import typing

from nion.swift import DisplayPanel
from nion.swift import ImageCanvasItem
from nion.ui import UserInterface
from nion.utils import Geometry
from nion.utils import Event

from . import InteractiveControlSettings


_Method = typing.Callable[..., bool]


class InteractiveControlManager:
    """Connect Nion Swift display events to the uSim controls."""

    def __init__(self, instrument: typing.Any, scan_module: typing.Any) -> None:
        self.__instrument = instrument
        self.__scan_module = scan_module
        self.__key_pressed_listener: typing.Optional[Event.EventListener] = (
            DisplayPanel.DisplayPanelManager()
            .key_pressed_event.listen(self.handle_key_pressed)
        )

    def close(self) -> None:
        if self.__key_pressed_listener is not None:
            self.__key_pressed_listener.close()
            self.__key_pressed_listener = None

    @property
    def __scan_hardware_source(self) -> typing.Any:
        # nionswift-instrumentation adds hardware_source to the registered
        # scan module. Resolve it at event time so plug-in initialization order
        # does not matter.
        return getattr(self.__scan_module, "hardware_source", None)

    @property
    def __ronchigram_camera(self) -> typing.Any:
        return getattr(getattr(self.__instrument, "value_manager", None), "ronchigram_camera", None)

    def __display_kind(self, display_panel: typing.Any) -> typing.Optional[str]:
        """Identify this instrument's scan/camera data by source ID."""

        scan_hardware_source = self.__scan_hardware_source
        if display_panel is None:
            return None

        data_item = getattr(display_panel, "data_item", None)
        if data_item is None:
            return None

        metadata = getattr(data_item, "metadata", None)
        if not isinstance(metadata, dict):
            return None

        scan_metadata = metadata.get("scan", dict())
        hardware_metadata = metadata.get("hardware_source", dict())
        hardware_source_id = getattr(
            scan_hardware_source,
            "hardware_source_id",
            None,
        )

        camera_id = getattr(self.__ronchigram_camera, "camera_id", None)
        camera_metadata = metadata.get("camera", {})
        if camera_id is not None and any(isinstance(section, dict) and
                section.get("hardware_source_id") == camera_id
                for section in (hardware_metadata, camera_metadata)):
            return "ronchigram"
        if hardware_source_id is not None and ((
            isinstance(scan_metadata, dict)
            and scan_metadata.get("hardware_source_id") == hardware_source_id
        ) or (
            isinstance(hardware_metadata, dict)
            and hardware_metadata.get("hardware_source_id") == hardware_source_id
        )):
            return "scan"
        return None

    def handle_mouse_wheel(
        self,
        image_canvas_item: typing.Any,
        dx: int,
        dy: int,
        is_horizontal: bool,
    ) -> bool:
        """Adjust defocus over this instrument's scan or Ronchigram image."""

        if not InteractiveControlSettings.INTERACTIVE_CONTROLS_ENABLED:
            return False

        if not InteractiveControlSettings.ENABLE_MOUSE_WHEEL_FOCUS:
            return False

        if is_horizontal or dy == 0:
            return False

        display_panel = getattr(image_canvas_item, "delegate", None)

        if self.__display_kind(display_panel) is None:
            return False

        wheel_sign = 1.0 if dy > 0 else -1.0

        delta_defocus_nm = (
            wheel_sign
            * InteractiveControlSettings.FOCUS_WHEEL_DIRECTION
            * InteractiveControlSettings.FOCUS_STEP_NM
        )

        # Use the same C10-writing function as the D/F keyboard controls.
        return self.__change_defocus(delta_defocus_nm)

    def handle_double_click(
        self,
        image_canvas_item: typing.Any,
        x: int,
        y: int,
        modifiers: UserInterface.KeyboardModifiers,
    ) -> bool:
        """Move the double-clicked specimen position to the scan center."""

        if not InteractiveControlSettings.INTERACTIVE_CONTROLS_ENABLED:
            return False
        if not InteractiveControlSettings.ENABLE_DOUBLE_CLICK_STAGE_MOVE:
            return False

        display_panel = getattr(image_canvas_item, "delegate", None)
        display_kind = self.__display_kind(display_panel)
        if display_kind is None:
            return False

        if (
            InteractiveControlSettings.DOUBLE_CLICK_REQUIRES_POINTER_TOOL
            and getattr(display_panel, "tool_mode", None) != "pointer"
        ):
            return False

        if (
            InteractiveControlSettings.DOUBLE_CLICK_REQUIRES_NO_MODIFIERS
            and modifiers.any_modifier
        ):
            return False

        mouse_mapping = getattr(image_canvas_item, "mouse_mapping", None)
        if mouse_mapping is None:
            return False

        image_position = mouse_mapping.map_point_widget_to_image(
            Geometry.FloatPoint(y=y, x=x)
        )

        data_item = getattr(display_panel, "data_item", None)
        data_shape = getattr(data_item, "data_shape", None)
        if data_shape is None or len(data_shape) != 2:
            return False

        image_height = int(data_shape[0])
        image_width = int(data_shape[1])
        if image_height <= 0 or image_width <= 0:
            return False

        if not (0 <= image_position.x < image_width and 0 <= image_position.y < image_height):
            return False
        if display_kind == "ronchigram":
            camera = self.__ronchigram_camera
            # Runtime cameras come from nion.device_kit, not the legacy local
            # CameraDevice module. Use their public simulator interface.
            mapping = getattr(getattr(camera, "simulator", None), "stage_displacement_for_pixel", None)
            if not callable(mapping):
                return False
            delta = mapping(image_position, data_shape)
            if delta is None:
                return False
            return self.__move_stage(delta*1e9)

        scan_hardware_source = self.__scan_hardware_source
        if scan_hardware_source is None:
            return False

        # Use the currently selected profile as the single FoV source of
        # truth. This keeps double-click stage movement consistent with the
        # FoV used by keyboard zoom and automated acquisitions.
        profile_index = scan_hardware_source.selected_profile_index
        frame_parameters = scan_hardware_source.get_frame_parameters(
            profile_index
        )
        fov_size_nm = Geometry.FloatSize.make(
            frame_parameters.fov_size_nm
        )

        # Pixel centers are used so that the center pixel maps as closely as
        # possible to zero displacement for both odd and even image sizes.
        normalized_x = (image_position.x + 0.5) / image_width - 0.5
        normalized_y = (image_position.y + 0.5) / image_height - 0.5

        image_delta_nm = Geometry.FloatPoint(
            y=normalized_y * fov_size_nm.height,
            x=normalized_x * fov_size_nm.width,
        )
        specimen_delta_nm = image_delta_nm.rotate(
            frame_parameters.rotation_rad
        )

        return self.__move_stage(specimen_delta_nm)

    def __move_stage(self, specimen_delta_nm: Geometry.FloatPoint) -> bool:

        stage_position_m = self.__instrument.get_value_2d(
            "stage_position_m"
        )
        new_stage_position_m = Geometry.FloatPoint(
            y=(
                stage_position_m.y
                - InteractiveControlSettings.STAGE_Y_DIRECTION
                * specimen_delta_nm.y
                * 1e-9
            ),
            x=(
                stage_position_m.x
                - InteractiveControlSettings.STAGE_X_DIRECTION
                * specimen_delta_nm.x
                * 1e-9
            ),
        )

        if not self.__instrument.set_value_2d(
            "stage_position_m",
            new_stage_position_m,
        ):
            return False

        logging.info(
            "uSim stage: x=%.3f nm, y=%.3f nm",
            new_stage_position_m.x * 1e9,
            new_stage_position_m.y * 1e9,
        )
        return True

    def handle_key_pressed(
        self,
        display_panel: typing.Any,
        key: UserInterface.Key,
    ) -> bool:
        """Handle E/R scan FoV, D/F focus and T/Y tilt in either display."""

        if not InteractiveControlSettings.INTERACTIVE_CONTROLS_ENABLED:
            return False
        if not InteractiveControlSettings.ENABLE_KEYBOARD_CONTROLS:
            return False
        if self.__display_kind(display_panel) is None:
            return False

        modifiers = key.modifiers
        if modifiers.control or modifiers.alt or getattr(modifiers, "meta", False):
            return False

        key_text = (key.text or "").lower()
        if key_text in (InteractiveControlSettings.TILT_X_KEY, InteractiveControlSettings.TILT_Y_KEY):
            tilt = self.__instrument.get_value_2d("stage_tilt_rad")
            delta = math.radians(InteractiveControlSettings.TILT_STEP_DEG) * (-1 if modifiers.shift else 1)
            new_tilt = Geometry.FloatPoint(y=tilt.y + (delta if key_text == InteractiveControlSettings.TILT_Y_KEY else 0),
                                          x=tilt.x + (delta if key_text == InteractiveControlSettings.TILT_X_KEY else 0))
            return bool(self.__instrument.set_value_2d("stage_tilt_rad", new_tilt))
        if modifiers.shift:
            return False

        if key_text == InteractiveControlSettings.DEFOCUS_DECREASE_KEY:
            return self.__change_defocus(
                -InteractiveControlSettings.DEFOCUS_STEP_NM
            )

        if key_text == InteractiveControlSettings.DEFOCUS_INCREASE_KEY:
            return self.__change_defocus(
                InteractiveControlSettings.DEFOCUS_STEP_NM
            )

        if key_text == InteractiveControlSettings.FOV_DECREASE_KEY:
            factor = InteractiveControlSettings.FOV_ZOOM_IN_FACTOR
        elif key_text == InteractiveControlSettings.FOV_INCREASE_KEY:
            factor = InteractiveControlSettings.FOV_ZOOM_OUT_FACTOR
        else:
            return False

        scan_hardware_source = self.__scan_hardware_source
        if scan_hardware_source is None:
            return False

        # Read the FoV from the currently selected scan profile. Do not use
        # transient current frame parameters because they can be discarded
        # when scanning stops or restarts.
        profile_index = scan_hardware_source.selected_profile_index
        frame_parameters = scan_hardware_source.get_frame_parameters(
            profile_index
        )
        new_fov_nm = max(
            InteractiveControlSettings.MINIMUM_FOV_NM,
            frame_parameters.fov_nm * factor,
        )

        if math.isclose(new_fov_nm, frame_parameters.fov_nm):
            return True

        frame_parameters.fov_nm = new_fov_nm

        # Write the new FoV back to the same profile. Updating the selected
        # profile also synchronizes the current scan parameters.
        scan_hardware_source.set_frame_parameters(
            profile_index,
            frame_parameters,
        )
        logging.info("uSim FoV: %.3f nm", new_fov_nm)
        return True

    def __change_defocus(self, delta_nm: float) -> bool:
        try:
            current_defocus_m = float(
                self.__instrument.GetVal("C10")
            )
        except (KeyError, TypeError, ValueError):
            return False

        current_defocus_nm = current_defocus_m * 1e9
        new_defocus_nm = min(
            InteractiveControlSettings.MAXIMUM_FOCUS_NM,
            max(
                InteractiveControlSettings.MINIMUM_FOCUS_NM,
                current_defocus_nm + delta_nm,
            ),
        )

        if not self.__instrument.SetVal(
            "C10",
            new_defocus_nm * 1e-9,
        ):
            return False

        effective_defocus_nm = (
            self.__instrument.GetVal("C10Control") * 1e9
        )
        logging.info(
            "uSim defocus: C10=%.3f nm, effective=%.3f nm",
            new_defocus_nm,
            effective_defocus_nm,
        )
        return True


_manager: typing.Optional[InteractiveControlManager] = None
_original_wheel_changed: typing.Optional[_Method] = None
_original_mouse_double_clicked: typing.Optional[_Method] = None
_original_display_key_pressed: typing.Optional[_Method] = None
_installed_wheel_changed: typing.Optional[_Method] = None
_installed_mouse_double_clicked: typing.Optional[_Method] = None
_installed_display_key_pressed: typing.Optional[_Method] = None


def run(instrument: typing.Any, scan_module: typing.Any) -> None:
    """Install the interactive controls once."""

    global _manager
    global _original_wheel_changed
    global _original_mouse_double_clicked
    global _original_display_key_pressed
    global _installed_wheel_changed
    global _installed_mouse_double_clicked
    global _installed_display_key_pressed

    if _manager is not None:
        return

    _manager = InteractiveControlManager(instrument, scan_module)

    image_canvas_item_class = ImageCanvasItem.ImageCanvasItem
    image_area_canvas_item_class = ImageCanvasItem.ImageAreaCanvasItem
    display_panel_class = DisplayPanel.DisplayPanel
    _original_wheel_changed = image_area_canvas_item_class.wheel_changed
    _original_mouse_double_clicked = (
        image_canvas_item_class.mouse_double_clicked
    )
    _original_display_key_pressed = display_panel_class._handle_key_pressed

    def display_key_pressed(
        display_panel: typing.Any,
        key: UserInterface.Key,
    ) -> bool:
        # Nion Swift maps E to the pointer tool before the public key event is
        # fired. Intercept microscope controls before the built-in tools.
        manager = _manager
        if manager is not None and manager.handle_key_pressed(
            display_panel,
            key,
        ):
            return True

        original = _original_display_key_pressed
        return bool(original(display_panel, key)) if original else False

    def wheel_changed(
        image_area_canvas_item: typing.Any,
        x: int,
        y: int,
        dx: int,
        dy: int,
        is_horizontal: bool,
    ) -> bool:
        # Wheel events are consumed by ImageAreaCanvasItem before they reach
        # ImageCanvasItem. Its direct container is the owning ImageCanvasItem.
        image_canvas_item = getattr(
            image_area_canvas_item,
            "container",
            None,
        )
        manager = _manager
        if (
            manager is not None
            and isinstance(
                image_canvas_item,
                ImageCanvasItem.ImageCanvasItem,
            )
            and manager.handle_mouse_wheel(
                image_canvas_item,
                dx,
                dy,
                is_horizontal,
            )
        ):
            return True

        original = _original_wheel_changed
        return bool(
            original(
                image_area_canvas_item,
                x,
                y,
                dx,
                dy,
                is_horizontal,
            )
        ) if original is not None else False

    def mouse_double_clicked(
        image_canvas_item: typing.Any,
        x: int,
        y: int,
        modifiers: UserInterface.KeyboardModifiers,
    ) -> bool:
        manager = _manager
        if manager is not None and manager.handle_double_click(
            image_canvas_item,
            x,
            y,
            modifiers,
        ):
            return True

        original = _original_mouse_double_clicked
        return bool(
            original(image_canvas_item, x, y, modifiers)
        ) if original is not None else False

    _installed_wheel_changed = wheel_changed
    _installed_mouse_double_clicked = mouse_double_clicked
    _installed_display_key_pressed = display_key_pressed

    setattr(
        image_area_canvas_item_class,
        "wheel_changed",
        wheel_changed,
    )
    setattr(
        image_canvas_item_class,
        "mouse_double_clicked",
        mouse_double_clicked,
    )
    setattr(
        display_panel_class,
        "_handle_key_pressed",
        display_key_pressed,
    )


def stop() -> None:
    """Remove listeners and restore the original Nion Swift methods."""

    global _manager
    global _original_wheel_changed
    global _original_mouse_double_clicked
    global _original_display_key_pressed
    global _installed_wheel_changed
    global _installed_mouse_double_clicked
    global _installed_display_key_pressed

    image_canvas_item_class = ImageCanvasItem.ImageCanvasItem
    image_area_canvas_item_class = ImageCanvasItem.ImageAreaCanvasItem
    display_panel_class = DisplayPanel.DisplayPanel

    # Restore only our own hooks. If another plug-in replaced a method after
    # this module did, leave that newer replacement untouched.
    if (
        _installed_wheel_changed is not None
        and image_area_canvas_item_class.wheel_changed
        is _installed_wheel_changed
        and _original_wheel_changed is not None
    ):
        setattr(
            image_area_canvas_item_class,
            "wheel_changed",
            _original_wheel_changed,
        )

    if (
        _installed_mouse_double_clicked is not None
        and image_canvas_item_class.mouse_double_clicked
        is _installed_mouse_double_clicked
        and _original_mouse_double_clicked is not None
    ):
        setattr(
            image_canvas_item_class,
            "mouse_double_clicked",
            _original_mouse_double_clicked,
        )

    if (
        _installed_display_key_pressed is not None
        and display_panel_class._handle_key_pressed
        is _installed_display_key_pressed
        and _original_display_key_pressed is not None
    ):
        setattr(
            display_panel_class,
            "_handle_key_pressed",
            _original_display_key_pressed,
        )

    if _manager is not None:
        _manager.close()
        _manager = None

    _original_wheel_changed = None
    _original_mouse_double_clicked = None
    _original_display_key_pressed = None
    _installed_wheel_changed = None
    _installed_mouse_double_clicked = None
    _installed_display_key_pressed = None
