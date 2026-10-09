"""Profile-persistent display brightness/contrast in the existing Scan Control.

Use Nion's display transfer properties, preserving quantitative acquisition data.
The panel extension is scoped to uSim and restored when the plugin stops.
"""
import math
import weakref
from nion.utils import Observable
from nion.ui import Declarative


def get_tone(parameters):
    getter = getattr(parameters, 'get_parameter', None)
    if getter is not None:
        return float(getter('usim_brightness', 0.)), float(getter('usim_contrast', 1.))
    return getattr(parameters, 'usim_brightness', 0.), getattr(parameters, 'usim_contrast', 1.)


def set_tone(parameters, brightness, contrast):
    if not math.isfinite(brightness) or not math.isfinite(contrast) or contrast <= 0:
        raise ValueError('Brightness must be finite and contrast must be finite and positive')
    setter = getattr(parameters, 'set_parameter', None)
    for key, value in (('usim_brightness', brightness), ('usim_contrast', contrast)):
        if setter is not None:
            setter(key, value)
        else:
            setattr(parameters, key, value)


def matches(data_item, source):
    metadata = getattr(data_item, 'metadata', {}) or {}
    hardware = metadata.get('hardware_source', {})
    camera = metadata.get('camera', {})
    camera_id = camera.get('hardware_source_id') or hardware.get('hardware_source_id')
    if camera_id:
        return camera_id in (source.hardware_source_id, 'usim_ronchigram_camera')
    return metadata.get('scan', {}).get('hardware_source_id') == source.hardware_source_id


def apply_to_document(document_controller, source, parameters):
    if document_controller is None:
        return
    brightness, contrast = get_tone(parameters)
    for display_item in document_controller.document_model.display_items:
        for channel in display_item.display_data_channels:
            if channel.data_item is not None and matches(channel.data_item, source):
                channel.brightness, channel.contrast = brightness, contrast


class ToneModel(Observable.Observable):
    def __init__(self, document_controller, source):
        super().__init__()
        self.document_controller, self.source = document_controller, source
        self.closed = self.pending = False
        self.data_listeners = {}
        self.listeners = [source.scan_settings.profile_changed_event.listen(self.schedule),
            source.scan_settings.frame_parameters_changed_event.listen(self.schedule),
            document_controller.document_model.item_inserted_event.listen(self.inserted),
            document_controller.document_model.item_removed_event.listen(self.removed)]
        for item in document_controller.document_model.data_items:
            self.inserted('data_items', item, 0)
        self.refresh()

    def inserted(self, key, item, index):
        if key == 'data_items' and item not in self.data_listeners:
            self.data_listeners[item] = item.data_changed_event.listen(self.schedule)

    def removed(self, key, item, index):
        if key == 'data_items' and item in self.data_listeners:
            self.data_listeners.pop(item).close()

    def schedule(self, *args):
        if not self.closed and not self.pending:
            self.pending = True
            self.document_controller.queue_task(self.refresh)

    def refresh(self):
        self.pending = False
        if self.closed:
            return
        self.notify_property_changed('brightness_str')
        self.notify_property_changed('contrast_str')
        parameters = self.source.get_frame_parameters(self.source.selected_profile_index)
        apply_to_document(self.document_controller, self.source, parameters)

    def write(self, value, brightness):
        try:
            number = float(value)
            index = self.source.selected_profile_index
            parameters = self.source.get_frame_parameters(index)
            old_brightness, old_contrast = get_tone(parameters)
            set_tone(parameters, number if brightness else old_brightness,
                     old_contrast if brightness else number)
        except (TypeError, ValueError):
            self.refresh()
            return
        self.source.set_frame_parameters(index, parameters)
        self.refresh()

    @property
    def brightness_str(self):
        return f'{get_tone(self.source.get_frame_parameters(self.source.selected_profile_index))[0]:.2f}'

    @brightness_str.setter
    def brightness_str(self, value):
        self.write(value, True)

    @property
    def contrast_str(self):
        return f'{get_tone(self.source.get_frame_parameters(self.source.selected_profile_index))[1]:.2f}'

    @contrast_str.setter
    def contrast_str(self, value):
        self.write(value, False)

    def close(self):
        self.closed = True
        for listener in (*self.listeners, *self.data_listeners.values()):
            listener.close()
        self.listeners.clear()
        self.data_listeners.clear()
        self.document_controller = self.source = None


_original_init = _original_close = None
_models = weakref.WeakSet()


def run():
    global _original_init, _original_close
    if _original_init is not None:
        return
    from nionswift_plugin.nion_instrumentation_ui import ScanControlPanel
    cls = ScanControlPanel.ScanPanelController
    _original_init, _original_close = cls.__init__, cls.close
    original_init, original_close = _original_init, _original_close
    def initialize(controller, document_controller, source):
        original_init(controller, document_controller, source)
        if not source.hardware_source_id.startswith('usim_'):
            return
        def remove_simulator_fov_warning(view):
            if isinstance(view, dict):
                for name, value in list(view.items()):
                    if value == '@binding(_model.fov_label_color)':
                        view[name] = 'black'
                    elif value == '@binding(_model.fov_label_tool_tip)':
                        view[name] = 'Simulated scan field of view; no hardware maximum.'
                    else:
                        remove_simulator_fov_warning(value)
            elif isinstance(view, list):
                for child in view:
                    remove_simulator_fov_warning(child)
        remove_simulator_fov_warning(controller.ui_view)
        controller._usim_tone = ToneModel(document_controller, source)
        _models.add(controller._usim_tone)
        u = Declarative.DeclarativeUI()
        rows = [u.create_row(u.create_label(text=label, width=100),
            u.create_line_edit(text=f'@binding(_usim_tone.{property_name})', width=60),
            u.create_stretch(), spacing=6, margin_horizontal=8)
            for label, property_name in (('Brightness (B)', 'brightness_str'), ('Contrast (C)', 'contrast_str'))]
        controller.ui_view['children'][1:1] = rows
    def close(controller):
        model = getattr(controller, '_usim_tone', None)
        if model is not None:
            model.close()
        original_close(controller)
    cls.__init__, cls.close = initialize, close


def stop():
    global _original_init, _original_close
    if _original_init is None:
        return
    from nionswift_plugin.nion_instrumentation_ui import ScanControlPanel
    cls = ScanControlPanel.ScanPanelController
    cls.__init__, cls.close = _original_init, _original_close
    _original_init = _original_close = None
    for model in list(_models):
        model.close()
    _models.clear()
