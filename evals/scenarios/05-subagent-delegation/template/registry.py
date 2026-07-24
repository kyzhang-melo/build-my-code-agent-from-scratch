from widget import Widget

_WIDGETS = {}


def register_widget(name):
    """Register a widget instance under the given name."""
    _WIDGETS[name] = Widget(name)
    return _WIDGETS[name]


def get_widget(name):
    return _WIDGETS.get(name)
