class Widget:
    """A UI widget with a name and a render method."""

    def __init__(self, name):
        self.name = name

    def render(self):
        return f"<widget:{self.name}>"
