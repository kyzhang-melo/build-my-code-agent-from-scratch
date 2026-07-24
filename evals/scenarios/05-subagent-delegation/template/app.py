from registry import register_widget, get_widget


def build_page():
    register_widget("header")
    register_widget("footer")
    return [get_widget("header").render(), get_widget("footer").render()]


if __name__ == "__main__":
    print("\n".join(build_page()))
