from skygui import SkyModelGUI


def main(*args):
    window = SkyModelGUI()
    window()


if __name__ == "__main__":
    import sys

    main(*sys.argv)
