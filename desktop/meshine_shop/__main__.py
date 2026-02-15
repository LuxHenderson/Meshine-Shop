"""
Application entry point for Meshine Shop.

This module is invoked when the package is run directly via:
    python -m meshine_shop

It creates the Qt application instance, instantiates the main window,
and starts the event loop. The event loop keeps the window open and
responsive until the user closes it — at which point sys.exit ensures
a clean shutdown with the appropriate exit code.
"""

import sys
from PySide6.QtWidgets import QApplication
from meshine_shop.app import MeshineShopApp


def main():
    # QApplication manages the GUI application's control flow and settings.
    # sys.argv is passed so Qt can process any command-line arguments it
    # recognizes (e.g., --style, --platform), which aids cross-platform support.
    app = QApplication(sys.argv)

    # Create and display the main application window.
    window = MeshineShopApp()
    window.show()

    # app.exec() enters the Qt event loop — blocking here until the window
    # is closed. sys.exit ensures the process returns Qt's exit code to the OS.
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
