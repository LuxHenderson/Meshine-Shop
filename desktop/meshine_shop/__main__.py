"""
Application entry point for Meshine Shop.

This module is invoked when the package is run directly via:
    python -m meshine_shop

It creates the Qt application instance, instantiates the main window,
and starts the event loop. The event loop keeps the window open and
responsive until the user closes it — at which point sys.exit ensures
a clean shutdown with the appropriate exit code.
"""

import logging
import sys
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication
from meshine_shop.app import MeshineShopApp

# Configure logging so viewport and painter messages appear in the terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)


def main():
    # Set up an OpenGL 4.1 Core Profile surface format before creating the
    # QApplication. This MUST happen before the application object is created
    # — Qt bakes the default format into every QOpenGLWidget context.
    #
    # macOS ships OpenGL up to version 4.1 Core Profile. moderngl requires
    # at least OpenGL 3.3 Core Profile. Without this, macOS defaults to the
    # legacy 2.1 Compatibility profile, which causes a SIGSEGV when moderngl
    # tries to call Core-only entry points (e.g. glBindVertexArray).
    gl_format = QSurfaceFormat()
    gl_format.setVersion(4, 1)
    gl_format.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    gl_format.setDepthBufferSize(24)
    gl_format.setStencilBufferSize(8)
    gl_format.setSamples(4)  # 4× MSAA for smoother mesh edges
    QSurfaceFormat.setDefaultFormat(gl_format)

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
