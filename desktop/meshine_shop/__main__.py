import sys
from PySide6.QtWidgets import QApplication
from meshine_shop.app import MeshineShopApp


def main():
    app = QApplication(sys.argv)
    window = MeshineShopApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
