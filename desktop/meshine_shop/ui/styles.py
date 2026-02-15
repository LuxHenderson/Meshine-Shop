DARK_THEME = """
QMainWindow {
    background-color: #2d2d2d;
}

QWidget {
    color: #e0e0e0;
    font-family: "Helvetica Neue", "Segoe UI", "Arial", sans-serif;
    font-size: 13px;
}

/* Sidebar */
QWidget#sidebar {
    background-color: #252525;
    border-right: 1px solid #3a3a3a;
}

QPushButton#nav_button {
    background-color: transparent;
    border: none;
    border-radius: 8px;
    color: #999999;
    font-size: 13px;
    font-weight: 500;
    padding: 12px 20px;
    text-align: left;
}

QPushButton#nav_button:hover {
    background-color: #333333;
    color: #d4d4d4;
}

QPushButton#nav_button:checked {
    background-color: #3a3a3a;
    color: #dc3545;
}

/* Drop Zone */
QWidget#drop_zone {
    background-color: #252525;
    border: 2px dashed #3a3a3a;
    border-radius: 16px;
}

QWidget#drop_zone:hover {
    border-color: #dc3545;
    background-color: #333333;
}

QLabel#drop_title {
    color: #d4d4d4;
    font-size: 18px;
    font-weight: 600;
}

QLabel#drop_subtitle {
    color: #999999;
    font-size: 13px;
}

QLabel#drop_icon {
    color: #dc3545;
    font-size: 48px;
}

/* Processing Queue */
QWidget#queue_panel {
    background-color: #252525;
    border-radius: 12px;
}

QLabel#section_title {
    color: #d4d4d4;
    font-size: 16px;
    font-weight: 600;
}

QLabel#queue_empty {
    color: #999999;
    font-size: 13px;
}

/* Status Bar */
QStatusBar {
    background-color: #1e1e1e;
    color: #dc3545;
    font-size: 12px;
    padding: 4px 12px;
}

/* Scrollbar */
QScrollBar:vertical {
    background-color: #2d2d2d;
    width: 8px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background-color: #3a3a3a;
    border-radius: 4px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #dc3545;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
"""
