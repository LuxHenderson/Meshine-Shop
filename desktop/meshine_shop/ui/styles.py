DARK_THEME = """
QMainWindow {
    background-color: #1a1a2e;
}

QWidget {
    color: #e0e0e0;
    font-family: "Helvetica Neue", "Segoe UI", "Arial", sans-serif;
    font-size: 13px;
}

/* Sidebar */
QWidget#sidebar {
    background-color: #16213e;
    border-right: 1px solid #0f3460;
}

QPushButton#nav_button {
    background-color: transparent;
    border: none;
    border-radius: 8px;
    color: #8892b0;
    font-size: 13px;
    font-weight: 500;
    padding: 12px 20px;
    text-align: left;
}

QPushButton#nav_button:hover {
    background-color: #1a1a40;
    color: #ccd6f6;
}

QPushButton#nav_button:checked {
    background-color: #0f3460;
    color: #64ffda;
}

/* Drop Zone */
QWidget#drop_zone {
    background-color: #16213e;
    border: 2px dashed #0f3460;
    border-radius: 16px;
}

QWidget#drop_zone:hover {
    border-color: #64ffda;
    background-color: #1a1a40;
}

QLabel#drop_title {
    color: #ccd6f6;
    font-size: 18px;
    font-weight: 600;
}

QLabel#drop_subtitle {
    color: #8892b0;
    font-size: 13px;
}

QLabel#drop_icon {
    color: #64ffda;
    font-size: 48px;
}

/* Processing Queue */
QWidget#queue_panel {
    background-color: #16213e;
    border-radius: 12px;
}

QLabel#section_title {
    color: #ccd6f6;
    font-size: 16px;
    font-weight: 600;
}

QLabel#queue_empty {
    color: #8892b0;
    font-size: 13px;
}

/* Status Bar */
QStatusBar {
    background-color: #0f3460;
    color: #64ffda;
    font-size: 12px;
    padding: 4px 12px;
}

/* Scrollbar */
QScrollBar:vertical {
    background-color: #1a1a2e;
    width: 8px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background-color: #0f3460;
    border-radius: 4px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #64ffda;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
"""
