"""
Global QSS (Qt Style Sheets) theme for Meshine Shop.

All visual styling is centralized here rather than scattered across
individual widgets. This makes the app's look-and-feel easy to update
and keeps the UI code focused on layout and behavior.

Color palette:
    - Background:     #2d2d2d (charcoal gray — main canvas)
    - Surface:        #252525 (darker gray — sidebar, cards, panels)
    - Elevated:       #333333 (lighter gray — hover states)
    - Border:         #3a3a3a (subtle dividers and outlines)
    - Primary text:   #e0e0e0 (light gray — body text)
    - Secondary text: #d4d4d4 (slightly dimmer — headings, titles)
    - Muted text:     #999999 (dimmed — subtitles, placeholders)
    - Accent:         #dc3545 (crimson — active states, branding, highlights)
    - Deep surface:   #1e1e1e (near-black — status bar)

The font stack prioritizes Helvetica Neue (macOS), falls back to
Segoe UI (Windows), then Arial as a universal fallback. This ensures
the app looks native on both target platforms.
"""

DARK_THEME = """
/* ===== Base Styles ===== */
/* Sets the main window background to charcoal gray. */
QMainWindow {
    background-color: #2d2d2d;
}

/* Global defaults for all widgets — text color, font family, and size.
   Every widget inherits these unless explicitly overridden below. */
QWidget {
    color: #e0e0e0;
    font-family: "Helvetica Neue", "Segoe UI", "Arial", sans-serif;
    font-size: 13px;
}

/* ===== Sidebar ===== */
/* The sidebar sits on the left edge of the window. It uses a slightly
   darker background than the main canvas to create visual separation,
   with a subtle border on its right edge. */
QWidget#sidebar {
    background-color: #252525;
    border-right: 1px solid #3a3a3a;
}

/* Navigation buttons in the sidebar — transparent by default so they
   blend into the sidebar background. Muted text color keeps inactive
   items visually quiet. */
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

/* Hover state — subtle background lift and brighter text to indicate
   the button is interactive. */
QPushButton#nav_button:hover {
    background-color: #333333;
    color: #d4d4d4;
}

/* Checked/active state — the currently selected nav item gets a visible
   background and the crimson accent color to clearly show which view
   the user is on. */
QPushButton#nav_button:checked {
    background-color: #3a3a3a;
    color: #dc3545;
}

/* ===== Drop Zone ===== */
/* The drag-and-drop area for importing photo sets. The dashed border
   signals "drop here" affordance, a common UX pattern for file upload. */
QWidget#drop_zone {
    background-color: #252525;
    border: 2px dashed #3a3a3a;
    border-radius: 16px;
}

/* Hover state — crimson border lights up when dragging files over
   the zone, giving clear visual feedback that the drop will be accepted. */
QWidget#drop_zone:hover {
    border-color: #dc3545;
    background-color: #333333;
}

/* Title text inside the drop zone — slightly larger and bolder than
   body text to draw the user's eye to the action prompt. */
QLabel#drop_title {
    color: #d4d4d4;
    font-size: 18px;
    font-weight: 600;
}

/* Subtitle text — muted color for the supported file types hint.
   Secondary information that shouldn't compete with the title. */
QLabel#drop_subtitle {
    color: #999999;
    font-size: 13px;
}

/* The "+" icon in the drop zone — uses the crimson accent color at a
   large size to serve as a visual focal point. */
QLabel#drop_icon {
    color: #dc3545;
    font-size: 48px;
}

/* ===== Processing Queue ===== */
/* The queue panel card — darker surface with rounded corners to create
   a contained "card" appearance within the Process view. */
QWidget#queue_panel {
    background-color: #252525;
    border-radius: 12px;
}

/* Section titles used across views (Import, Processing Queue, Export).
   Slightly larger and bolder than body text for visual hierarchy. */
QLabel#section_title {
    color: #d4d4d4;
    font-size: 16px;
    font-weight: 600;
}

/* Empty state message in the queue — muted to convey "nothing here yet"
   without being visually heavy. */
QLabel#queue_empty {
    color: #999999;
    font-size: 13px;
}

/* ===== Status Bar ===== */
/* Sits at the very bottom of the window. Uses the deepest background
   color to anchor the UI, with crimson text for status messages so
   they're noticeable but not distracting. */
QStatusBar {
    background-color: #1e1e1e;
    color: #dc3545;
    font-size: 12px;
    padding: 4px 12px;
}

/* ===== Scrollbar ===== */
/* Custom scrollbar styling to match the dark theme. Thin (8px) and
   rounded for a modern, unobtrusive look. */
QScrollBar:vertical {
    background-color: #2d2d2d;
    width: 8px;
    border-radius: 4px;
}

/* Scrollbar handle — the draggable thumb. Subtle gray that lights up
   to crimson on hover for discoverability. */
QScrollBar::handle:vertical {
    background-color: #3a3a3a;
    border-radius: 4px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #dc3545;
}

/* Hide the up/down arrow buttons — they add clutter and most users
   scroll via trackpad/mouse wheel anyway. */
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

/* ===== Start Processing Button ===== */
/* Crimson accent button that stands out against the charcoal theme.
   Disabled state dims the button to signal it's not yet interactive
   (files must be dropped first). */
QPushButton#start_button {
    background-color: #dc3545;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    padding: 10px 24px;
}

QPushButton#start_button:hover {
    background-color: #c82333;
}

QPushButton#start_button:pressed {
    background-color: #a71d2a;
}

QPushButton#start_button:disabled {
    background-color: #3a3a3a;
    color: #666666;
}

/* ===== Pipeline Stage Rows ===== */
/* Each row in the processing queue represents one pipeline stage.
   Minimal styling to keep the focus on the status indicators. */
QWidget#stage_row {
    background-color: transparent;
}

/* Stage name labels — slightly brighter than body text for readability. */
QLabel#stage_name {
    color: #d4d4d4;
    font-size: 13px;
}

/* Status indicator colors for each pipeline stage state. */
QLabel#stage_status_pending {
    color: #4a5568;
    font-size: 16px;
}

QLabel#stage_status_running {
    color: #dc3545;
    font-size: 16px;
}

QLabel#stage_status_done {
    color: #28a745;
    font-size: 16px;
}

QLabel#stage_status_error {
    color: #dc3545;
    font-size: 16px;
    font-weight: 600;
}

/* Status message text — muted so the stage name and indicator are primary. */
QLabel#stage_message {
    color: #999999;
    font-size: 12px;
}
"""
