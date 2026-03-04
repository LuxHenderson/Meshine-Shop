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
/* The sidebar sits on the left edge of the window. Slightly darker
   background than the main canvas to distinguish the menu zone. The
   right border is now handled by a dedicated QFrame separator widget
   in app.py for precise 1px rendering. */
QWidget#sidebar {
    background-color: #252525;
}

/* App title at the top of the sidebar. */
QLabel#sidebar_title {
    color: #dc3545;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 2px;
    padding: 0 8px 16px 8px;
}

/* Horizontal rule below the sidebar title — visually separates branding
   from the navigation items. */
QFrame#sidebar_divider {
    background-color: #3a3a3a;
    border: none;
}

/* "NAVIGATION" section label — small all-caps label that groups the
   nav buttons and makes the sidebar read as a structured menu. */
QLabel#nav_section_label {
    color: #4a5568;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
    padding: 12px 8px 4px 8px;
}

/* Vertical separator line between sidebar and main content. */
QFrame#sidebar_separator {
    background-color: #3a3a3a;
    border: none;
}

/* Navigation buttons — transparent by default so they blend into the
   sidebar. Muted text keeps inactive items visually quiet. A left border
   is reserved for the active state indicator. */
QPushButton#nav_button {
    background-color: transparent;
    border: none;
    border-left: 3px solid transparent;
    border-radius: 0px;
    color: #999999;
    font-size: 13px;
    font-weight: 500;
    padding: 12px 20px;
    text-align: left;
}

/* Hover state — subtle background lift and brighter text. */
QPushButton#nav_button:hover {
    background-color: #333333;
    color: #d4d4d4;
    border-left: 3px solid #555555;
}

/* Checked/active state — crimson left accent bar clearly marks the
   current view, a standard navigation pattern in sidebar menus. */
QPushButton#nav_button:checked {
    background-color: #2d2d2d;
    color: #dc3545;
    border-left: 3px solid #dc3545;
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
/* Outlined style matching the Browse Folder button for a cohesive look.
   Disabled state dims the button to signal it's not yet interactive
   (files must be dropped first). */
QPushButton#start_button {
    background-color: transparent;
    color: #d4d4d4;
    border: 1px solid #555555;
    border-radius: 8px;
    font-size: 13px;
    padding: 8px 20px;
}

QPushButton#start_button:hover {
    background-color: #3a3a3a;
    border-color: #dc3545;
    color: #ffffff;
}

QPushButton#start_button:pressed {
    background-color: #2d2d2d;
}

QPushButton#start_button:disabled {
    background-color: transparent;
    color: #666666;
    border-color: #3a3a3a;
}

/* ===== Clear Button ===== */
/* Small inline button next to the file count label. Same color scheme
   as other buttons but retains its compact size and position. */
QPushButton#clear_button {
    background-color: transparent;
    color: #d4d4d4;
    border: 1px solid #555555;
    border-radius: 4px;
    font-size: 12px;
    padding: 4px 12px;
}

QPushButton#clear_button:hover {
    background-color: #3a3a3a;
    border-color: #dc3545;
    color: #ffffff;
}

/* ===== Browse Folder Button ===== */
/* Secondary action button — outlined style so it doesn't compete with
   the primary crimson Start Processing button for visual attention. */
QPushButton#browse_button {
    background-color: transparent;
    color: #d4d4d4;
    border: 1px solid #555555;
    border-radius: 8px;
    font-size: 13px;
    padding: 8px 20px;
}

QPushButton#browse_button:hover {
    background-color: #3a3a3a;
    border-color: #dc3545;
    color: #ffffff;
}

QPushButton#browse_button:pressed {
    background-color: #2d2d2d;
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

/* ===== Reset Button ===== */
/* Same outlined style as all other buttons for visual consistency. */
QPushButton#reset_button {
    background-color: transparent;
    color: #d4d4d4;
    border: 1px solid #555555;
    border-radius: 8px;
    font-size: 13px;
    padding: 8px 20px;
}

QPushButton#reset_button:hover {
    background-color: #3a3a3a;
    border-color: #dc3545;
    color: #ffffff;
}

QPushButton#reset_button:pressed {
    background-color: #2d2d2d;
}

/* ===== Export View ===== */
/* Export button — same outlined style as all other buttons. */
QPushButton#export_button {
    background-color: transparent;
    color: #d4d4d4;
    border: 1px solid #555555;
    border-radius: 8px;
    font-size: 13px;
    padding: 8px 20px;
}

QPushButton#export_button:hover {
    background-color: #3a3a3a;
    border-color: #dc3545;
    color: #ffffff;
}

QPushButton#export_button:pressed {
    background-color: #a71d2a;
}

/* Mesh info label — slightly brighter text for the stats readout so
   it stands out against the dark background. */
QLabel#mesh_info {
    color: #d4d4d4;
    font-size: 14px;
    padding: 12px 0;
}

/* Format label next to the dropdown. */
QLabel#export_label {
    color: #d4d4d4;
    font-size: 13px;
}

/* Quality label next to the preset dropdown on the Import page. */
QLabel#quality_label {
    color: #d4d4d4;
    font-size: 13px;
}

/* Format dropdown — dark surface with subtle border to match the
   overall theme. The drop-down arrow area gets a slightly different
   background for visual separation. */
QComboBox#format_combo {
    background-color: #252525;
    color: #e0e0e0;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 13px;
}

QComboBox#format_combo:hover {
    border-color: #999999;
}

QComboBox#format_combo::drop-down {
    border: none;
    width: 28px;
}

/* Dropdown list styling — matches the dark surface palette. */
QComboBox#format_combo QAbstractItemView {
    background-color: #252525;
    color: #e0e0e0;
    border: 1px solid #3a3a3a;
    selection-background-color: #3a3a3a;
    selection-color: #dc3545;
}

/* ===== Quality Preset Dropdown ===== */
/* Matches the format dropdown styling for visual consistency across
   both the Import and Export views. */
QComboBox#quality_combo {
    background-color: #252525;
    color: #e0e0e0;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 13px;
}

QComboBox#quality_combo:hover {
    border-color: #999999;
}

QComboBox#quality_combo::drop-down {
    border: none;
    width: 28px;
}

QComboBox#quality_combo QAbstractItemView {
    background-color: #252525;
    color: #e0e0e0;
    border: 1px solid #3a3a3a;
    selection-background-color: #3a3a3a;
    selection-color: #dc3545;
}

/* ===== Viewport Tools Panel ===== */
/* The 200px fixed-width right panel on the Viewport page. Slightly darker
   surface than the main canvas, with a left border to separate it from the
   OpenGL viewport area. */
QWidget#viewport_tools_panel {
    background-color: #252525;
    border-left: 1px solid #3a3a3a;
}

/* Section header labels in the tools panel. */
QLabel#tools_section_header {
    color: #4a5568;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
}

/* Text labels next to sliders (Size, Opacity). */
QLabel#tools_label {
    color: #d4d4d4;
    font-size: 12px;
}

/* Numeric readout next to each slider. */
QLabel#tools_readout {
    color: #999999;
    font-size: 12px;
}

/* Horizontal dividers within the tools panel. */
QFrame#tools_divider {
    background-color: #3a3a3a;
    border: none;
}

/* Tool selector buttons (Brush / Region) — square icon-style.
   Transparent background by default, hover lifts slightly, checked
   state uses the crimson accent border as the active indicator. */
QPushButton#tool_button {
    background: transparent;
    color: #999999;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    padding: 4px;
    font-size: 18px;
}

QPushButton#tool_button:hover {
    background: #333333;
    color: #e0e0e0;
}

QPushButton#tool_button:checked {
    background: #2d2d2d;
    border-color: #dc3545;
    color: #e0e0e0;
}

/* Color swatch button — displays the current paint color as its background.
   The actual background-color is set inline via setStyleSheet() per-color
   change; this rule just provides the border and sizing defaults. */
QPushButton#color_swatch {
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    min-height: 28px;
}

QPushButton#color_swatch:hover {
    border-color: #dc3545;
}

/* ⚙ Controls button at the bottom of the tools panel. */
QPushButton#controls_gear_btn {
    background-color: transparent;
    color: #d4d4d4;
    border: 1px solid #555555;
    border-radius: 6px;
    font-size: 12px;
    padding: 6px 12px;
    text-align: left;
}

QPushButton#controls_gear_btn:hover {
    background-color: #333333;
    border-color: #dc3545;
    color: #ffffff;
}

/* Viewport placeholder label — shown in the OpenGL area before any
   mesh is loaded. Centered muted text over the dark background. */
QLabel#viewport_placeholder {
    color: #999999;
    font-size: 13px;
}

/* ===== Viewport Controls Dialog ===== */
/* The scheme selector bar at the top of the controls dialog. */
QWidget#controls_scheme_bar {
    background-color: #252525;
}

/* Horizontal and vertical dividers inside the controls dialog. */
QFrame#controls_divider {
    background-color: #3a3a3a;
    border: none;
}

/* Left category list — no frame, dark surface. */
QListWidget#controls_category_list {
    background-color: #252525;
    color: #d4d4d4;
    font-size: 13px;
    border: none;
    padding: 8px 0;
}

QListWidget#controls_category_list::item {
    padding: 10px 16px;
    border-left: 3px solid transparent;
}

QListWidget#controls_category_list::item:hover {
    background-color: #333333;
    color: #e0e0e0;
}

QListWidget#controls_category_list::item:selected {
    background-color: #2d2d2d;
    color: #dc3545;
    border-left: 3px solid #dc3545;
}

/* Section header labels inside category pages. */
QLabel#controls_section_header {
    color: #d4d4d4;
    font-size: 14px;
    font-weight: 600;
    padding-bottom: 4px;
}

/* Action label text in binding rows. */
QLabel#binding_action_label {
    color: #d4d4d4;
    font-size: 12px;
}

/* Binding pill — shows current binding text in a rounded box. */
QLabel#binding_pill {
    background-color: #333333;
    color: #e0e0e0;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    font-size: 12px;
    padding: 2px 8px;
}

/* ✕ / ↺ / ✎ icon buttons in binding rows. */
QPushButton#binding_icon_btn {
    background: transparent;
    color: #999999;
    border: 1px solid #3a3a3a;
    border-radius: 3px;
    font-size: 11px;
    padding: 0;
}

QPushButton#binding_icon_btn:hover {
    background: #333333;
    color: #e0e0e0;
    border-color: #dc3545;
}

/* Save Changes button in controls dialog footer. */
QPushButton#primary_button {
    background-color: #dc3545;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    padding: 8px 20px;
}

QPushButton#primary_button:hover {
    background-color: #c82333;
}

QPushButton#primary_button:pressed {
    background-color: #a71d2a;
}

/* Scheme dropdown inside controls dialog. */
QComboBox#scheme_combo {
    background-color: #333333;
    color: #e0e0e0;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 13px;
}

QComboBox#scheme_combo::drop-down {
    border: none;
    width: 24px;
}

QComboBox#scheme_combo QAbstractItemView {
    background-color: #252525;
    color: #e0e0e0;
    border: 1px solid #3a3a3a;
    selection-background-color: #3a3a3a;
    selection-color: #dc3545;
}
"""
