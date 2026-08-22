import sys
import polars as pl
from typing import cast
from typing import TypeAlias

DataSource: TypeAlias= (pl.DataFrame | pl.LazyFrame)

from PySide6.QtCore import (
    QAbstractTableModel,
    QEvent,
    QLoggingCategory,
    QRect,
    Qt,
)

from PySide6.QtGui import QCursor, QKeySequence, QShortcut

from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

HEADER_HORIZONTAL_PADDING = 24


def silence_wayland_text_input_warning():
    QLoggingCategory.setFilterRules(
        "qt.qpa.wayland.textinput.warning=false"
    )


def configure_table(table):
    table.setAlternatingRowColors(True)
    table.setSelectionBehavior(QTableView.SelectionBehavior.SelectItems)
    table.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
    table.resizeColumnsToContents()

    header = table.horizontalHeader()
    for column in range(table.model().columnCount()):
        header_width = (
            header.sectionSizeHint(column)
            + HEADER_HORIZONTAL_PADDING
        )
        table.setColumnWidth(column, max(table.columnWidth(column), header_width))


class PolarsModel(QAbstractTableModel):
    def __init__(self, df):
        super().__init__()

        self.df = df
        self.columns = df.columns

    def rowCount(self, parent=None):
        return self.df.height

    def columnCount(self, parent=None):
        return self.df.width

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            value = self.df[index.row(), index.column()]
            return str(value)

        return None

    def headerData(self, section, orientation, role):
        if role != Qt.ItemDataRole.DisplayRole:
            return None

        if orientation == Qt.Orientation.Horizontal:
            return self.columns[section]

        return str(section)


class TrafficButton(QPushButton):
    """
    macOS-style traffic button.

    Normal:
        ●

    Hover:
        × / − / + / restore
    """

    def __init__(self, color, symbol, size=14):
        super().__init__()

        self.symbol = symbol

        self.setFixedSize(size, size)

        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        self.setProperty("buttonColor", color)

        radius = size // 2

        self.setStyleSheet(
            f"""
            QPushButton {{
                background: {color};
                border: none;
                border-radius: {radius}px;
                color: rgba(0, 0, 0, 0.65);
                font-family: "CaskaydiaCove Nerd Font";
                font-size: 14px;
                font-weight: bold;
            }}

            QPushButton:hover {{
                background: {color};
            }}
            """
        )

    def enterEvent(self, event):
        self.setText(self.symbol)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setText("")
        super().leaveEvent(event)


# Traffic button placement / styling config.
#
# position : "left"  -> macOS style, buttons on the left of the title bar
#            "right" -> Windows style, buttons on the right of the title bar
# order    : which buttons to show and in what order (subset of
#            "close", "minimize", "maximize")
# size     : diameter of each button in px
# spacing  : gap between buttons in px
# margin   : distance from the title bar's outer edge in px
DEFAULT_TRAFFIC_CONFIG = {
    "position": "left",
    "order": ["close", "minimize", "maximize"],
    "size": 14,
    "spacing": 10,
    "margin": 16,
}


class TitleBar(QWidget):
    def __init__(self, window, traffic_config=None):
        super().__init__()

        self.main_window = window
        self.drag_position = None

        self.setFixedHeight(38)

        self.traffic_config = {**DEFAULT_TRAFFIC_CONFIG, **(traffic_config or {})}

        margin = self.traffic_config["margin"]
        size = self.traffic_config["size"]

        layout = QHBoxLayout(self)
        layout.setContentsMargins(margin, 0, margin, 0)
        layout.setSpacing(self.traffic_config["spacing"])

        # Traffic buttons
        self.close_button = TrafficButton("#E9524A", "×", size)
        self.minimize_button = TrafficButton("#F1AE1B", "−", size)
        self.maximize_button = TrafficButton("#59C837", "□", size)

        self.close_button.clicked.connect(self.main_window.close)
        self.minimize_button.clicked.connect(self.main_window.showMinimized)
        self.maximize_button.clicked.connect(self.toggle_maximize)

        buttons_by_name = {
            "close": self.close_button,
            "minimize": self.minimize_button,
            "maximize": self.maximize_button,
        }

        # Title
        self.title = QLabel(window.windowTitle())
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        order = self.traffic_config["order"]
        traffic_widgets = [
            buttons_by_name[name] for name in order if name in buttons_by_name
        ]

        if self.traffic_config["position"] == "right":
            layout.addStretch()
            layout.addWidget(self.title)
            layout.addStretch()

            for w in traffic_widgets:
                layout.addWidget(w)
        else:
            for w in traffic_widgets:
                layout.addWidget(w)

            layout.addStretch()
            layout.addWidget(self.title)
            layout.addStretch()

    def toggle_maximize(self):
        if self.main_window.isMaximized():
            self.main_window.showNormal()
            self.maximize_button.symbol = "□"
        else:
            self.main_window.showMaximized()
            self.maximize_button.symbol = "❐"

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return

        self.drag_position = (
            event.globalPosition().toPoint()
            - self.main_window.frameGeometry().topLeft()
        )

        event.accept()

    def mouseMoveEvent(self, event):
        if (
            self.drag_position is not None
            and event.buttons() & Qt.MouseButton.LeftButton
            and not self.main_window.isMaximized()
        ):
            current = event.globalPosition().toPoint()
            self.main_window.move(current - self.drag_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        self.drag_position = None
        event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle_maximize()


class DataFrameWindow(QMainWindow):
    # Make the resize detection area thinner so the scrollbar is easier
    # to reach with the mouse pointer.
    BORDER = 4

    def __init__(self, df, title="Polars DataFrame", traffic_config=None):
        super().__init__()

        self.setWindowTitle(title)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Window
        )

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(500, 300)

        QShortcut(QKeySequence("Esc"), self, activated=self.close)
        QShortcut(QKeySequence("Q"), self, activated=self.close)

        # Resize state
        self.resizing = False
        self.resize_direction = None
        self.resize_start = None
        self.original_geometry = None

        # Title bar
        self.titlebar = TitleBar(self, traffic_config)

        # Table
        self.table = QTableView()

        self.model = PolarsModel(df)
        self.table.setModel(self.model)
        configure_table(self.table)

        # Layout
        central = QWidget()

        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self.titlebar)
        layout.addWidget(self.table)

        self.setCentralWidget(central)

        self.resize(1200, 700)

        # Styling
        self.setStyleSheet(
            """
            QMainWindow {
                background: #1e1e1e;
                border: 1px solid #3a3a3a;
                border-radius: 12px;
            }

            QWidget {
                background: #1e1e1e;
                color: #eeeeee;
            }

            QLabel {
                color: #d0d0d0;
                font-size: 13px;
                font-weight: 500;
            }

            QTableView {
                border: none;
                background: #1e1e1e;
                alternate-background-color: #242424;
                color: #eeeeee;
                gridline-color: #333333;
                selection-background-color: #3d5a80;
                selection-color: white;
            }

            QHeaderView::section {
                background: #292929;
                color: #dddddd;
                padding: 6px 12px;
                border: none;
                border-right: 1px solid #3a3a3a;
                border-bottom: 1px solid #3a3a3a;
            }

            QScrollBar:vertical {
                background: transparent;
                width: 10px;
                margin: 2px;
            }

            QScrollBar::handle:vertical {
                background: #555555;
                border-radius: 5px;
                min-height: 30px;
            }

            QScrollBar::handle:vertical:hover {
                background: #777777;
            }

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }

            QScrollBar:horizontal {
                background: transparent;
                height: 10px;
                margin: 2px;
            }

            QScrollBar::handle:horizontal {
                background: #555555;
                border-radius: 5px;
                min-width: 30px;
            }

            QScrollBar::handle:horizontal:hover {
                background: #777777;
            }

            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            """
        )

        # Application-wide event filter.
        app = QApplication.instance()

        if app is not None:
            app.installEventFilter(self)

    # Resize direction
    def get_resize_direction(self, pos):
        x = pos.x()
        y = pos.y()

        w = self.width()
        h = self.height()
        b = self.BORDER

        left = x <= b
        right = x >= w - b
        top = y <= b
        bottom = y >= h - b

        if top and left:
            return "top-left"

        if top and right:
            return "top-right"

        if bottom and left:
            return "bottom-left"

        if bottom and right:
            return "bottom-right"

        if left:
            return "left"

        if right:
            return "right"

        if top:
            return "top"

        if bottom:
            return "bottom"

        return None

    # Cursor
    def update_cursor(self, direction):

        if direction in ("top-left", "bottom-right"):
            self.setCursor(
                Qt.CursorShape.SizeFDiagCursor
            )

        elif direction in ("top-right", "bottom-left"):
            self.setCursor(
                Qt.CursorShape.SizeBDiagCursor
            )

        elif direction in ("left", "right"):
            self.setCursor(
                Qt.CursorShape.SizeHorCursor
            )

        elif direction in ("top", "bottom"):
            self.setCursor(
                Qt.CursorShape.SizeVerCursor
            )

        else:
            self.setCursor(
                Qt.CursorShape.ArrowCursor
            )

    # Resize
    def _apply_resize(self, global_pos):

        if (
            not self.resizing
            or self.original_geometry is None
            or self.resize_start is None
        ):
            return

        delta = global_pos - self.resize_start

        geometry = QRect(self.original_geometry)

        direction = self.resize_direction

        # Left
        if "left" in direction:
            new_left = (
                self.original_geometry.left()
                + delta.x()
            )

            max_left = (
                self.original_geometry.right()
                - self.minimumWidth()
                + 1
            )

            new_left = min(new_left, max_left)

            geometry.setLeft(new_left)

        # Right
        if "right" in direction:
            new_right = (
                self.original_geometry.right()
                + delta.x()
            )

            min_right = (
                self.original_geometry.left()
                + self.minimumWidth()
                - 1
            )

            new_right = max(new_right, min_right)

            geometry.setRight(new_right)

        # Top
        if "top" in direction:
            new_top = (
                self.original_geometry.top()
                + delta.y()
            )

            max_top = (
                self.original_geometry.bottom()
                - self.minimumHeight()
                + 1
            )

            new_top = min(new_top, max_top)

            geometry.setTop(new_top)

        # Bottom
        if "bottom" in direction:
            new_bottom = (
                self.original_geometry.bottom()
                + delta.y()
            )

            min_bottom = (
                self.original_geometry.top()
                + self.minimumHeight()
                - 1
            )

            new_bottom = max(new_bottom, min_bottom)

            geometry.setBottom(new_bottom)

        self.setGeometry(geometry)

    # Global mouse event filter
    def eventFilter(self, watched, event):

        event_type = event.type()

        # Only handle widgets belonging to this window.
        if (
            not isinstance(watched, QWidget)
            or watched.window() is not self
        ):
            return super().eventFilter(watched, event)

        # Mouse press
        if event_type == QEvent.Type.MouseButtonPress:

            if (
                event.button()
                == Qt.MouseButton.LeftButton
                and not self.isMaximized()
            ):

                global_pos = (
                    event.globalPosition().toPoint()
                )

                local_pos = self.mapFromGlobal(
                    global_pos
                )

                direction = self.get_resize_direction(
                    local_pos
                )

                if direction:

                    self.resizing = True
                    self.resize_direction = direction

                    self.resize_start = global_pos

                    self.original_geometry = (
                        self.geometry()
                    )

                    self.update_cursor(direction)

                    # Consume only the initial press.
                    #
                    # This prevents TitleBar from thinking
                    # that this is a window-drag operation.
                    return True

        # Mouse move
        elif event_type == QEvent.Type.MouseMove:

            global_pos = (
                event.globalPosition().toPoint()
            )

            # IMPORTANT:
            #
            # Do NOT return True here.
            #
            # Returning True prevents QPushButton from
            # receiving hover events.
            if self.resizing:

                self._apply_resize(global_pos)

                # Let the original widget receive the event too.
                return False

            # Normal cursor update
            if not self.isMaximized():

                local_pos = self.mapFromGlobal(
                    global_pos
                )

                direction = self.get_resize_direction(
                    local_pos
                )

                self.update_cursor(direction)

        # Mouse release
        elif event_type == QEvent.Type.MouseButtonRelease:

            if (
                self.resizing
                and event.button()
                == Qt.MouseButton.LeftButton
            ):

                self.resizing = False
                self.resize_direction = None
                self.resize_start = None
                self.original_geometry = None

                global_pos = (
                    event.globalPosition().toPoint()
                )

                local_pos = self.mapFromGlobal(
                    global_pos
                )

                self.update_cursor(
                    self.get_resize_direction(
                        local_pos
                    )
                )

                # Do not consume release.
                return False

        # Wheel (mouse scroll) - support Shift+wheel => horizontal scroll
        elif event_type == QEvent.Type.Wheel:

            # Only act for widgets in this window (same guard as above)
            # If Shift is held, translate vertical wheel movement into
            # horizontal scrolling for the table view so users can hold
            # Shift and scroll to move horizontally.
            try:
                modifiers = event.modifiers()
            except Exception:
                modifiers = None

            if modifiers is not None and (modifiers & Qt.KeyboardModifier.ShiftModifier):
                delta = event.angleDelta()

                # Prefer any horizontal delta first, otherwise use vertical
                dx = delta.x()
                dy = delta.y()

                scroll_amount = dx if dx != 0 else dy

                # If there's nothing to scroll, let the event pass through.
                if scroll_amount == 0:
                    return False

                hbar = self.table.horizontalScrollBar()

                # Map Qt's angleDelta (120 per notch) to scrollbar steps.
                # Use the scrollbar's singleStep to scale the movement.
                step = int((scroll_amount / 120.0) * hbar.singleStep())

                # Invert sign so wheel-up moves left/right naturally.
                hbar.setValue(hbar.value() - step)

                # Consume the event.
                return True

        return super().eventFilter(watched, event)

    # ============================================================
    # Cleanup
    # ============================================================

    def closeEvent(self, event):

        app = QApplication.instance()

        if app is not None:
            app.removeEventFilter(self)

        super().closeEvent(event)


def show_tables(
    dfs: DataSource,
    titles=None,
    title="Polars DataFrames",
    traffic_config=None,
):
    """Show a list of Polars DataFrames as tabs in one viewer window."""
    silence_wayland_text_input_warning()
    app = QApplication.instance()

    if app is None:
        app = QApplication(sys.argv)

    dataframes = list(dfs)
    if not dataframes:
        raise ValueError("show_tables requires at least one DataFrame")

    if titles is None:
        tab_titles = [f"Table {index}" for index in range(1, len(dataframes) + 1)]
    else:
        tab_titles = list(titles)
        if len(tab_titles) != len(dataframes):
            raise ValueError("titles must contain one title per DataFrame")

    window = DataFrameWindow(
        dataframes[0],
        title=title,
        traffic_config=traffic_config,
    )

    tabs = QTabWidget()
    for dataframe, tab_title in zip(dataframes, tab_titles):
        table = QTableView()
        table.setModel(PolarsModel(dataframe))
        configure_table(table)
        tabs.addTab(table, tab_title)

    central = QWidget()
    layout = QVBoxLayout(central)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    layout.addWidget(window.titlebar)
    layout.addWidget(tabs)
    window.setCentralWidget(central)

    window.table = cast(QTableView, tabs.currentWidget())
    tabs.currentChanged.connect(
        lambda index: setattr(
            window,
            "table",
            cast(QTableView, tabs.widget(index)),
        )
    )

    window.show()

    app.exec()


def show(df: DataSource, title="Polars DataFrame", traffic_config=None):
    """
    Show a DataFrame in the custom viewer window using the default traffic
    button configuration.
    """
    silence_wayland_text_input_warning()
    app = QApplication.instance()

    if app is None:
        app = QApplication(sys.argv)

    window = DataFrameWindow(df, title, traffic_config)
    window.show()

    app.exec()
