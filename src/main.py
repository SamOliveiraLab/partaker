from PySide6.QtWidgets import QApplication
import os
import sys

from ui import App

# base_path = getattr(
#     sys, '_MEIPASS', os.path.dirname(
#         os.path.abspath(__file__)))
# ui_file_path = os.path.join(base_path, "ui.py")

# print("Looking for ui.py:", os.path.exists(ui_file_path))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    tabWidgetApp = App()
    tabWidgetApp.show()
    sys.exit(app.exec())
