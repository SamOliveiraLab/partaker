from PySide6.QtWidgets import QApplication
from ui import TabWidgetApp
import sys

if __name__ == '__main__':
    app = QApplication(sys.argv)
    tabWidgetApp = TabWidgetApp()
    tabWidgetApp.show()
    sys.exit(app.exec())
