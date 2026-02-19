import sys

# Lazy Qt imports: only load when main() is called. This allows CLI tools
# (e.g. partaker-benchmark) to run without loading Qt/GPU-heavy deps.


def main():
    from PySide6.QtWidgets import QApplication
    from nd2_analyzer.ui import App

    app = QApplication(sys.argv)
    app.setApplicationName("Partaker")
    mainWin = App()
    mainWin.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
