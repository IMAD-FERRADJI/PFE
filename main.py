import sys
from PyQt5.QtWidgets import QApplication, QMessageBox
from modern_dashboard import ModernDashboardApp


def exception_hook(exctype, value, traceback):
    """
    Global function to catch unhandled exceptions.
    """
    print(f"Unhandled exception: {exctype}, {value}")
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)


if __name__ == "__main__":
    try:
        # Set exception hook
        sys._excepthook = sys.excepthook
        sys.excepthook = exception_hook

        app = QApplication(sys.argv)

        # Set application style to Fusion for a more modern look
        app.setStyle("Fusion")

        window = ModernDashboardApp()
        window.show()

        sys.exit(app.exec_())
    except Exception as e:
        print(f"Critical error: {e}")
        app = QApplication.instance()
        if not app:
            app = QApplication(sys.argv)
        QMessageBox.critical(None, "Critical Error", f"An unhandled exception occurred: {str(e)}")
        sys.exit(1)
