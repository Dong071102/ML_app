import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from mainUI import Ui_MainWindow  # Replace 'main' with the name of your generated .py file

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
