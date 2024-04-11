import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout, QSizePolicy
from PyQt5.QtCore import pyqtSlot, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from fastai.vision.all import load_learner

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    prediction_signal = pyqtSignal(str, float)
    camera_started_signal = pyqtSignal()

    def run(self):
        self.cap = cv2.VideoCapture(0)
        self.camera_started_signal.emit()
        self.frame_count = 0
        while self.cap.isOpened():
            ret, cv_img = self.cap.read()
            if ret:
                self.frame_count += 1
                if self.frame_count == 10:
                    self.make_prediction(cv_img)
                    self.frame_count = 0
                self.change_pixmap_signal.emit(cv_img)

    def make_prediction(self, frame):
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pred_class, pred_idx, probs = model.predict(processed_frame)
        class_name_fastAI = str(pred_class)
        max_prob = probs[pred_idx].item() * 100  # Convert to percentage
        self.prediction_signal.emit(class_name_fastAI, max_prob)

    def stop(self):
        self.cap.release()

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesture Recognition Interface")
        self.setStyleSheet("background-color: #2c2c2c;")
        self.resize(1600, 900)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        layout.addWidget(self.image_label)

        self.prediction_label = QLabel('')
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setStyleSheet("QLabel { color : white; font: 16pt; }")
        layout.addWidget(self.prediction_label)

        button_layout = QHBoxLayout()
        button_style = "QPushButton { font: 18pt; background-color: #3c3c3c; color: white; }"

        self.start_button = QPushButton('Lancée la caméra')
        self.start_button.setStyleSheet(button_style)
        self.start_button.clicked.connect(self.start_camera)
        button_layout.addWidget(self.start_button)

        self.sign_list_button = QPushButton('Liste des signes')
        self.sign_list_button.setStyleSheet(button_style)
        self.sign_list_button.clicked.connect(self.show_sign_list)
        button_layout.addWidget(self.sign_list_button)

        self.camera_status_label = QLabel('')
        self.camera_status_label.setAlignment(Qt.AlignCenter)
        self.camera_status_label.setStyleSheet("QLabel { color : white; font: 24pt; }")
        layout.addWidget(self.camera_status_label)

        layout.addLayout(button_layout)
        self.setLayout(layout)

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.prediction_signal.connect(self.update_prediction)
        self.thread.camera_started_signal.connect(self.camera_started)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img.scaled(self.image_label.size(), Qt.KeepAspectRatio))

    @pyqtSlot(str, float)
    def update_prediction(self, class_name_fastAI, max_prob):
        self.prediction_label.setText(f"{class_name_fastAI}\nProbability: {max_prob:.2f}%")

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(convert_to_Qt_format)

    def start_camera(self):
        self.camera_status_label.setText("Lancement de la caméra, veuillez attendre...")
        self.thread.start()

    @pyqtSlot()
    def camera_started(self):
        self.camera_status_label.setText('')

    def show_sign_list(self):
        self.sign_list_window = SignListWindow()
        self.sign_list_window.show()
        self.sign_list_window.resize(1000, 800)

    def closeEvent(self, event):
        self.thread.stop()

class SignListWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Liste des Signes")
        self.label = QLabel(self)
        self.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.label.setScaledContents(True)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.pixmap = QPixmap('list_hands_signs.jpg')
        self.label.setPixmap(self.pixmap)

    def resizeEvent(self, event):
        scaled_pixmap = self.pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(scaled_pixmap)

if __name__ == "__main__":
    model_path = "models/fastai-v1.pth"
    model = load_learner(model_path)

    app = QApplication(sys.argv)
    main_window = App()
    main_window.show()
    sys.exit(app.exec_())
