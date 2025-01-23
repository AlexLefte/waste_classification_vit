import sys
from PyQt5.QtWidgets import (
    QSizePolicy,
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog,
    QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QTimer
import torch
from torchvision import transforms
from PIL import Image
from torchvision.models import vit_b_16
import cv2

class ViTClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Waste Classification App")
        self.setGeometry(400, 400, 800, 600)

        self.init_ui()
        self.model = self.load_vit_model()
        self.transform = self.get_image_transform()

        self.camera = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.capture_mode = False 
        self.frame = None

    def init_ui(self):
        # Main Layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(50, 50, 50, 50)

        # Image display
        self.image_label = QLabel("No image loaded") 
        self.image_label.setMinimumSize(625, 500) 
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #8aa29e;")
        self.image_label.setFont(QFont("Times New Roman", 14))
        self.image_label.setFixedSize(625, 500)
        self.layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        # Buttons
        button_layout = QHBoxLayout()

        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)
        self.style_button(self.upload_button)
        button_layout.addWidget(self.upload_button)

        self.capture_button = QPushButton("Capture and Classify")
        self.capture_button.clicked.connect(lambda: (self.access_camera() if self.camera is None else self.capture_and_classify()))
        self.capture_button.setEnabled(True)
        self.style_button(self.capture_button)
        button_layout.addWidget(self.capture_button)

        self.classify_button = QPushButton("Classify Image")
        self.classify_button.clicked.connect(self.classify_image)
        self.classify_button.setEnabled(True)
        self.style_button(self.classify_button)
        button_layout.addWidget(self.classify_button)

        self.layout.addLayout(button_layout)

        # Rezultat clasificare
        self.result_label = QLabel("Result: N/A")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Times New Roman", 14))
        self.result_label.setStyleSheet("color: black; font-weight: bold;")
        self.layout.addWidget(self.result_label)

        self.main_widget.setLayout(self.layout)

    def style_button(self, button):
        button.setStyleSheet(
            "QPushButton {"
            "    background-color: #bec5ad;"
            "    color: #0E402D;"
            "    border: none;"
            "    padding: 10px 20px;"
            "    text-align: center;"
            "    font-size: 12px;"
            "    margin: 4px 2px;"
            "    border-radius: 8px;"
            "}"
            "QPushButton:hover {"
            "    background-color: #519872;"
            "}"
        )

    def load_vit_model(self):
        model = vit_b_16(pretrained=True)
        model.eval()
        return model

    def get_image_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def upload_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.jpeg *.bmp)", options=options)

        if file_path:
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
                self.image_path = file_path
                self.classify_button.setEnabled(True)

    def access_camera(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                self.result_label.setText("Error: Camera cannot be accessed.")
                self.camera = None
                return

            self.timer.start(1)  
            self.capture_button.setEnabled(True)
            print("Camera started.")

    def update_frame(self):
        if self.camera:
            ret, frame = self.camera.read()
            if ret:
                self.frame = frame  # Store the current frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame_rgb.shape
                qimg = QImage(frame_rgb.data, width, height, channel * width, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)  # Correctly derive pixmap from qimg
                self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))

    def close_camera(self):
        if self.camera:
            self.timer.stop()
            self.camera.release()
            self.camera = None
            print("Camera closed.")

    def capture_and_classify(self):
        if self.camera is None:
            self.access_camera()  # Open the camera on the first click
            self.result_label.setText("Camera started. Click again to freeze the frame.")
        elif not self.capture_mode:
            self.capture_mode = True  # Freeze the frame on the second click
            self.result_label.setText("Frame frozen. Click again to reset.")
            if self.frame is not None:
                frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame_rgb.shape
                qimg = QImage(frame_rgb.data, width, height, channel * width, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
                self.timer.stop()  # Stop the camera updates
        else:
            self.close_camera()  # Reset on the third click
            self.image_label.setText("Camera reset. Click to start again.")
            self.result_label.setText("Camera reset.")
            self.capture_mode = False

                    
    def classify_image(self):
        if not hasattr(self, 'image_path'):
            self.result_label.setText("Error: No image uploaded.")
            return

        try:
            # Load image and preprocess
            image = Image.open(self.image_path).convert("RGB")
            input_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension

            # Perform inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                _, predicted_class = outputs.max(1)

            # Map class index to label
            class_label = self.get_class_label(predicted_class.item())
            self.result_label.setText(f"Result: {class_label}")
        except Exception as e:
            self.result_label.setText(f"Error: {str(e)}")

    def get_class_label(self, class_index):
        with open("imagenet_classes.txt", "r") as f:
            labels = [line.strip() for line in f.readlines()]
        return labels[class_index]

    def closeEvent(self, event):
        self.close_camera()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ViTClassifierApp()
    window.show()
    sys.exit(app.exec_())
