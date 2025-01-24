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
from safetensors.torch import load_file
from PIL import Image
from torchvision.models import vit_b_16
import cv2
from transformers import ViTForImageClassification, AutoImageProcessor, AutoModel, AutoConfig
from src.test_inference import *

class ViTClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Waste Classification App")
        self.setGeometry(100, 100, 900, 700)

        self.init_ui()
        try:
         self.model = load_vit_model(model_path="model.safetensors", num_labels=10)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None       
        try:
            self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        except Exception as e:
            print(f"Error loading processor: {str(e)}")
            self.processor = None
        except Exception as e:
            print(f"Error loading processor: {str(e)}")
            self.processor = None
            
        self.camera = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.capture_mode = False 
        self.frame = None
        self.image_path = None  # Initialize image_path

    def init_ui(self):
        # Main Layout
        self.main_widget = QWidget()
        self.main_widget.setStyleSheet("background-color: #CEEDDB;")
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(20, 20, 20, 20)

        # Image display
        self.image_label = QLabel("No image loaded") 
        self.image_label.setMinimumSize(480, 480) 
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #8aa29e;")
        self.image_label.setFont(QFont("Times New Roman", 14))
        self.image_label.setFixedSize(480, 480)
        self.layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(5)

        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)
        self.upload_button.setFixedSize(160, 70)
        self.style_button(self.upload_button)
        button_layout.addWidget(self.upload_button)

        self.capture_button = QPushButton("Capture Image")
        self.capture_button.clicked.connect(lambda: (self.access_camera() if self.camera is None else self.capture_and_classify()))
        self.capture_button.setEnabled(True)
        self.capture_button.setFixedSize(160, 70)
        self.style_button(self.capture_button)
        button_layout.addWidget(self.capture_button)

        self.classify_button = QPushButton("Classify Image")
        self.classify_button.clicked.connect(self.classify_image)
        self.classify_button.setEnabled(True)
        self.classify_button.setFixedSize(160, 70)
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
            "    padding: 5px 5px;"
            "    text-align: center;"
            "    font-size: 14px;"
            "    font-weight: bold;"
            "    margin: 2px 2px;"
            "    border-radius: 4px;"
            "    margin-Left: 1px;"
            "    margin-Right: 1px;"
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
                self.frame = None
                self.image_label.setPixmap(pixmap.scaled(600, 480, Qt.KeepAspectRatio))
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
                # salveaza frame-ul curent
                frame_temp = frame.copy()  # Store the current frame
                h, w, _ = frame_temp.shape
                start_x = (w - 480) // 2
                start_y = (h - 480) // 2
                self.frame = frame_temp[start_y:start_y+480, start_x:start_x+480]

                # Afisare imagine in widget
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame_rgb.shape
                qimg = QImage(frame_rgb.data, width, height, channel * width, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)  # Correctly derive pixmap from qimg
                self.image_label.setPixmap(pixmap.scaled(600, 480, Qt.KeepAspectRatio))

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
                self.image_label.setPixmap(pixmap.scaled(600, 480, Qt.KeepAspectRatio))
                self.timer.stop()  # Stop the camera updates
        else:
            self.close_camera()  # Reset on the third click
            self.image_label.setText("Camera reset. Click to start again.")
            self.result_label.setText("Camera reset.")
            self.capture_mode = False
            
    def classify_image(self):
        try:
            # Check if the model and processor are loaded
            if self.model is None:
                print("Debug: Model is not loaded.")
                self.result_label.setText("Error: Model not loaded.")
                return
            if self.processor is None:
                print("Debug: Processor is not loaded.")
                self.result_label.setText("Error: Processor not loaded.")
                return

            # Check if a frame is frozen
            if self.capture_mode and self.frame is not None:
                print("Debug: Processing frozen frame.")
                frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
            elif self.image_path:
                print("Debug: Processing uploaded image.")
                # Load uploaded image
                image = Image.open(self.image_path).convert("RGB")
            else:
                print("Debug: No image or frame available for classification.")
                self.result_label.setText("Error: No image or frame to classify.")
                return

            # Predict using the model and processor
            predicted_class = predict_image(self.model, self.processor, image)

            # Map class index to label
            class_label = self.get_class_label(predicted_class)
            self.result_label.setText(f"Result: {class_label}")
        except FileNotFoundError:
            print("Debug: Class label file not found.")
            self.result_label.setText("Error: Class label file not found.")
        except AttributeError as e:
            print(f"Debug: Attribute error - {str(e)}")
            self.result_label.setText("Error: Model or image processing attribute missing.")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            self.result_label.setText("Error: An unexpected issue occurred during classification.")

    def get_class_label(self, class_index):
        class_dict = {
                0: "battery",
                1: "biological",
                2: "cardboard",
                3: "clothes",
                4: "glass",
                5: "metal",
                6: "paper",
                7: "plastic",
                8: "shoes",
                9: "trash"
            }
        return class_dict[class_index]

    def closeEvent(self, event):
        self.close_camera()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ViTClassifierApp()
    window.show()
    sys.exit(app.exec_())
