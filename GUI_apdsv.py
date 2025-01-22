import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog,
    QHBoxLayout, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import torch
from torchvision import transforms
from PIL import Image
from torchvision.models import vit_b_16

class ViTClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Waste Classification App")
        self.setGeometry(200, 200, 800, 600)

        self.init_ui()
        self.model = self.load_vit_model()
        self.transform = self.get_image_transform()

    def init_ui(self):
        # Main Layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout()

        # Image display
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.layout.addWidget(self.image_label)

        # Buttons
        button_layout = QHBoxLayout()

        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)
        button_layout.addWidget(self.upload_button)

        self.classify_button = QPushButton("Classify Image")
        self.classify_button.clicked.connect(self.classify_image)
        self.classify_button.setEnabled(False)
        button_layout.addWidget(self.classify_button)

        self.layout.addLayout(button_layout)

        # Classification result
        self.result_label = QLabel("Result: N/A")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.result_label)

        self.main_widget.setLayout(self.layout)

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
            self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
            self.image_path = file_path
            self.classify_button.setEnabled(True)

    def classify_image(self):
        if not hasattr(self, 'image_path'):
            self.result_label.setText("Error: No image uploaded.")
            return

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

    def get_class_label(self, class_index):
        # Using ImageNet class labels
        with open("imagenet_classes.txt", "r") as f:
            labels = [line.strip() for line in f.readlines()]
        return labels[class_index]

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ViTClassifierApp()
    window.show()
    sys.exit(app.exec_())