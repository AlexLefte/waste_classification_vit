from torchvision import transforms, models
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
import random
import cv2
import seaborn as sns
from PIL import Image
from transformers import ViTForImageClassification, AutoImageProcessor, AutoModel, AutoConfig

#### Models ####
# Function to load the pre-trained model
# def load_model(model_path, model_name, device):
#     if model_name == "vit_b_16":
#         model = models.vit_b_16(pretrained=False)  # Don't load pretrained weights, we'll load our own
#     elif model_name == "vit_b_32":
#         model = models.vit_b_32(pretrained=False)
#     elif model_name == "vit_l_16":
#         model = models.vit_l_16(pretrained=False)
#     elif model_name == "vit_l_32":
#         model = models.vit_l_32(pretrained=False)
#     num_classes = 10
#     model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)

#     # Load the model's weights from the given path
#     model.load_state_dict(torch.load(model_path))
#     model = model.to(device)
#     model.eval()  # Set the model to evaluation mode
#     return model

def load_vit_model(model_path, model_name, num_labels):
    """
    Load a ViT model with weights from a .safetensors file.
    :param model_path: Path to the .safetensors file
    :param num_labels: Number of output classes
    :return: Loaded model
    """
    # Inițializează configurația cu numărul de etichete
    config = AutoConfig.from_pretrained(
        f"google/{model_name}",
        num_labels=num_labels
    )
    
    # Inițializează modelul cu configurația personalizată
    model = ViTForImageClassification(config)
    
    # Încarcă ponderile din fișierul .pth
    state_dict = torch.load(model_path)  # Încarcă ponderile din fișierul .pth
    model.load_state_dict(state_dict)  # Încarcă ponderile în model
    
    model.eval()  # Setează modelul în modul de evaluare
    return model


### Transforms ###
def get_image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

### Data ###
# Set de date custom
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, images, labels, transform=None):
        self.root_dir = root_dir
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx]

def prepare_dataset(root_dir, class_names):
    # Adună toate imaginile și etichetele asociate
    images = []  # Listă pentru stocarea căilor către imagini
    labels = []  # Listă pentru stocarea etichetelor numerice corespunzătoare claselor
    
    for label, class_name in enumerate(class_names):
        # Pentru fiecare clasă (numită în `class_names`), obține calea completă a folderului clasei
        class_path = os.path.join(root_dir, class_name)
        
        # Creează o listă cu toate imaginile din folderul clasei (filtrează fișierele pe baza extensiilor acceptate)
        class_images = [
            os.path.join(class_name, img)  # Salvează calea relativă (față de directorul principal)
            for img in os.listdir(class_path)  # Iterează toate fișierele din director
            if img.lower().endswith(('.png', '.jpg', '.jpeg'))  # Selectează doar imaginile cu extensii acceptate
        ]
        
        # Amestecă imaginile din această clasă pentru a evita părtinirea
        random.shuffle(class_images)
        
        # Adaugă căile imaginilor în lista principală `images`
        images.extend(class_images)
        
        # Adaugă etichetele numerice pentru toate imaginile din această clasă (reprezentată de `label`)
        labels.extend([label] * len(class_images))  # Eticheta este repetată pentru fiecare imagine

    return images, labels


# Dictionarul de clase
def get_class_dict():
    return {
        0: "biodegradabil",
        1: "carton",
        2: "haine",
        3: "electronice",
        4: "sticlă",
        5: "metal",
        6: "hârtie",
        7: "plastic",
        8: "pantofi",
        9: "nereciclabil"
    }
    

### Metrics ###
def save_losses_plot(train_losses, val_losses, exp_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(exp_path, 'loss_plot.png'))
    plt.close()

def save_confusion_matrix(cm, class_names, path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()