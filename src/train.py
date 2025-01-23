import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import random
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

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
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx]

def prepare_dataset(root_dir, class_names):
    # Collect all images and labels
    images = []
    labels = []
    
    for label, class_name in enumerate(class_names):
        class_path = os.path.join(root_dir, class_name)
        class_images = [os.path.join(class_name, img) for img in os.listdir(class_path) 
                        if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(class_images)
        images.extend(class_images)
        labels.extend([label] * len(class_images))
    
    return images, labels

def stratified_split(images, labels, test_size=0.1, val_size=0.1):
    # First split: train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        images, labels, 
        test_size=test_size, 
        stratify=labels, 
        random_state=42
    )
    
    # Second split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=val_size/(1-test_size), 
        stratify=y_train_val, 
        random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_splits(save_path, X_train, X_val, X_test, y_train, y_val, y_test):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    splits = {
        "train": {"images": X_train, "labels": y_train},
        "val": {"images": X_val, "labels": y_val},
        "test": {"images": X_test, "labels": y_test},
    }
    with open(save_path, 'w') as f:
        json.dump(splits, f)
    print(f"Splits saved to {save_path}")

def load_splits(load_path):
    with open(load_path, 'r') as f:
        splits = json.load(f)
    print(f"Splits loaded from {load_path}")
    return (splits['train']['images'], splits['val']['images'], splits['test']['images'],
            splits['train']['labels'], splits['val']['labels'], splits['test']['labels'])


def plot_losses(train_losses, val_losses, exp_path):
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


def train_model(model, dataloaders, criterion, optimizer, num_epochs=20, device='cpu', exp_path='experiments/experiment', save_path='best_model.pth'):   
    # Define best model
    best_val_loss = float('inf')
    best_model_wts = None

    # Store losses
    train_losses = []
    val_losses = []

    # Iterate through the specified number of epochs
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        # Training phase
        model.train() 
        train_loss = 0.0

        # Wrap the training loop in a tqdm progress bar
        for inputs, labels in tqdm(dataloaders['train'], desc="Training", leave=False):
            # Move data to the appropriate device
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients from the previous step
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate training loss
            train_loss += loss.item()

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0

        # Disable gradient computation during validation for efficiency
        with torch.inference_mode():
            # Wrap the validation loop in a tqdm progress bar
            for inputs, labels in tqdm(dataloaders['val'], desc="Validation", leave=False):
                # Move data to the appropriate device
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Accumulate validation loss
                val_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate and print epoch statistics
        train_loss /= len(dataloaders['train'])
        val_loss /= len(dataloaders['val'])
        val_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Save best model based on the loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict().copy()
            torch.save(best_model_wts, save_path)
            print(f"New best model saved with Validation Loss: {best_val_loss:.4f}")        

    # Plot loss curves
    plot_losses(train_losses, val_losses, exp_path)

    # Return best model
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    return model

def evaluate_model(model, dataloader, dataloader_name, class_names, exp_path):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_path, f'{dataloader_name}_cnf_matrix.png'))
    plt.close()
    
    # Classification Report
    class_report = classification_report(all_labels, all_preds, target_names=class_names)
    # Save the classification report to a text file
    with open(os.path.join(exp_path, f"{dataloader_name}_classification_report.txt"), "w") as file:
        file.write(class_report)
    print("\nClassification Report:\n", class_report)


def min_max_normalize(image):
    """
    Normalizează imaginea la intervalul [0, 1]
    
    Args:
        image (numpy.ndarray): Imagine color în format RGB
    
    Returns:
        numpy.ndarray: Imagine normalizată
    """
    min_val = image.min()
    max_val = image.max()
    
    # Evită împărțirea la zero
    if min_val == max_val:
        return np.zeros_like(image, dtype=np.float32)
    
    normalized = (image - min_val) / (max_val - min_val)
    return normalized.astype(np.float32)


def main():
    # Configuration
    root_dir = '/home/alex/Projects/Waste_project/dataset_final'  # Update this path
    exp_name = 'base_vit_224_16'

    # Create experiment directory
    experiment_path = os.path.join('experiments', exp_name)
    os.makedirs(experiment_path, exist_ok=True)

    # Modifică transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])
    
    # Path pentru split-uri
    class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    splits_path = "dataset/splits.json"
    # Salvează sau încarcă split-urile
    if os.path.exists(splits_path):
        X_train, X_val, X_test, y_train, y_val, y_test = load_splits(splits_path)
    else:
        # Prepare dataset
        images, labels = prepare_dataset(root_dir)

        # Creează split-urile dacă nu există deja
        X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(images, labels)
        save_splits(splits_path, X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Create datasets
    train_dataset = CustomImageDataset(root_dir, X_train, y_train, transform)
    val_dataset = CustomImageDataset(root_dir, X_val, y_val, transform)
    test_dataset = CustomImageDataset(root_dir, X_test, y_test, transform)
    
    # DataLoaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=32),
        'test': DataLoader(test_dataset, batch_size=32)
    }
    
    # Model setup
    model = models.vit_b_16(pretrained=True)
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Linear(num_ftrs, len(class_names))
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Training
    trained_model = train_model(model, dataloaders, criterion, optimizer, device=device, exp_path=experiment_path)

    # Evaluation
    for mode, dl in dataloaders.items():
        evaluate_model(trained_model, dl, mode, class_names, experiment_path)

if __name__ == '__main__':
    main()