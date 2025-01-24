import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import argparse
from utils import *


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


def train_model(model, dataloaders, criterion, optimizer, num_epochs=10, device='cpu', exp_path='experiments/experiment', model_name='best_model.pth'):   
    # Resetarea celui mai bun model
    best_val_loss = float('inf')
    best_model_wts = None

    # Lista de functii de cost (pentru grafic)
    train_losses = []
    val_losses = []

    # Bucla de antrenare
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        # Antrenare
        model.train() 
        train_loss = 0.0
        for inputs, labels in tqdm(dataloaders['train'], desc="Training", leave=False):
            # Comutarea datelor pe cpu/gpu
            inputs, labels = inputs.to(device), labels.to(device)

            # Resetarea gradientilor
            optimizer.zero_grad()

            # Inferenta prin retea
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Optimizarea parametrilor
            loss.backward()
            optimizer.step()

            # Acumulare functie de cost
            train_loss += loss.item()

        # Validare
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        # Dezactiveaza calculul gradientilor pe parcursul evaluarii
        with torch.inference_mode():
            for inputs, labels in tqdm(dataloaders['val'], desc="Validation", leave=False):
                # Comuta datele pe cpu/gpu
                inputs, labels = inputs.to(device), labels.to(device)

                # Predictie
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Acumulare functie de cost
                val_loss += loss.item()

                # Calcul acuratete
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculeaza metricile pe setul de validare
        train_loss /= len(dataloaders['train'])
        val_loss /= len(dataloaders['val'])
        val_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Verifica cel mai bun model pe baza functiei de cost
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict().copy()
            torch.save(best_model_wts, os.path.join(exp_path, model_name))
            print(f"New best model saved with Validation Loss: {best_val_loss:.4f}")        

    # Afiseaza graficul func'iilor de cost
    save_losses_plot(train_losses, val_losses, exp_path)

    # Intoarce cel mai bun model
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    return model

def evaluate_model(model, dataloader, dataloader_name, exp_path):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    all_preds = []
    all_labels = []
    
    # Perform prediction
    with torch.inference_mode():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Realizarea si salvarea matricei de confuzie
    class_names = get_class_dict()
    class_names = list(class_names.values())
    cm = confusion_matrix(all_labels, all_preds)
    cm_path = os.path.join(exp_path, f'{dataloader_name}_cnf_matrix.png')
    save_confusion_matrix(cm, class_names, cm_path)
    
    # Realizarea si salvarea raportului de clasificare
    class_report = classification_report(all_labels, all_preds, target_names=class_names)
    with open(os.path.join(exp_path, f"{dataloader_name}_classification_report.txt"), "w") as file:
        file.write(class_report)
    print("\nClassification Report:\n", class_report)


def main():
    parser = argparse.ArgumentParser(description="Train Vision Transformer on Custom Dataset")
    parser.add_argument("--exp_name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--data_path", type=str, default="dataset/images", help="Path towards the dataset")
    parser.add_argument("--model_name", type=str, default="vit_b_16", 
                        choices=["vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32"],
                        help="Vision Transformer model to use (from torchvision)")
    args = parser.parse_args()

    # Parametri de configurare
    data_path = args.data_path
    exp_name = args.exp_name
    model_name = args.model_name

    # Creaza directorul de experimente
    os.makedirs('experiments', exist_ok=True)
    experiment_path = os.path.join('experiments', exp_name)
    os.makedirs(experiment_path, exist_ok=True)

    # Modifică transforms
    transform = get_image_transform()
    
    # Path pentru split-uri
    class_names = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    splits_path = os.path.join(data_path, "splits.json")
    # Salvează sau încarcă split-urile
    if os.path.exists(splits_path):
        X_train, X_val, X_test, y_train, y_val, y_test = load_splits(splits_path)
    else:
        # Prepare dataset
        images, labels = prepare_dataset(data_path, class_names)

        # Creează split-urile dacă nu există deja
        X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(images, labels)
        save_splits(splits_path, X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Create seturile de date si "data loaders"
    train_dataset = CustomImageDataset(data_path, X_train, y_train, transform)
    val_dataset = CustomImageDataset(data_path, X_val, y_val, transform)
    test_dataset = CustomImageDataset(data_path, X_test, y_test, transform)
    
    # DataLoaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=32),
        'test': DataLoader(test_dataset, batch_size=32)
    }
    
    # Initializarea modelului
    model_func = getattr(models, model_name)
    model = model_func(pretrained=True)
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Linear(num_ftrs, len(class_names))
    
    # Definirea functiei de cost si a functiei de optimizare
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Verificare disponibilitate gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Antrenare model
    trained_model = train_model(model, dataloaders, criterion, optimizer, device=device, exp_path=experiment_path)

    # Evaluare pe cele 3 seturi de date
    for mode, dl in dataloaders.items():
        evaluate_model(trained_model, dl, mode, experiment_path)

if __name__ == '__main__':
    main()