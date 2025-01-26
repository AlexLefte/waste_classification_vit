import os
import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import numpy as np
from utils import *
from time import time
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, dataloader, class_names, device, output_path):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.inference_mode():
        inference_time = []
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            start = time()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).logits
            _, preds = torch.max(outputs, 1)
            stop = time()
            inference_time.append(stop-start)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"Mean inference time: {np.mean(inference_time[5:])}")
    # Realizarea si salvarea matricei de confuzie
    cm = confusion_matrix(all_labels, all_preds)
    cm_path = os.path.join(output_path, f'cnf_matrix.png')
    save_confusion_matrix(cm, class_names, cm_path)
    
    # # Realizarea si salvarea raportului de clasificare
    # class_report = classification_report(all_labels, all_preds, target_names=class_names)
    # with open(os.path.join(output_path, f"classification_report.txt"), "w") as file:
    #     file.write(class_report)
    # print("\nClassification Report:\n", class_report)

    return all_preds, all_labels

def main():
    parser = argparse.ArgumentParser(description="Test Model on Custom Dataset")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--model_name", type=str, default="vit-base-patch16-224", 
                        choices=["vit-base-patch16-224", "vit-base-patch32-224-in21k", "vit-large-patch16-224", "vit-large-patch32-224-in21k"],
                        help="Pre-trained model to use (default: vit-base-patch16-224)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for testing")
    parser.add_argument("--output", type=str, default="test_output", help="Batch size for testing")
    args = parser.parse_args()

    # Transformările pentru imagini
    transform = get_image_transform()

    # Verificare disponibilitate GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Incarcarea modelului
    model = load_vit_model(args.model_path, args.model_name, 10)
    model.to(device)

    # Creare set de date
    class_names = sorted([d for d in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, d))])
    X, Y = prepare_dataset(args.data_path, class_names)
    dataset = CustomImageDataset(args.data_path, X, Y, transform)
    dataloader = DataLoader(dataset, batch_size=32)

    # Creeaza director de output
    output_path = args.output
    os.makedirs(output_path, exist_ok=True)

    # Efectuează evaluarea
    class_names = list(get_class_dict().values())
    evaluate_model(model, dataloader, class_names, device, output_path)

if __name__ == '__main__':
    main()
