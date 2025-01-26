import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utils import *
import argparse
import math
from time import time

# Definirea clasei Dataset pentru încărcarea imaginilor
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_path


def load_images_from_subfolders(input_folder):
    image_paths = []
    # Recursiv, parcurge toate directoarele și subdirectoarele
    for root, dirs, files in os.walk(input_folder):
        for fname in files:
            # Verifică dacă fișierul este o imagine (png, jpg, jpeg)
            if fname.lower().endswith(('png', 'jpg', 'jpeg')):
                image_paths.append(os.path.join(root, fname))  # Calea completă a imaginii
    return image_paths


# Funcția care încarcă modelul pre-antrenat și face inferența
def inference(model, dataloader, device, class_names, save_csv=True, save_figures=True, output_folder='inference_output', max_images_per_fig=9):
    model.eval()
    image_paths = []

    # Creaza un director pentru resultate
    os.makedirs(output_folder, exist_ok=True)
    
    # Creare CSV pentru rezultate
    if save_csv:
        results = []

    # Index figura
    fig_index = 0

     # Realizează predicții
    with torch.inference_mode():
        inf_time = []
        for inputs, paths in dataloader:
            start = time()
            inputs = inputs.to(device)
            
            # Predicție
            outputs = model(inputs).logits
            probs, preds = torch.max(torch.nn.functional.softmax(outputs, dim=1), dim=1)
            
            for i, path in enumerate(paths):
                image_paths.append(path)
                
                # Adaugă rezultatele în CSV dacă este necesar
                if save_csv:
                    results.append([path, class_names[preds[i].item()], round(probs[i].item() * 100, 2)])
                
                # Dacă dorim să salvăm imagini într-un grid
                if save_figures:
                    img = Image.open(path)
                    
                    # Dacă există o singură imagine, salvăm într-o figură 1x1
                    if len(paths) == 1:
                        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
                        ax.imshow(img)
                        ax.set_title(f"{os.path.basename(path)}\nPredicted: {class_names[preds[i].item()]} ({probs[i].item() * 100:.2f}%)")
                        ax.axis('off')
                        figure_path = os.path.join(output_folder, f"predictions_{i + 1}.png")
                        plt.savefig(figure_path)
                        plt.close()
                    else:
                        # Dacă numărul de imagini este mai mare decât max_images_per_fig, le punem în figuri separate
                        if i % max_images_per_fig == 0:
                            # Calculăm numărul de rânduri necesare, de exemplu pentru max_images_per_fig = 9 (3x3):
                            num_images = min(len(paths) - i, max_images_per_fig)  # Asigură-te că nu depășești numărul de imagini
                            num_rows = math.ceil(num_images / 3)
                            fig, axes = plt.subplots(num_rows, 3, figsize=(12, num_rows * 4))  # Înălțimea se ajustează în funcție de rânduri
                            axes = axes.flatten()

                        # Afișăm imaginea în grid
                        ax = axes[i % max_images_per_fig]
                        ax.imshow(img)
                        ax.set_title(f"{os.path.basename(path)}\nPredicted: {class_names[preds[i].item()]} ({probs[i].item()*100:.2f}%)")
                        ax.axis('off')

                        # Dacă am ajuns la maximul de imagini per figură, salvăm figura
                        if (i + 1) % max_images_per_fig == 0 or i == len(paths) - 1:
                            # Eliminăm axele goale
                            for j in range(i % max_images_per_fig + 1, len(axes)):
                                axes[j].axis('off')
                                
                            plt.tight_layout()
                            figure_path = os.path.join(output_folder, f"predictions_{fig_index + 1}.png")
                            fig_index += 1
                            plt.savefig(figure_path)
                            plt.close()
            end = time()
            inf_time.append(end - start)

    print(f"Mean inference time: {np.mean(inf_time[10:])}")

    # Dacă se cere CSV, salvează rezultatele
    if save_csv:
        csv_path = os.path.join(output_folder, 'results.csv')
        df = pd.DataFrame(results, columns=['Image Path', 'Prediction', 'Probability [%]'])
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")


# Funcția principală
def main():
    parser = argparse.ArgumentParser(description="Run inference on images using a pre-trained model")
    parser.add_argument('--input', type=str, help="Path to the folder with images to classify")
    parser.add_argument('--output_folder', type=str, default='output', help="Folder to save result images")
    parser.add_argument('--model_path', type=str, required=True, help="Pre-trained model path")
    parser.add_argument('--model_name', type=str, default="vit-base-patch16-224", 
                        choices=["vit-base-patch16-224", "vit-base-patch32-224-in21k", "vit-large-patch16-224", "vit-large-patch32-224-in21k"],
                        help="Pre-trained model to use (default: vit-base-patch16-224)")
    parser.add_argument('--save_csv', action='store_true', help="Whether to save prediction as csv")
    parser.add_argument('--save_figures', action='store_true', help="Whether to save prediction images with titles")

    args = parser.parse_args()
    
    # Parametri de configurare
    input_folder = args.input
    save_csv = args.save_csv
    output_folder = args.output_folder
    model_path = args.model_path
    model_name = args.model_name
    save_figures = args.save_figures

    # Verificare disponibilitate GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Incarcarea modelului
    model = load_vit_model(model_path, model_name, num_labels=10)
    model.to(device)

    # Încărcarea imaginilor din folderul de intrare
    # image_paths = [os.path.join(input_folder, fname) for fname in os.listdir(input_folder) if fname.lower().endswith(('png', 'jpg', 'jpeg'))]
    image_paths = load_images_from_subfolders(input_folder)
    random.shuffle(image_paths)
    
    # Definirea transformărilor pentru preprocesarea imaginilor
    transform = get_image_transform()

    # Creare DataLoader
    dataset = ImageDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Definire dictionar clase
    class_names = get_class_dict()

    # Realizează inferența
    inference(model, dataloader, device, class_names, save_csv=save_csv, save_figures=save_figures, output_folder=output_folder)


if __name__ == '__main__':
    main()
