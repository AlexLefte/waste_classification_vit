import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utils import load_model
import argparse

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

# Funcția care încarcă modelul pre-antrenat și face inferența
def inference(model, dataloader, device, class_names, output_csv=None, save_figures=False, output_folder='output', max_images_per_fig=9):
    model.eval()
    image_paths = []

    # Creaza un director pentru resultate
    os.makedirs(output_folder, exist_ok=True)
    
    # Creare CSV pentru rezultate
    if output_csv:
        results = []

     # Realizează predicții
    with torch.inference_mode():
        for inputs, paths in dataloader:
            inputs = inputs.to(device)
            
            # Predicție
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.argmax(outputs, 1).cpu().numpy()
            probs = probs[:, preds]
            
            for i, path in enumerate(paths):
                image_paths.append(path)
                
                # Adaugă rezultatele în CSV dacă este necesar
                if output_csv:
                    results.append([path, class_names[preds[i]], round(probs[i, preds[i]] * 100, 2)])
                
                # Dacă dorim să salvăm imagini într-un grid
                if save_figures:
                    img = Image.open(path)
                    
                    # Dacă numărul de imagini este mai mare decât max_images_per_fig, le punem în figuri separate
                    if i % max_images_per_fig == 0:
                        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
                        axes = axes.flatten()

                    # Afișăm imaginea în grid
                    ax = axes[i % max_images_per_fig]
                    ax.imshow(img)
                    ax.set_title(f"{os.path.basename(path)}\nPredicted: {class_names[preds[i]]} ({probs[i, preds[i]]*100:.2f})")
                    ax.axis('off')

                    # Dacă am ajuns la maximul de imagini per figură, salvăm figura
                    if (i + 1) % max_images_per_fig == 0 or i == len(paths) - 1:
                        # Eliminăm axele goale
                        for j in range(i % max_images_per_fig + 1, len(axes)):
                            axes[j].axis('off')

                        plt.tight_layout()
                        figure_path = os.path.join(output_folder, f"predictions_{(i // max_images_per_fig) + 1}.png")
                        plt.savefig(figure_path)
                        plt.close()

    # Dacă se cere CSV, salvează rezultatele
    if output_csv:
        df = pd.DataFrame(results, columns=['Image Path', 'Prediction', 'Probability'])
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")


# Funcția principală
def main():
    parser = argparse.ArgumentParser(description="Run inference on images using a pre-trained model")
    parser.add_argument('--input', type=str, help="Path to the folder with images to classify")
    parser.add_argument('--output_csv', type=str, default="results.csv", help="Path to save the output CSV with predictions")
    parser.add_argument('--output_folder', type=str, default='output', help="Folder to save result images")
    parser.add_argument('--model_path', type=str, required=True, help="Pre-trained model path")
    parser.add_argument('--model_name', type=str, default="vit_b_16", 
                        choices=["vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32"],
                        help="Pre-trained model to use (default: vit_b_16)")
    parser.add_argument('--save_figures', action='store_true', help="Whether to save prediction images with titles")

    args = parser.parse_args()
    
    # Parametri de configurare
    input_folder = args.input
    output_csv = args.output_csv
    output_folder = args.output_folder
    model_path = args.model_path
    model_name = args.model_name
    save_figures = args.save_figures

    # Verificare disponibilitate GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Incarcarea modelului
    model = load_model(model_path, model_name, device)

    # Încărcarea imaginilor din folderul de intrare
    image_paths = [os.path.join(input_folder, fname) for fname in os.listdir(input_folder) if fname.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    # Definirea transformărilor pentru preprocesarea imaginilor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # Creare DataLoader
    dataset = ImageDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Definire dictionar clase
    class_names = {
        0: "biologic",
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
    
    # Realizează inferența
    inference(model, dataloader, device, class_names, output_csv=output_csv, save_figures=save_figures, output_folder=output_folder)


if __name__ == '__main__':
    main()
