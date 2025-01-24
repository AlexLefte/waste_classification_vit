import torch
from safetensors.torch import load_file
from torchvision.models import vit_b_16  # Exemplu pentru un ViT predefinit
from torchvision import transforms
from PIL import Image
import os
from transformers import ViTForImageClassification, AutoImageProcessor, AutoModel, AutoConfig


# Global dict
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

# Load HuggingFace ViT model and processor
# Load the model and weights
def load_vit_model(model_path, num_labels):
    """
    Load a ViT model with weights from a .safetensors file.
    :param model_path: Path to the .safetensors file
    :param num_labels: Number of output classes
    :return: Loaded model
    """
    # Initialize the configuration with the required number of labels
    config = AutoConfig.from_pretrained(
        "google/vit-base-patch16-224",  # Use the same base model used during training
        num_labels=num_labels
    )
    
    # Initialize the model with the custom configuration
    model = ViTForImageClassification(config)
    
    # Load weights from the .safetensors file
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)  # Load weights into the model
    
    model.eval()  # Set the model to evaluation mode
    return model

# Image preprocessing using the AutoImageProcessor
def preprocess_image(image_path, processor):
    """
    Preprocess the image for ViT input.
    :param image_path: Path to the input image
    :param processor: Preprocessor from HuggingFace
    :return: Preprocessed image tensor
    """
    image = Image.open(image_path).convert("RGB")  # Open and convert to RGB
    inputs = processor(images=image, return_tensors="pt")  # Preprocess image
    return inputs

# Predict the class for a single image
def predict_image(model, processor, image):
    """
    Predict the class of a single image.
    :param model: Loaded ViT model
    :param processor: Preprocessor from HuggingFace
    :param image_path: Path to the input image
    :return: Predicted class
    """
    # inputs = preprocess_image(image_path, processor)
    inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)  # Perform inference
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return predicted_class

# Predict all images in a folder
def predict_folder(model, processor, folder_path, output_file="predictions.txt"):
    """
    Predict all images in a folder and save results to a file.
    :param model: Loaded ViT model
    :param processor: Preprocessor from HuggingFace
    :param folder_path: Path to the folder containing images
    :param output_file: File to save predictions
    """
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    with open(output_file, 'w') as f:
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            predicted_class = predict_image(model, processor, image_path)
            f.write(f"{image_file}: {predicted_class}\n")
            print(f"Predicted {image_file}: {class_dict[predicted_class]}")

# Main script
if __name__ == "__main__":
    folder_path = "C:\\Users\\Alex\\Documents\\Alex\\master\\APDSV\\Proiect_2_0\\dataset\\propriu\\uploaded"
    model_path = "C:\\Users\\Alex\\Documents\\Alex\\master\\APDSV\\Proiect_2_0\\dataset\\propriu\\model.safetensors" 
    num_labels = 10  # Replace with the number of your classes

    # Load model and processor
    model = load_vit_model(model_path, num_labels)
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

    # Predict all images in the folder
    predict_folder(model, processor, folder_path)
