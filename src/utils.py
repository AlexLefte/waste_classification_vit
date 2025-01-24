from torchvision import transforms, models
import torch


#### Models ####
# Function to load the pre-trained model
def load_model(model_path, model_name, device):
    if model_name == "vit_b_16":
        model = models.vit_b_16(pretrained=False)  # Don't load pretrained weights, we'll load our own
    elif model_name == "vit_b_32":
        model = models.vit_b_32(pretrained=False)
    elif model_name == "vit_l_16":
        model = models.vit_l_16(pretrained=False)
    elif model_name == "vit_l_32":
        model = models.vit_l_32(pretrained=False)
    num_classes = 10
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)

    # Load the model's weights from the given path
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model


### Transforms ###
def get_image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])