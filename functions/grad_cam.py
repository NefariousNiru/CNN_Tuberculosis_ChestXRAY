import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image

from CNN.DenseNet121WithCBAM import DenseNet121WithCBAM


# Grad-CAM function
def grad_cam(model, image_tensor, class_idx, target_layer):
    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer_module = dict(model.named_modules())[target_layer]
    target_layer_module.register_forward_hook(forward_hook)
    target_layer_module.register_backward_hook(backward_hook)

    output = model(image_tensor)
    target = output[:, class_idx]
    target.backward()

    gradients = gradients[0].cpu().data.numpy()
    activations = activations[0].cpu().data.numpy()

    weights = np.mean(gradients, axis=(2, 3), keepdims=True)
    grad_cam = np.sum(weights * activations, axis=1).squeeze()
    grad_cam = np.maximum(grad_cam, 0)
    heatmap = grad_cam / np.max(grad_cam)
    return heatmap

# Visualize function
def visualize_grad_cam(image_tensor, heatmap, original_image, class_name):
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_image, 0.5, heatmap_colored, 0.5, 0)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(original_image[..., ::-1])  # Convert BGR to RGB
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title(f"Grad-CAM Heatmap ({class_name})")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay[..., ::-1])  # Convert BGR to RGB
    plt.title("Overlay")
    plt.axis("off")
    plt.show()

def run_grad_cam():
    # Load dataset
    data_dir = "dataset/chest_XRAY/train"  # Path to your dataset
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    class_names = dataset.classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121WithCBAM(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load("model.pth"))  # Replace with your model checkpoint

    # Select two images from each class
    images_per_class = 2
    for class_idx, class_name in enumerate(class_names):
        count = 0
        for i, (image_tensor, label) in enumerate(dataset):
            if label == class_idx and count < images_per_class:
                image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
                original_image = cv2.imread(dataset.samples[i][0])  # Load original image as NumPy array

                # Model prediction
                output = model(image_tensor)
                predicted_class = torch.argmax(output, dim=1).item()

                # Generate Grad-CAM heatmap
                heatmap = grad_cam(model, image_tensor, predicted_class, target_layer='features')

                # Visualize Grad-CAM
                visualize_grad_cam(image_tensor, heatmap, original_image, class_name)
                count += 1
