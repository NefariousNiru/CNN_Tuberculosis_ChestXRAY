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
        gradients.append(grad_out[0].detach())  # Ensure gradients are detached

    def forward_hook(module, input, output):
        activations.append(output.detach())  # Ensure activations are detached

    target_layer_module = dict(model.named_modules())[target_layer]
    target_layer_module.register_forward_hook(forward_hook)
    target_layer_module.register_backward_hook(backward_hook)

    output = model(image_tensor)  # Forward pass
    target = output[:, class_idx]  # Target class score
    target.backward()  # Backward pass to compute gradients

    # Process gradients and activations
    gradients_np = gradients[0].cpu().numpy()
    activations_np = activations[0].cpu().numpy()

    weights = np.mean(gradients_np, axis=(2, 3), keepdims=True)  # Global average pooling
    grad_cam = np.sum(weights * activations_np, axis=1).squeeze()
    grad_cam = np.maximum(grad_cam, 0)  # ReLU to remove negative values
    heatmap = grad_cam / np.max(grad_cam)  # Normalize heatmap
    return heatmap


# Visualize function
def visualize_grad_cam(image_tensor, heatmap, original_image, true_label, predicted_label):
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_image, 0.5, heatmap_colored, 0.5, 0)

    plt.figure(figsize=(12, 6))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(original_image[..., ::-1])  # Convert BGR to RGB
    plt.title(f"Original Image\n(True: {true_label})")
    plt.axis("off")

    # Heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title(f"Grad-CAM Heatmap\n(Predicted: {predicted_label})")
    plt.axis("off")

    # Overlayed Image
    plt.subplot(1, 3, 3)
    plt.imshow(overlay[..., ::-1])  # Convert BGR to RGB
    plt.title(f"Overlay\n(True: {true_label}, Pred: {predicted_label})")
    plt.axis("off")

    plt.show()


def run_grad_cam(test_path, model_path):
    # Load dataset
    data_dir = test_path  # Path to your dataset
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    class_names = dataset.classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121WithCBAM(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(model_path))  # Replace with your model checkpoint

    # Select two images from each class
    images_per_class = 3
    for class_idx, class_name in enumerate(class_names):
        count = 0
        for i, (image_tensor, label) in enumerate(dataset):
            if label == class_idx and count < images_per_class:
                image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
                original_image = cv2.imread(dataset.samples[i][0])  # Load original image as NumPy array

                # Model prediction
                output = model(image_tensor)
                predicted_class_idx = torch.argmax(output, dim=1).item()
                predicted_label = class_names[predicted_class_idx]
                true_label = class_name

                # Generate Grad-CAM heatmap
                heatmap = grad_cam(model, image_tensor, predicted_class_idx, target_layer='features')

                # Visualize Grad-CAM
                visualize_grad_cam(image_tensor, heatmap, original_image, true_label, predicted_label)
                count += 1

