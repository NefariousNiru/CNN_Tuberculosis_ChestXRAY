from torchvision import transforms

def get_transforms_v0():
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform


def get_transforms_v1():
    transform = transforms.Compose([
        # Randomly resize and crop the image
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 1.33)),

        # Apply random horizontal flip
        transforms.RandomHorizontalFlip(p=0.5),

        # Apply random vertical flip (less common, but can be useful)
        transforms.RandomVerticalFlip(p=0.2),

        # Randomly adjust brightness, contrast, saturation, and hue
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),

        # Random rotation with a small angle to avoid distorting medical images too much
        transforms.RandomRotation(degrees=10),

        # Add random affine transformation (translation, scaling, shearing)
        transforms.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=5
        ),

        # Add random Gaussian noise for robustness against noise
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),

        # Convert to tensor and normalize using ImageNet mean and std
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return transform

def get_transforms_v2():
    transform = transforms.Compose([
        # Randomly resize and crop the image
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),

        # Apply random horizontal flip (removed vertical flip)
        transforms.RandomHorizontalFlip(p=0.5),

        # Adjust brightness and contrast but not saturation or hue
        transforms.ColorJitter(brightness=0.1, contrast=0.2),

        # Random rotation with a reduced angle range
        transforms.RandomRotation(degrees=5),

        # Apply random affine transformations
        transforms.RandomAffine(
            degrees=5,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            shear=5
        ),

        # Add random Gaussian blur with a lower probability
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),

        # Apply random erasing
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)),

        # Convert to tensor and normalize using dataset-specific mean and std (if available)
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return transform
