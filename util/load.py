from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from util import augment

def load_data():
    transform = augment.get_transforms()
    train_data = datasets.ImageFolder('dataset/chest_XRAY/train', transform=transform)
    val_data = datasets.ImageFolder('dataset/chest_XRAY/test', transform=transform)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    return train_loader, train_data, val_loader, val_data, train_data.classes