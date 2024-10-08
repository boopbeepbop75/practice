import torchvision.transforms as transforms
from torchvision import datasets

# Transform to convert images to Tensor without resizing (keeping original 28x28 size)
transform = transforms.Compose([
  transforms.ToTensor()  # No resizing needed for 28x28 FashionMNIST images
])

def get_training_data():
    # Downloading the dataset
    train_data = datasets.FashionMNIST(
        root = "data",
        train = True,
        download = True,
        transform = transform
    )
    return train_data

def get_testing_data():
    test_data = datasets.FashionMNIST(
        root = "data",
        train = False,
        download = True,
        transform = transform
    )
    return test_data