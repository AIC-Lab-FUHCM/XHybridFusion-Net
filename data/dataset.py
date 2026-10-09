from torchvision.datasets import ImageFolder


def build_dataset(root, transform=None):
    """Load images stored as root/<class_name>/<image>."""
    return ImageFolder(root=root, transform=transform)
