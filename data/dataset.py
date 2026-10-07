from torchvision.datasets import ImageFolder


def build_dataset(root, transform=None):
    """Đọc ảnh theo cấu trúc root/<class_name>/<image>."""
    return ImageFolder(root=root, transform=transform)
