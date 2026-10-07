from torchvision import transforms


def build_train_transform(image_size=224):
    """Tiền xử lý và augmentation ảnh train."""
    return transforms.Compose(
        [
            # Bước 1: resize tất cả ảnh về 224 x 224.
            transforms.Resize((image_size, image_size)),

            # Bước 2: thay đổi ngẫu nhiên màu sắc của ảnh.
            transforms.ColorJitter(
                brightness=(0.0, 0.6),
                contrast=(0.0, 1.0),
                saturation=(0.0, 0.3),
                hue=(0.0, 0.1)
            ),

            # Bước 3: lật ngang và lật dọc với xác suất 0.5.
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),

            # Bước 4: xoay -90 hoặc +90 độ với xác suất 0.5.
            transforms.RandomApply(
                [
                    transforms.RandomChoice(
                        [
                            transforms.RandomRotation((-90, -90)),
                            transforms.RandomRotation((90, 90))
                        ]
                    )
                ],
                p=0.5
            ),

            # Bước 5: chuyển RGB từ [0, 255] về [0, 1].
            transforms.ToTensor(),
        ]
    )


def build_eval_transform(image_size=224):
    """Validation/test chỉ resize và đưa RGB về [0, 1]."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ]
    )
