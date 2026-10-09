from torchvision import transforms


def build_train_transform(image_size=224):
    """Build the training preprocessing and augmentation pipeline."""
    return transforms.Compose(
        [
            # 1. Resize every image to the configured square resolution.
            transforms.Resize((image_size, image_size)),

            # 2. Apply the color-jitter ranges reported by the reference paper.
            transforms.ColorJitter(
                brightness=(0.0, 0.6),
                contrast=(0.0, 1.0),
                saturation=(0.0, 0.3),
                hue=(0.0, 0.1)
            ),

            # 3. Flip horizontally and vertically with probability 0.5.
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),

            # 4. Rotate by -90 or +90 degrees with probability 0.5.
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

            # 5. Convert RGB values from [0, 255] to [0, 1].
            transforms.ToTensor(),
        ]
    )


def build_eval_transform(image_size=224):
    """Resize validation/test images and scale RGB values to [0, 1]."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ]
    )
