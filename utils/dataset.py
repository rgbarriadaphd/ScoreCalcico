import torch
from torchvision import datasets, transforms

class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, dataset, transform=None):
        super(CustomImageFolder, self).__init__(dataset, transform=transform)

    def __getitem__(self, index):
        sample, label = super(datasets.ImageFolder, self).__getitem__(index)
        return sample, label, self.imgs[index]

def load_and_transform_data(dataset, log, batch_size=1, data_augmentation=False):
    # Define transformations that will be applied to the images
    # VGG-16 Takes 224x224 images as input, so we resize all of them
    log.info(f'Loading data from {dataset}')
    if data_augmentation:
        data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(20),
            transforms.RandomRotation(110),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    image_datasets = CustomImageFolder(dataset, transform=data_transforms)
    data_loader = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=False, num_workers=4)

    log.info(f'Loaded {len(image_datasets)} images under {dataset}: Classes: {image_datasets.classes}')

    return data_loader