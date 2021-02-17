import logging
import os

import torch
from PIL import Image, ImageOps
from torchvision import datasets, models, transforms
from data_selection import ScoreCalciumSelection
from hyperparams import *
import torch.nn as nn
from torch import optim
from tqdm import tqdm


def load_and_transform_data(dataset, batch_size=1, data_augmentation=False):
    # Define transformations that will be applied to the images
    # VGG-16 Takes 224x224 images as input, so we resize all of them
    logging.info(f'Loading data from {dataset}')
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

    image_datasets = datasets.ImageFolder(dataset, transform=data_transforms)
    data_loader = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=True, num_workers=4)

    logging.info(f'Loaded {len(image_datasets)} images under {dataset}: Classes: {image_datasets.classes}')

    return data_loader, len(image_datasets.classes)


def modify_net_architecture(nclasess, pret=True, freeze_layers=True):
    model = models.vgg16(pretrained=pret)

    # Freeze training for all layers
    if freeze_layers:
        for param in model.features.parameters():
            param.require_grad = False

    # Newly created modules have require_grad=True by default
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1]  # Remove last layer
    features.extend([nn.Linear(num_features, nclasess)])  # Add our layer with 2 outputs
    model.classifier = nn.Sequential(*features)  # Replace the model classifier

    return model


def evaluate_model(net, dataloader, device):
    correct = 0
    total = 0
    # with torch.no_grad():
    #     for data in dataloader:
    #         sample, ground = data
    #         outputs = net(sample)
    #         print(torch.max(outputs, 1)[1])
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += ground.size(0)
    #         correct += (predicted == ground).sum().item()
    #
    # logging.info('Accuracy (1): %d %%' % (100 * correct / total))

    acc_test = 0
    for data in dataloader:
        net.train(False)
        net.eval()
        sample, ground = data

        sample = sample.to(device=device)
        ground = ground.to(device=device)

        outputs = net(sample)
        logging.info(outputs)

        _, predicted = torch.max(outputs.data, 1)
        acc_test += torch.sum(predicted == ground.data)

    logging.info('Accuracy (2): %d %%' % (acc_test.item() * 100 / len(dataloader)))
    return acc_test.item() * 100 / len(dataloader)


if __name__ == '__main__':
    """
    VGG-16 pretrained simple example.
    Trying to understand both data augmentation and output values (Model robustness)
    """
    # Login instance
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Import device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # test model
    logging.info("Test model before training")
    data_loader_test, n_classes = load_and_transform_data(os.path.join(SCORE_CALCIUM_DATA, TEST))

    # Get model with modifications
    net = modify_net_architecture(n_classes)
    net.to(device=device)

    acc_model_test = evaluate_model(net, data_loader_test, device)
