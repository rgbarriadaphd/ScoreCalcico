import math
import time
import logging
import os
import statistics
import torch
from PIL import Image, ImageOps
from torchvision import datasets, models, transforms
from data_selection import ScoreCalciumSelection
from hyperparams import *
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from utils import statistics

def load_and_transform_data(dataset, batch_size=1, data_augmentation=False):
    # Define transformations that will be applied to the images
    # VGG-16 Takes 224x224 images as input, so we resize all of them
    logging.info(f'Loading data from {dataset}')
    if data_augmentation:
        data_transforms = transforms.Compose([
            # transforms.Resize((224, 224)),
            # transforms.RandomRotation(20),
            # transforms.RandomRotation(110),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        data_transforms = transforms.Compose([
            transforms.Resize((1152, 1152)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    image_datasets = datasets.ImageFolder(dataset, transform=data_transforms)
    data_loader = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=True, num_workers=4)

    logging.info(f'Loaded {len(image_datasets)} images under {dataset}: Classes: {image_datasets.classes}')

    return data_loader


def modify_net_architecture(nclasses, pret=True, freeze_layers=True, architecture=''):
    logging.info(f'Loading model from Pytorch. Pretrained={pret}')
    if architecture == 'vgg16':
        model = models.vgg16(pretrained=pret)
    elif architecture == 'vgg19':
        model = models.vgg19(pretrained=pret)
    else:
        logging.error(f'Architecture not defined!')

    # Freeze training for all layers
    if freeze_layers:
        logging.info(f'Freeze pretrained layers')
        for param in model.features.parameters():
            param.require_grad = False

    # Newly created modules have require_grad=True by default
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1]  # Remove last layer
    features.extend([nn.Linear(num_features, nclasses)])  # Add our layer with 2 outputs
    model.classifier = nn.Sequential(*features)  # Replace the model classifier
    logging.info(f'Modifying last layer with {nclasses} classes')
    return model


def evaluate_model(model, dataloader, device):
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for data in dataloader:
            sample, ground = data
            sample = sample.to(device=device)
            ground = ground.to(device=device)

            outputs = model(sample)

            _, predicted = torch.max(outputs.data, 1)

            total += ground.size(0)
            correct += (predicted == ground).sum().item()

    return (100 * correct) / total


def train_model(model, device, train_loader, epochs=1, batch_size=4, lr=0.1, test_loader=None):
    n_train = len(train_loader.dataset)
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Device:          {device.type}
    ''')
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=4e-2)
    criterion = nn.CrossEntropyLoss()

    acc_model = 0
    for epoch in range(epochs):
        if epoch % 5 == 0 and test_loader:
            acc_model = evaluate_model(model, test_loader, device)

        model.train(True)
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                sample, ground = batch

                sample = sample.to(device=device, dtype=torch.float32)
                ground = ground.to(device=device, dtype=torch.long)
                print(sample.size())
                print(ground.size())

                optimizer.zero_grad()
                prediction = model(sample)
                loss = criterion(prediction, ground)
                loss.backward()
                optimizer.step()

                if epoch % 5 == 0 and test_loader:
                    pbar.set_postfix(**{'loss (batch) ': loss.item(), 'test ': acc_model})
                else:
                    pbar.set_postfix(**{'loss (batch) ': loss.item()})
                pbar.update(sample.shape[0])
    return model


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

    # Define model architecture
    model_architecture = 'vgg16'

    # Init model if exists in order to avoid random intit of weights on each iteration
    net = modify_net_architecture(N_CLASSES, freeze_layers=False, architecture=model_architecture)
    model_path = os.path.join(BASE_OUTPUT, 'model_glaucoma_pretrained.pt')
    net.to(device=device)

    # Generate run test
    rt = ScoreCalciumSelection()
    folds_acc = []
    for i in range(5):
        t0 = time.time()
        rt.generate_run_set(i + 1)

        logging.info(f'Generate test with fold {i + 1}')

        logging.info("Train model")
        # Load and transform datasets
        train_data_loader = load_and_transform_data(os.path.join(SCORE_CALCIUM_DATA_ORG, TRAIN), BATCH_SIZE,
                                                               data_augmentation=False)
        net = train_model(model=net,
                          device=device,
                          train_loader=train_data_loader,
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          lr=LEARNING_RATE)

        # # Save model
        # model_trained_path = os.path.join(BASE_OUTPUT, 'model_trained.pt')
        # torch.save(net.state_dict(), model_trained_path)

        data_loader_test = load_and_transform_data(os.path.join(SCORE_CALCIUM_DATA_ORG, TEST))

        # test model
        logging.info("Test model after training")
        acc_model_test_after = evaluate_model(net, data_loader_test, device)

        logging.info(
            f'Accuracy after training {acc_model_test_after}. | [{time.time() - t0}]')
        folds_acc.append(acc_model_test_after)

    # Confident interval computation
    mean, stdev, offset, ci = statistics.get_fold_metrics(folds_acc)
    logging.info(f'Model performance:')
    logging.info(f'     Folds Acc.: {folds_acc}')
    logging.info(f'     Mean: {mean}')
    logging.info(f'     Stdev: {stdev}')
    logging.info(f'     Offset: {offset}')
    logging.info(f'     CI:(95%) : {ci}')
