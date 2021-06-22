import logging
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch import optim
from tqdm import tqdm

INPUT = 750
BATCH_SIZE = 1

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

def out_dims(n, f, p, s):
    return int(((n + 2 * p - f) / s) + 1)

def conv(n):
    return out_dims(n, 3, 1, 1)

def pool(n):
    return out_dims(n, 2, 0, 2)

def compute_size(input_size):
    features_ops = ["(0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))",
                    "(1): ReLU(inplace=True)",
                    "(2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))",
                    "(3): ReLU(inplace=True)",
                    "(4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)",
                    "(5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))",
                    "(6): ReLU(inplace=True)",
                    "(7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))",
                    "(8): ReLU(inplace=True)",
                    "(9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)",
                    "(10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))",
                    "(11): ReLU(inplace=True)",
                    "(12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))",
                    "(13): ReLU(inplace=True)",
                    "(14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))",
                    "(15): ReLU(inplace=True)",
                    "(16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)",
                    "(17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))",
                    "(18): ReLU(inplace=True)",
                    "(19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))",
                    "(20): ReLU(inplace=True)",
                    "(21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))",
                    "(22): ReLU(inplace=True)",
                    "(23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)",
                    "(24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))",
                    "(25): ReLU(inplace=True)",
                    "(26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))",
                    "(27): ReLU(inplace=True)",
                    "(28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))",
                    "(29): ReLU(inplace=True)",
                    "(30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)"]

    n = input_size[0]
    nc = input_size[2]
    print(f'Input size: {n} x {n} x {nc} ')

    for op in features_ops:
        if 'Conv2d' in op:
            nc = op.split(',')[1].replace(' ','')
            n = conv(n)
            op_type = op[0:3]
        if 'MaxPool2d' in op:
            n = pool(n)
            op_type = op[0:3]
        if 'ReLU' in op:
            n = n
            op_type = op[0:3]
        print(f'After op {op_type} : {n} x {n} x {nc}')
    return int(n), int(n), int(nc)




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

    sz = compute_size((INPUT, INPUT, 3))

    print(f'------\nReturned size --> {sz}')
    print(f'Total number of features : {sz[0]*sz[1]*sz[2]}')

    model = models.vgg16(pretrained=True)

    for param in model.features.parameters():
        param.require_grad = False

    model.avgpool = nn.AdaptiveAvgPool2d((sz[0], sz[1]))
    model.classifier[0] = nn.Linear(sz[0]*sz[1]*sz[2], 4096)
    model.classifier[-1] = nn.Linear(4096, 2)

    if torch.cuda.is_available():
        model.cuda()

    # # num_features = model.classifier[6].in_features
    # # print(num_features)
    # # features = list(model.classifier.children())[:-1]  # Remove last layer
    # # features.extend([nn.Linear(num_features, 2)])  # Add our layer with 2 outputs
    # # model.classifier = nn.Sequential(*features)  # Replace the model classifier
    #
    #
    # print(model)

    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=4e-2)
    criterion = nn.CrossEntropyLoss()

    batch_size = BATCH_SIZE
    sample = torch.rand((batch_size, 3, INPUT, INPUT))
    ground = torch.zeros(batch_size, dtype=torch.long)
    sample = sample.to(device=device)
    ground = ground.to(device=device)

    optimizer.zero_grad()
    prediction = model(sample)
    loss = criterion(prediction, ground)
    loss.backward()
    optimizer.step()

    print(prediction.size())
    print(prediction)
