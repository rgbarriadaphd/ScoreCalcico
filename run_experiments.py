import logging
import torch
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



    return data_loader, len(image_datasets.classes)


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


def experiment_1(net):
    logging.info(f'\nRunning experiment 1\n')

    for param in net.features.parameters():
        assert param.requires_grad == True
    for param in net.classifier.parameters():
        assert param.requires_grad == True

    # Generate run test
    rt = ScoreCalciumSelection()
    folds_acc = []

    torch.save(net.state_dict(), 'tmp_model.pt')
    for i in range(5):
        rt.generate_run_set(i + 1)
        net.load_state_dict(torch.load('tmp_model.pt'))

        data_loader_test, n_classes = load_and_transform_data(os.path.join(SCORE_CALCIUM_DATA, TEST))
        folds_acc.append(evaluate_model(net, data_loader_test, device))

    # Confident interval computation
    mean, stdev, offset, ci = statistics.get_fold_metrics(folds_acc)
    logging.info(f'[Experiment 1] --> Model performance:')
    logging.info(f'     Folds Acc.: {folds_acc}')
    logging.info(f'     Mean: {mean}')
    logging.info(f'     Stdev: {stdev}')
    logging.info(f'     Offset: {offset}')
    logging.info(f'     CI:(95%) : {ci}')


def experiment_2(net):
    logging.info(f'\nRunning experiment 2\n')

    for param in net.features.parameters():
        assert param.requires_grad == True
    for param in net.classifier.parameters():
        assert param.requires_grad == True

    # Generate run test
    rt = ScoreCalciumSelection()
    folds_acc = []
    torch.save(net.state_dict(), 'tmp_model.pt')
    for i in range(5):
        rt.generate_run_set(i + 1)
        net.load_state_dict(torch.load('tmp_model.pt'))

        # Load and transform datasets
        train_data_loader, n_classes = load_and_transform_data(os.path.join(SCORE_CALCIUM_DATA, TRAIN), BATCH_SIZE,
                                                               data_augmentation=False)
        net = train_model(model=net,
                          device=device,
                          train_loader=train_data_loader,
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          lr=LEARNING_RATE)

        # test model
        data_loader_test, n_classes = load_and_transform_data(os.path.join(SCORE_CALCIUM_DATA, TEST))
        folds_acc.append(evaluate_model(net, data_loader_test, device))

    # Confident interval computation
    mean, stdev, offset, ci = statistics.get_fold_metrics(folds_acc)
    logging.info(f'[Experiment 2] --> Model performance:')
    logging.info(f'     Folds Acc.: {folds_acc}')
    logging.info(f'     Mean: {mean}')
    logging.info(f'     Stdev: {stdev}')
    logging.info(f'     Offset: {offset}')
    logging.info(f'     CI:(95%) : {ci}')


def experiment_3(net):
    logging.info(f'\nRunning experiment 3\n')

    for param in net.features.parameters():
        param.requires_grad = False

    for param in net.features.parameters():
        assert param.requires_grad == False
    for param in net.classifier.parameters():
        assert param.requires_grad == True


    # Generate run test
    rt = ScoreCalciumSelection()
    folds_acc = []
    torch.save(net.state_dict(), 'tmp_model.pt')
    for i in range(5):
        rt.generate_run_set(i + 1)
        net.load_state_dict(torch.load('tmp_model.pt'))

        # Load and transform datasets
        train_data_loader, n_classes = load_and_transform_data(os.path.join(SCORE_CALCIUM_DATA, TRAIN), BATCH_SIZE,
                                                               data_augmentation=False)
        net = train_model(model=net,
                          device=device,
                          train_loader=train_data_loader,
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          lr=LEARNING_RATE)

        # test model
        data_loader_test, n_classes = load_and_transform_data(os.path.join(SCORE_CALCIUM_DATA, TEST))
        folds_acc.append(evaluate_model(net, data_loader_test, device))

    # Confident interval computation
    mean, stdev, offset, ci = statistics.get_fold_metrics(folds_acc)
    logging.info(f'[Experiment 3] --> Model performance:')
    logging.info(f'     Folds Acc.: {folds_acc}')
    logging.info(f'     Mean: {mean}')
    logging.info(f'     Stdev: {stdev}')
    logging.info(f'     Offset: {offset}')
    logging.info(f'     CI:(95%) : {ci}')

def experiment_4(net):
    logging.info(f'\nRunning experiment 4\n')

    for param in net.features.parameters():
        param.requires_grad = False
    for param in net.classifier.parameters():
        param.requires_grad = False
    for param in net.classifier[-1].parameters():
        param.requires_grad=True

    for param in net.features.parameters():
        assert param.requires_grad == False

    for param in net.classifier[:-1].parameters():
        assert param.requires_grad == False

    for param in net.classifier[-1].parameters():
        assert param.requires_grad==True

    # Generate run test
    rt = ScoreCalciumSelection()
    folds_acc = []
    torch.save(net.state_dict(), 'tmp_model.pt')
    for i in range(5):
        rt.generate_run_set(i + 1)
        net.load_state_dict(torch.load('tmp_model.pt'))

        # Load and transform datasets
        train_data_loader, n_classes = load_and_transform_data(os.path.join(SCORE_CALCIUM_DATA, TRAIN), BATCH_SIZE,
                                                               data_augmentation=False)
        net = train_model(model=net,
                          device=device,
                          train_loader=train_data_loader,
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          lr=LEARNING_RATE)

        # test model
        data_loader_test, n_classes = load_and_transform_data(os.path.join(SCORE_CALCIUM_DATA, TEST))
        folds_acc.append(evaluate_model(net, data_loader_test, device))

    # Confident interval computation
    mean, stdev, offset, ci = statistics.get_fold_metrics(folds_acc)
    logging.info(f'[Experiment 4] --> Model performance:')
    logging.info(f'     Folds Acc.: {folds_acc}')
    logging.info(f'     Mean: {mean}')
    logging.info(f'     Stdev: {stdev}')
    logging.info(f'     Offset: {offset}')
    logging.info(f'     CI:(95%) : {ci}')

def experiment_5(net):
    logging.info(f'\nRunning experiment 5\n')

    glaucoma_model_path = os.path.join(BASE_OUTPUT, MODELS['glaucoma'])
    glaucoma_train_set, _ = load_and_transform_data(GLAUCOMA_DATA, BATCH_SIZE,data_augmentation=False)
    net = train_model(model=net,
                      device=device,
                      train_loader=glaucoma_train_set,
                      epochs=GLAUCOMA_EPOCHS,
                      batch_size=BATCH_SIZE,
                      lr=LEARNING_RATE)
    torch.save(net.state_dict(), glaucoma_model_path)

    # Generate run test
    rt = ScoreCalciumSelection()
    folds_acc = []
    torch.save(net.state_dict(), 'tmp_model.pt')
    for i in range(5):
        rt.generate_run_set(i + 1)
        net.load_state_dict(torch.load('tmp_model.pt'))

        data_loader_test, n_classes = load_and_transform_data(os.path.join(SCORE_CALCIUM_DATA, TEST))
        folds_acc.append(evaluate_model(net, data_loader_test, device))

    # Confident interval computation
    mean, stdev, offset, ci = statistics.get_fold_metrics(folds_acc)
    logging.info(f'[Experiment 5] --> Model performance:')
    logging.info(f'     Folds Acc.: {folds_acc}')
    logging.info(f'     Mean: {mean}')
    logging.info(f'     Stdev: {stdev}')
    logging.info(f'     Offset: {offset}')
    logging.info(f'     CI:(95%) : {ci}')

def experiment_6(net):
    logging.info(f'\nRunning experiment 6\n')

    glaucoma_model_path = os.path.join(BASE_OUTPUT, MODELS['glaucoma'])
    net.load_state_dict(torch.load(glaucoma_model_path))

    for param in net.features.parameters():
        param.requires_grad = False
    for param in net.classifier.parameters():
        param.requires_grad =True

    for param in net.features.parameters():
        assert param.requires_grad == False
    for param in net.classifier.parameters():
        assert param.requires_grad == True

    # Generate run test
    rt = ScoreCalciumSelection()
    folds_acc = []
    torch.save(net.state_dict(), 'tmp_model.pt')
    for i in range(5):
        rt.generate_run_set(i + 1)
        net.load_state_dict(torch.load('tmp_model.pt'))

        # Load and transform datasets
        train_data_loader, n_classes = load_and_transform_data(os.path.join(SCORE_CALCIUM_DATA, TRAIN), BATCH_SIZE,
                                                               data_augmentation=False)
        net = train_model(model=net,
                          device=device,
                          train_loader=train_data_loader,
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          lr=LEARNING_RATE)

        # test model
        data_loader_test, n_classes = load_and_transform_data(os.path.join(SCORE_CALCIUM_DATA, TEST))
        folds_acc.append(evaluate_model(net, data_loader_test, device))

    # Confident interval computation
    mean, stdev, offset, ci = statistics.get_fold_metrics(folds_acc)
    logging.info(f'[Experiment 6] --> Model performance:')
    logging.info(f'     Folds Acc.: {folds_acc}')
    logging.info(f'     Mean: {mean}')
    logging.info(f'     Stdev: {stdev}')
    logging.info(f'     Offset: {offset}')
    logging.info(f'     CI:(95%) : {ci}')

def experiment_7(net):
    logging.info(f'\nRunning experiment 7\n')

    glaucoma_model_path = os.path.join(BASE_OUTPUT, MODELS['glaucoma'])
    net.load_state_dict(torch.load(glaucoma_model_path))

    for param in net.features.parameters():
        param.requires_grad = False
    for param in net.classifier.parameters():
        param.requires_grad = False
    for param in net.classifier[-1].parameters():
        param.requires_grad=True

    for param in net.features.parameters():
        assert param.requires_grad == False

    for param in net.classifier[:-1].parameters():
        assert param.requires_grad == False

    for param in net.classifier[-1].parameters():
        assert param.requires_grad==True

    # Generate run test
    rt = ScoreCalciumSelection()
    folds_acc = []
    torch.save(net.state_dict(), 'tmp_model.pt')
    for i in range(5):
        rt.generate_run_set(i + 1)
        net.load_state_dict(torch.load('tmp_model.pt'))

        # Load and transform datasets
        train_data_loader, n_classes = load_and_transform_data(os.path.join(SCORE_CALCIUM_DATA, TRAIN), BATCH_SIZE,
                                                               data_augmentation=False)
        net = train_model(model=net,
                          device=device,
                          train_loader=train_data_loader,
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          lr=LEARNING_RATE)

        # test model
        data_loader_test, n_classes = load_and_transform_data(os.path.join(SCORE_CALCIUM_DATA, TEST))
        folds_acc.append(evaluate_model(net, data_loader_test, device))

    # Confident interval computation
    mean, stdev, offset, ci = statistics.get_fold_metrics(folds_acc)
    logging.info(f'[Experiment 7] --> Model performance:')
    logging.info(f'     Folds Acc.: {folds_acc}')
    logging.info(f'     Mean: {mean}')
    logging.info(f'     Stdev: {stdev}')
    logging.info(f'     Offset: {offset}')
    logging.info(f'     CI:(95%) : {ci}')

def experiment_8(net):
    logging.info(f'\nRunning experiment 8\n')

    for param in net.features.parameters():
        param.requires_grad = False
    for param in net.classifier.parameters():
        param.requires_grad = True

    for param in net.features.parameters():
        assert param.requires_grad == False
    for param in net.classifier.parameters():
        assert param.requires_grad == True

    glaucoma_model_path = os.path.join(BASE_OUTPUT, MODELS['glaucoma'])
    glaucoma_train_set, _ = load_and_transform_data(GLAUCOMA_DATA, BATCH_SIZE, data_augmentation=False)
    net = train_model(model=net,
                      device=device,
                      train_loader=glaucoma_train_set,
                      epochs=GLAUCOMA_EPOCHS,
                      batch_size=BATCH_SIZE,
                      lr=LEARNING_RATE)
    torch.save(net.state_dict(), glaucoma_model_path)

    # Generate run test
    rt = ScoreCalciumSelection()
    folds_acc = []
    torch.save(net.state_dict(), 'tmp_model.pt')
    for i in range(5):
        rt.generate_run_set(i + 1)
        net.load_state_dict(torch.load('tmp_model.pt'))

        data_loader_test, n_classes = load_and_transform_data(os.path.join(SCORE_CALCIUM_DATA, TEST))
        folds_acc.append(evaluate_model(net, data_loader_test, device))

    # Confident interval computation
    mean, stdev, offset, ci = statistics.get_fold_metrics(folds_acc)
    logging.info(f'[Experiment 8] --> Model performance:')
    logging.info(f'     Folds Acc.: {folds_acc}')
    logging.info(f'     Mean: {mean}')
    logging.info(f'     Stdev: {stdev}')
    logging.info(f'     Offset: {offset}')
    logging.info(f'     CI:(95%) : {ci}')

def experiment_9(net):
    logging.info(f'\nRunning experiment 9\n')

    glaucoma_model_path = os.path.join(BASE_OUTPUT, MODELS['glaucoma'])
    net.load_state_dict(torch.load(glaucoma_model_path))

    for param in net.features.parameters():
        param.requires_grad = False
    for param in net.classifier.parameters():
        param.requires_grad = True


    for param in net.features.parameters():
        assert param.requires_grad == False
    for param in net.classifier.parameters():
        assert param.requires_grad == True

    # Generate run test
    rt = ScoreCalciumSelection()
    folds_acc = []
    torch.save(net.state_dict(), 'tmp_model.pt')
    for i in range(5):
        rt.generate_run_set(i + 1)
        net.load_state_dict(torch.load('tmp_model.pt'))

        # Load and transform datasets
        train_data_loader, n_classes = load_and_transform_data(os.path.join(SCORE_CALCIUM_DATA, TRAIN), BATCH_SIZE,
                                                               data_augmentation=False)
        net = train_model(model=net,
                          device=device,
                          train_loader=train_data_loader,
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          lr=LEARNING_RATE)

        # test model
        data_loader_test, n_classes = load_and_transform_data(os.path.join(SCORE_CALCIUM_DATA, TEST))
        folds_acc.append(evaluate_model(net, data_loader_test, device))

    # Confident interval computation
    mean, stdev, offset, ci = statistics.get_fold_metrics(folds_acc)
    logging.info(f'[Experiment 9] --> Model performance:')
    logging.info(f'     Folds Acc.: {folds_acc}')
    logging.info(f'     Mean: {mean}')
    logging.info(f'     Stdev: {stdev}')
    logging.info(f'     Offset: {offset}')
    logging.info(f'     CI:(95%) : {ci}')

def experiment_10(net):
    logging.info(f'\nRunning experiment 10\n')


    glaucoma_model_path = os.path.join(BASE_OUTPUT, MODELS['glaucoma'])
    net.load_state_dict(torch.load(glaucoma_model_path))

    for param in net.features.parameters():
        param.requires_grad = False
    for param in net.classifier.parameters():
        param.requires_grad = False
    for param in net.classifier[-1].parameters():
        param.requires_grad = True

    for param in net.features.parameters():
        assert param.requires_grad == False

    for param in net.classifier[:-1].parameters():
        assert param.requires_grad == False

    for param in net.classifier[-1].parameters():
        assert param.requires_grad == True

    # Generate run test
    rt = ScoreCalciumSelection()
    folds_acc = []
    torch.save(net.state_dict(), 'tmp_model.pt')
    for i in range(5):
        rt.generate_run_set(i + 1)
        net.load_state_dict(torch.load('tmp_model.pt'))

        # Load and transform datasets
        train_data_loader, n_classes = load_and_transform_data(os.path.join(SCORE_CALCIUM_DATA, TRAIN), BATCH_SIZE,
                                                               data_augmentation=False)
        net = train_model(model=net,
                          device=device,
                          train_loader=train_data_loader,
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          lr=LEARNING_RATE)

        # test model
        data_loader_test, n_classes = load_and_transform_data(os.path.join(SCORE_CALCIUM_DATA, TEST))
        folds_acc.append(evaluate_model(net, data_loader_test, device))

    # Confident interval computation
    mean, stdev, offset, ci = statistics.get_fold_metrics(folds_acc)
    logging.info(f'[Experiment 10] --> Model performance:')
    logging.info(f'     Folds Acc.: {folds_acc}')
    logging.info(f'     Mean: {mean}')
    logging.info(f'     Stdev: {stdev}')
    logging.info(f'     Offset: {offset}')
    logging.info(f'     CI:(95%) : {ci}')

def experiment_11(net):
    logging.info(f'\nRunning experiment 11\n')

    for param in net.features.parameters():
        param.requires_grad = False
    for param in net.classifier.parameters():
        param.requires_grad = False
    for param in net.classifier[-1].parameters():
        param.requires_grad=True

    for param in net.features.parameters():
        assert param.requires_grad == False

    for param in net.classifier[:-1].parameters():
        assert param.requires_grad == False

    for param in net.classifier[-1].parameters():
        assert param.requires_grad==True

    glaucoma_model_path = os.path.join(BASE_OUTPUT, MODELS['glaucoma'])
    glaucoma_train_set, _ = load_and_transform_data(GLAUCOMA_DATA, BATCH_SIZE, data_augmentation=False)
    net = train_model(model=net,
                      device=device,
                      train_loader=glaucoma_train_set,
                      epochs=GLAUCOMA_EPOCHS,
                      batch_size=BATCH_SIZE,
                      lr=LEARNING_RATE)
    torch.save(net.state_dict(), glaucoma_model_path)

    # Generate run test
    rt = ScoreCalciumSelection()
    folds_acc = []
    torch.save(net.state_dict(), 'tmp_model.pt')
    for i in range(5):
        rt.generate_run_set(i + 1)
        net.load_state_dict(torch.load('tmp_model.pt'))

        data_loader_test, n_classes = load_and_transform_data(os.path.join(SCORE_CALCIUM_DATA, TEST))
        folds_acc.append(evaluate_model(net, data_loader_test, device))

    # Confident interval computation
    mean, stdev, offset, ci = statistics.get_fold_metrics(folds_acc)
    logging.info(f'[Experiment 11] --> Model performance:')
    logging.info(f'     Folds Acc.: {folds_acc}')
    logging.info(f'     Mean: {mean}')
    logging.info(f'     Stdev: {stdev}')
    logging.info(f'     Offset: {offset}')
    logging.info(f'     CI:(95%) : {ci}')

def experiment_12(net):
    logging.info(f'\nRunning experiment 12\n')

    glaucoma_model_path = os.path.join(BASE_OUTPUT, MODELS['glaucoma'])
    net.load_state_dict(torch.load(glaucoma_model_path))

    for param in net.features.parameters():
        param.requires_grad = False
    for param in net.classifier.parameters():
        param.requires_grad = True

    for param in net.features.parameters():
        assert param.requires_grad == False
    for param in net.classifier.parameters():
        assert param.requires_grad == True

    # Generate run test
    rt = ScoreCalciumSelection()
    logging.info("here 1")
    folds_acc = []
    torch.save(net.state_dict(), 'tmp_model.pt')
    logging.info("here 2")
    for i in range(5):
        rt.generate_run_set(i + 1)
        logging.info("here 3")
        net.load_state_dict(torch.load('tmp_model.pt'))

        # Load and transform datasets
        train_data_loader, n_classes = load_and_transform_data(os.path.join(SCORE_CALCIUM_DATA, TRAIN), BATCH_SIZE,
                                                               data_augmentation=False)
        net = train_model(model=net,
                          device=device,
                          train_loader=train_data_loader,
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          lr=LEARNING_RATE)

        # test model
        data_loader_test, n_classes = load_and_transform_data(os.path.join(SCORE_CALCIUM_DATA, TEST))
        folds_acc.append(evaluate_model(net, data_loader_test, device))

    # Confident interval computation
    mean, stdev, offset, ci = statistics.get_fold_metrics(folds_acc)
    logging.info(f'[Experiment 12] --> Model performance:')
    logging.info(f'     Folds Acc.: {folds_acc}')
    logging.info(f'     Mean: {mean}')
    logging.info(f'     Stdev: {stdev}')
    logging.info(f'     Offset: {offset}')
    logging.info(f'     CI:(95%) : {ci}')

def experiment_13(net):
    logging.info(f'\nRunning experiment 13\n')

    glaucoma_model_path = os.path.join(BASE_OUTPUT, MODELS['glaucoma'])
    net.load_state_dict(torch.load(glaucoma_model_path))

    for param in net.features.parameters():
        param.requires_grad = False
    for param in net.classifier.parameters():
        param.requires_grad = False
    for param in net.classifier[-1].parameters():
        param.requires_grad = True

    for param in net.features.parameters():
        assert param.requires_grad == False

    for param in net.classifier[:-1].parameters():
        assert param.requires_grad == False

    for param in net.classifier[-1].parameters():
        assert param.requires_grad == True

    # Generate run test
    rt = ScoreCalciumSelection()
    folds_acc = []
    torch.save(net.state_dict(), 'tmp_model.pt')
    for i in range(5):
        rt.generate_run_set(i + 1)
        net.load_state_dict(torch.load('tmp_model.pt'))

        # Load and transform datasets
        train_data_loader, n_classes = load_and_transform_data(os.path.join(SCORE_CALCIUM_DATA, TRAIN), BATCH_SIZE,
                                                               data_augmentation=False)
        net = train_model(model=net,
                          device=device,
                          train_loader=train_data_loader,
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          lr=LEARNING_RATE)

        # test model
        data_loader_test, n_classes = load_and_transform_data(os.path.join(SCORE_CALCIUM_DATA, TEST))
        folds_acc.append(evaluate_model(net, data_loader_test, device))

    # Confident interval computation
    mean, stdev, offset, ci = statistics.get_fold_metrics(folds_acc)
    logging.info(f'[Experiment 13] --> Model performance:')
    logging.info(f'     Folds Acc.: {folds_acc}')
    logging.info(f'     Mean: {mean}')
    logging.info(f'     Stdev: {stdev}')
    logging.info(f'     Offset: {offset}')
    logging.info(f'     CI:(95%) : {ci}')

def experiment_14(net):
    logging.info(f'\nRunning experiment 14\n')

    glaucoma_model_path = os.path.join(BASE_OUTPUT, MODELS['glaucoma'])
    glaucoma_train_set, _ = load_and_transform_data(GLAUCOMA_DATA, BATCH_SIZE, data_augmentation=False)
    net = train_model(model=net,
                      device=device,
                      train_loader=glaucoma_train_set,
                      epochs=GLAUCOMA_EPOCHS,
                      batch_size=BATCH_SIZE,
                      lr=LEARNING_RATE)
    torch.save(net.state_dict(), glaucoma_model_path)

    # Generate run test
    rt = ScoreCalciumSelection()
    folds_acc = []
    torch.save(net.state_dict(), 'tmp_model.pt')
    for i in range(5):
        rt.generate_run_set(i + 1)
        net.load_state_dict(torch.load('tmp_model.pt'))

        # Load and transform datasets
        train_data_loader, n_classes = load_and_transform_data(os.path.join(SCORE_CALCIUM_DATA, TRAIN), BATCH_SIZE,
                                                               data_augmentation=False)
        net = train_model(model=net,
                          device=device,
                          train_loader=train_data_loader,
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          lr=LEARNING_RATE)

        # test model
        data_loader_test, n_classes = load_and_transform_data(os.path.join(SCORE_CALCIUM_DATA, TEST))
        folds_acc.append(evaluate_model(net, data_loader_test, device))

    # Confident interval computation
    mean, stdev, offset, ci = statistics.get_fold_metrics(folds_acc)
    logging.info(f'[Experiment 14] --> Model performance:')
    logging.info(f'     Folds Acc.: {folds_acc}')
    logging.info(f'     Mean: {mean}')
    logging.info(f'     Stdev: {stdev}')
    logging.info(f'     Offset: {offset}')
    logging.info(f'     CI:(95%) : {ci}')

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

    net = modify_net_architecture(2, freeze_layers=False, architecture='vgg16')
    net.to(device=device)

    for param in net.features.parameters():
        assert param.requires_grad == True
    for param in net.classifier.parameters():
        assert param.requires_grad == True

    model_path = os.path.join(BASE_OUTPUT, MODELS['init_architecture']['vgg16'])
    if os.path.exists(model_path):
        logging.info(f'Importing model.......{model_path}')
        net.load_state_dict(torch.load(model_path))
    else:
        logging.info(f'Saving model.......{model_path}')
        torch.save(net.state_dict(), model_path)

    # experiment_1(net)
    # net.load_state_dict(torch.load(model_path))
    # experiment_2(net)
    # net.load_state_dict(torch.load(model_path))
    # experiment_3(net)
    # net.load_state_dict(torch.load(model_path))
    # experiment_4(net)
    net.load_state_dict(torch.load(model_path))
    experiment_5(net)
    net.load_state_dict(torch.load(model_path))
    experiment_6(net)
    net.load_state_dict(torch.load(model_path))
    experiment_7(net)
    net.load_state_dict(torch.load(model_path))
    # experiment_8(net)
    # net.load_state_dict(torch.load(model_path))
    # experiment_9(net)
    # net.load_state_dict(torch.load(model_path))
    # experiment_10(net)
    # net.load_state_dict(torch.load(model_path))
    # experiment_11(net)
    # net.load_state_dict(torch.load(model_path))
    # experiment_12(net)
    # net.load_state_dict(torch.load(model_path))
    # experiment_13(net)
    net.load_state_dict(torch.load(model_path))
    experiment_14(net)

