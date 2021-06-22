import torch
from utils.common import *
from utils.log import get_logger
from torchvision import models
from data_selection import ScoreCalciumSelection
import torch.nn as nn
from utils import statistics, dataset, train, test
import time
import matplotlib.pyplot as plt
from datetime import datetime

######################################
# LOCAL HYPER-PARAMETERS             #
BATCH_SIZE = 4
EPOCHS = 3
MONOFOLD = True
LEARNING_RATE = 0.001
N_CLASSES = 2
SAVE_LOSS = True
######################################

# Login instance
log = get_logger()


def get_model():
    '''

    :return:
    '''
    base_model = 'tmp/base_model.pt'
    net = models.vgg16(pretrained=True)

    # Freeze trained weights
    for param in net.features.parameters():
        param.requires_grad = False

    # Newly created modules have require_grad=True by default
    num_features = net.classifier[6].in_features
    features = list(net.classifier.children())[:-1]  # Remove last layer
    features.extend([nn.Linear(num_features, N_CLASSES)])  # Add our layer with 2 outputs
    net.classifier = nn.Sequential(*features)  # Replace the model classifier

    if os.path.exists(base_model):
        net.load_state_dict(torch.load(base_model))
        log.info(f'loading {base_model}')
    else:
        torch.save(net.state_dict(), 'tmp/base_model.pt')
        log.info(f'save {base_model}')

    net.to(device=device)

    return net


if __name__ == '__main__':
    """
    Set of experiments that combine image data and clincal data:
    Sequence:
        0.- Init models to be common to each experiment
        1.- VGG16 pretrained imagenet
        2.- Ensembles classifiers of clinical data
        3.- Add extra column with deep learning prediction
        4.- Remove column DR from clinical data 
    """
    # Import device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Using device: {device}')

    # Import model
    model = get_model()

    # Generate run test
    rt = ScoreCalciumSelection()
    folds_acc = []
    for i in range(5):
        t0 = time.time()
        rt.generate_run_set(i + 1)

        log.info(f'Generate test with fold {i + 1}')

        log.info("Train model")

        # Load and transform datasets
        test_data_loader = dataset.load_and_transform_data(os.path.join(SCORE_CALCIUM_DATA, TEST), log=log)
        train_data_loader = dataset.load_and_transform_data(os.path.join(SCORE_CALCIUM_DATA, TRAIN), log=log,
                                                            batch_size=BATCH_SIZE,
                                                            data_augmentation=False)

        # train model
        model, test_acc, lrs, epoch_list, losses = train.train_model(model=model,
                                                                     device=device,
                                                                     train_loader=train_data_loader,
                                                                     log=log,
                                                                     epochs=EPOCHS,
                                                                     batch_size=BATCH_SIZE,
                                                                     lr=LEARNING_RATE,
                                                                     test_loader=None)

        # test model
        log.info("Test model after training")
        acc_model_test_after = test.evaluate_model(model, test_data_loader, device, log=log)

        log.info(
            f'Accuracy after training {acc_model_test_after}. | [{time.time() - t0}]')
        folds_acc.append(acc_model_test_after)

        if SAVE_LOSS:
            plt.plot(list(range(1, len(losses) + 1)), losses)
            plt.ylabel("loss")
            plt.xlabel("epochs")
            plt.title(f'loos evolution at fold {i + 1}')
            plot_name = datetime.now().strftime("%d-%m-%Y_%H_%M_%S") + "_loss_fold_" + str(i + 1) + ".png"
            log.info(plot_name)
            plt.savefig('plots/' + plot_name)

        if MONOFOLD:
            break

        log.info(f'******************************* FOLD: ' + str(i))
        log.info("Partial tests" + str(test_acc))
        log.info("LRs" + str(lrs))
        log.info("Epochs" + str(epoch_list))

    # Confident interval computation
    mean, stdev, offset, ci = statistics.get_fold_metrics(folds_acc)
    log.info(f'******************************************')
    log.info(f'Model performance:')
    log.info(f'     Folds Acc.: {folds_acc}')
    log.info(f'     Mean: {mean}')
    log.info(f'     Stdev: {stdev}')
    log.info(f'     Offset: {offset}')
    log.info(f'     CI:(95%) : {ci}')
