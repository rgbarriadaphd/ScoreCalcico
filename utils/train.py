from utils.test import evaluate_model
import torch
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import os

def train_model(model, device, train_loader, log, epochs=1, epoch_split=None, batch_size=4, lr=0.1, test_loader=None):
    n_train = len(train_loader.dataset)
    log.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Device:          {device.type}
    ''')
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=4e-2)
    criterion = nn.CrossEntropyLoss()

    test_acc = []
    lrs = []
    epoch_list=[]
    acc_model = 0
    losses=[]
    for epoch in range(epochs):
        if test_loader and epoch % epoch_split == 0:
            acc_model = evaluate_model(model, test_loader, device, epoch)
            test_acc.append(acc_model)
            lrs.append(optimizer.param_groups[0]['lr'])
            epoch_list.append(epoch)

        model.train(True)
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for i,batch in enumerate(train_loader):
                sample, ground, _ = batch
                sample = sample.to(device=device, dtype=torch.float32)
                ground = ground.to(device=device, dtype=torch.long)

                optimizer.zero_grad()
                prediction = model(sample)
                loss = criterion(prediction, ground)
                loss.backward()
                optimizer.step()

                if test_loader and epoch % epoch_split == 0:
                    pbar.set_postfix(**{'LR': optimizer.param_groups[0]['lr'], 'loss (batch) ': loss.item(), 'test ': acc_model})

                else:
                    pbar.set_postfix(**{'loss (batch) ': loss.item()})
                pbar.update(sample.shape[0])
        losses.append(loss.item())
    return model, test_acc, lrs, epoch_list, losses
