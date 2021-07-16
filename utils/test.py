import torch
import os
import numpy as np


def evaluate_model(model, dataloader, device, log, epoch=None, fold=None):
    correct = 0
    total = 0
    prediction = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            sample, ground, file_info = data
            sample = sample.to(device=device)
            ground = ground.to(device=device)
            outputs = model(sample)
            _, predicted = torch.max(outputs.data, 1)

            sample_name = file_info[0][0]
            class_name = sample_name.split('/')[3]
            sample_name = os.path.splitext(sample_name)[0].split('/')[-1]
            prediction.append((sample_name, predicted.item(), np.array(outputs.cpu())[0][0],
                               np.array(outputs.cpu())[0][1], np.array(ground.cpu())[0], fold, class_name))
            total += ground.size(0)
            correct += (predicted == ground).sum().item()
    return (100 * correct) / total, prediction
