import statistics
import math


def get_fold_metrics(folds, confidence=1.96):
    '''
    Get statistical metrics from each fold accuracy over test set

    :param confidence: Confidence interval percentage
    :param folds: list containing the accuracy of each fold
    :return: mean and std deviation, offset of the CI and CI at 95% confidence
    '''
    mean = statistics.mean(folds)
    stdev = statistics.stdev(folds)
    offset = confidence * stdev / math.sqrt(len(folds))
    return mean, stdev, offset, (mean - offset, mean + offset)
