import torch
import numpy as np


def get_mask(labels: torch.tensor) -> torch.tensor:
    """ Returns mask from labels
    Args:
        labels: tensor containing labels
    """
    labels = labels.clone()
    zero_mask = labels[:, :, 0]

    return zero_mask


def accuracy(pred: torch.tensor, labels: torch.tensor) -> float:
    """ Returns accuracy
    Args:
        inputs: tensor with predicted values
        labels: tensor with correct values
    """

    return (sum((pred == labels)) / len(labels)).item()


def metric_q8(outputs: torch.tensor, labels: torch.tensor) -> float:
    """ Returns q8 metric
    Args:
        outputs: tensor with predicted values
        labels: tensor with correct values
    """
    mask = get_mask(labels)

    labels = torch.argmax(labels[:, :, 1:9], dim=2)[mask == 1]
    outputs = torch.argmax(outputs, dim=2)[mask == 1]

    return accuracy(outputs, labels)


def metric_q3(outputs: torch.tensor, labels: torch.tensor) -> float:
    """ Returns q3 metric
    Args:
        outputs: tensor with predicted values
        labels: tensor with correct values
    """
    mask = get_mask(labels)

    structure_mask = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2]).to(labels.device)

    labels = torch.max(labels[:, :, 1:9] * structure_mask, dim=2)[0].long()[mask == 1]
    outputs = torch.argmax(outputs, dim=2)[mask == 1]

    return accuracy(outputs, labels)