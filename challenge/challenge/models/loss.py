import torch
import torch.nn as nn

from challenge.models.metric import get_mask


def cross_entropy(outputs: torch.tensor, labels: torch.tensor, mask: torch.tensor) -> torch.tensor:
    """ Returns cross entropy loss using masking
    Args:
        outputs: tensor with predictions
        labels: tensor with labels
        mask: tensor with masking
    """
    labels = labels.clone()
    labels[mask == 0] = -1

    return nn.CrossEntropyLoss(ignore_index=-1)(outputs, labels.long())


def q8(outputs: torch.tensor, labels: torch.tensor) -> torch.tensor:
    """ Returns q8 loss
    Args:
        outputs: tensor with q8 predictions
        labels: tensor with labels
    """
    mask = get_mask(labels)

    labels = torch.argmax(labels[:, :, 1:9], dim=2)
    outputs = outputs.permute(0, 2, 1)

    return cross_entropy(outputs, labels, mask)


def q3(outputs: torch.tensor, labels: torch.tensor) -> torch.tensor:
    """ Returns q3 loss
    Args:
        outputs: tensor with q3 predictions
        labels: tensor with labels
    """
    mask = get_mask(labels)

    # convert q8 to q3 class
    structure_mask = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2]).to(labels.device)
    labels = torch.max(labels[:, :, 1:9] * structure_mask, dim=2)[0].long()
    outputs = outputs.permute(0, 2, 1)

    return cross_entropy(outputs, labels, mask)


def secondary_structure_loss(outputs: torch.tensor, labels: torch.tensor) -> torch.tensor:
    """ Returns a weighted double task loss for secondary structure. 
    Args:
        outputs: tensor with psi predictions
        labels: tensor with labels
    """
    # weighted losses
    _q8 = q8(outputs[0], labels) * 1
    _q3 = q3(outputs[1], labels) * 5

    loss = torch.stack([_q8, _q3])

    return loss.sum()
