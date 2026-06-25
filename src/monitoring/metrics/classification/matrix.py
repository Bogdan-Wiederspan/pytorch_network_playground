import torch
import sklearn.metrics

def confusion_matrix(targets, predictions, labels=None, sample_weight=None, normalize=None):
    """
    Calculates a Confusion Matrix using the truth *y_true* and prediction *y_pred* of the model.
    The categories are defined using *target_map* and to weight give *sample_weight*. Styles are
    changed using *cmap*.

    Args:
        y_true (torch.tensor): tensor of true labels
        y_pred (torch.tensor): tensor of predicted labels
        target_map (dict[int]): dict of class name to index mapping
        sample_weight (torch.tensor, optional): tensor of weights on event basis. Defaults to None.
        cmap (str, optional): style of the confusion matrix. Defaults to "Blues".

    Returns:
        tuple: figure, axis, confusion matrix
    """
    # convert prediction to class indices
    y_pred = torch.argmax(predictions, dim=1)
    y_true = torch.argmax(targets, dim=1)
    cm = sklearn.metrics.confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        sample_weight=sample_weight,
        normalize=normalize,  # normalize to get probabilities
    )
    return cm


def confusion_matrix_rates(confusion_matrix, class_idx):
    """
    Extract the rates for a given *class_idx* from a given *confusion matrix*

    Args:
        confusion_matrix (torch.tensor): confusion matrix of shape (num_classes, num_classes)
        class_idx (int): _index of the class to extract rates for

    Returns:
        torch.tensor: tensor of shape (4,) with tp, tn, fp, fn for the given class
    """
    tp = confusion_matrix[class_idx, class_idx]
    fn = torch.sum(confusion_matrix[class_idx, :]) - tp
    fp = torch.sum(confusion_matrix[:, class_idx]) - tp
    tn = torch.sum(confusion_matrix) - (tp + fn + fp)
    return torch.stack((tp, tn, fp, fn))
