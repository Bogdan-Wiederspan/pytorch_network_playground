### Metrics and tools for multi-class classification
def map_prediction_to_binary(prediction):
    """
    Helper to map prediction tensor to a binary one-hot encoded tensor based on arg max

    Args:
        prediction (torch.tensor): tensor of shape (num_events, num_classes) with class scores/probabilities

    Returns:
        torch.tensor: binary one-hot encoded tensor of same shape as *prediction*
    """

    _prediction = torch.zeros_like(prediction)
    _predicted_class = torch.argmax(prediction, dim=1)
    # index enables to set values at specific indices by using a source
    _prediction = _prediction.scatter_(dim=1, index=_predicted_class.unsqueeze(1), value=1)
    return _prediction
