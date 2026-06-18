import torch

class logit():
    def __init__(self):
        pass

    @staticmethod
    def forward(x, eps=1e-6):
        return torch.log(( x + eps) / (1 - x + eps))

    @staticmethod
    def inverse(y, eps=1e-6):
        num = (torch.exp(y) - eps + torch.exp(y) * eps)
        denom = (1 + torch.exp(y))
        return num / denom
