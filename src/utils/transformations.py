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

class tangent():
    """
    Tangent naturally decompresses around edges. Shift by 0.5 result in symmetric for 0 to 1
    """
    def __init__(self):
        pass

    @staticmethod
    def forward(x, shift=0.5):
        return torch.tan(torch.pi * ( x - shift))

    @staticmethod
    def inverse(y, shift=0.5):
        return torch.arctan(torch.pi * (y - shift))

# TODO Not completed
class cubic():
    def __init__(self, shift=0.5, min=None, max=None, stretching_factor=None):
        self.shift = shift
        if stretching_factor is None:
            self.a = self.calculate_a(min, max)
        else:
            self.a = stretching_factor

    def calculate_a(self, min, max):
        return 4 * ( max - min ) - 4

    def forward(self, x):
        return self.a * ( x - self.shift)**3 + (x - self.shift) + self.shift
