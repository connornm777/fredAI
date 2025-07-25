import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor



x = torch.randn(32, 2, device='gpu', requires_grad=True)


