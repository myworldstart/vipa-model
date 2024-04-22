import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import models
model = models.load_model(model_name="ResNet18", num_classes=1000)


def get_model():
    return model