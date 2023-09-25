import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from models import SRResNet
from torchinfo import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
srresnet = SRResNet(large_kernel_size=9,
                    small_kernel_size=3,
                    n_channels=64,
                    n_blocks=16,
                    scaling_factor=2)
model1 = srresnet.to(device)
model2 = models.resnet18().to(device)
summary(model1,(1,3,352,256))
#print(model1)