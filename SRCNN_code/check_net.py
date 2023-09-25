import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from models import SRCNN
from torchinfo import summary

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    srcnn = SRCNN(num_channels=3)
    model = srcnn.to(device)
    summary(model,(3,352,256))
    #print(model)