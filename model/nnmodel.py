"""
Pytorch Neural Net Classes
"""
from model.nnloader import *
import torch.nn as nn

import os


# Basic model (i.e., couple of CNNs) to train a single dataset
# Each argument (e.g., dim, #y_labels, etc) is subject to a specific dataset
# Note: choose option rgb/init_padding according to:
#   1) MNIST: rgb=False, init_padding="same"
#   2) SVHN: rgb=True, init_padding="valid"
#   3) MNIST-M: rgb=True, init_padding="same"
class DigitNN(torch.nn.Module):
    def __init__(self, md_file:str, rgb:bool = False, init_padding:str = "same"):
        super(DigitNN, self).__init__()
        assert init_padding in ["same", "valid"]

        nn_path = os.path.join('model/dict', md_file)
        self.nn = load_nn(nn_path)

    def forward(self, input_data:torch.tensor) -> torch.tensor:
        """ Inference """
        y = self.nn(input_data)

        return y
    

def build_fc_classifier(dim_in, num_classes=10):
    """ 3 Fully Connected Neural Networks """
    fc = nn.Sequential()
    fc.add_module('fc1', nn.Linear(dim_in, 128))
    fc.add_module('fc2', nn.Linear(128, 64))
    fc.add_module('fc3', nn.Linear(64, num_classes))
    fc.add_module('softmax', nn.LogSoftmax(dim = 1))
        
    return fc


def build_conv_net(in_channel, init_padding="same"):
    """ 3 Convolutional Neural Networks """
    conv = nn.Sequential()
    conv.add_module('conv1', nn.Conv2d(in_channel, 32, kernel_size=5, padding=init_padding))
    conv.add_module('pool1', nn.MaxPool2d(2))
    conv.add_module('relu1', nn.ReLU(True))
    conv.add_module('conv2', nn.Conv2d(32, 32, kernel_size=3, padding="same"))
    conv.add_module('drop1', nn.Dropout2d(p=0.1))
    conv.add_module('pool2', nn.MaxPool2d(2))
    conv.add_module('relu2', nn.ReLU(True))
    conv.add_module('conv3', nn.Conv2d(32, 32, kernel_size=3, padding="same"))
    conv.add_module('pool3', nn.MaxPool2d(2))
    conv.add_module('relu3', nn.ReLU(True))
    
    return conv


class CNN_MNIST(nn.Module):
    """ CNN model for MNIST dataset """
    def __init__(self):
        super(CNN_MNIST, self).__init__()

        self.convL = build_conv_net(in_channel=1, init_padding="same")
        self.fcL = build_fc_classifier(dim_in=32 * 3 * 3, num_classes=10)

    def forward(self, input_data):
        """ Inference """
        x = self.convL(input_data)
        x = x.view(-1, 3 * 3 * 32)
        y = self.fcL(x)

        return y
    

class CNN_SVHN(nn.Module):
    """ CNN model for SVHN dataset """
    def __init__(self):
        super(CNN_SVHN, self).__init__()

        self.convL = build_conv_net(in_channel=3, init_padding="valid")
        self.fcL = build_fc_classifier(dim_in=32 * 3 * 3, num_classes=10)

    def forward(self, input_data):
        """ Inference """
        x = self.convL(input_data)
        x = x.view(-1, 3 * 3 * 32)
        y = self.fcL(x)

        return y
    

class CNN_MNISTM(nn.Module):
    """ CNN model for MNIST-M dataset """
    def __init__(self):
        super(CNN_MNISTM, self).__init__()
        
        self.convL = build_conv_net(in_channel=3, init_padding="same")
        self.fcL = build_fc_classifier(dim_in=32 * 3 * 3, num_classes=10)

    def forward(self, input_data):
        """ Inference """
        x = self.convL(input_data)
        x = x.view(-1, 3 * 3 * 32)
        y = self.fcL(x)

        return y