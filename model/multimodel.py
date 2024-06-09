"""
Pytorch Neural Net Classes
"""
import torch.nn as nn


class MultiEncoder(nn.Module):
    """ Multi Encoder model targeting MNIST/SVHN/MNIST-M """
    def __init__(self):
        super(MultiEncoder, self).__init__()
        
        self.idEn_MNIST = self.build_Indep_Encoder(channel_in=1, init_padding="same")
        self.idEn_SVHN = self.build_Indep_Encoder(channel_in=3, init_padding="valid")
        self.idEn_MNISTM = self.build_Indep_Encoder(channel_in=3, init_padding="same")
        
        self.fuEn = self.build_Multi_Encoder(channel_in=32, channel_out=32)
        
    def forward(self, x_a, x_b, x_c):
        """ Inference """
        x_a = self.idEn_MNIST(x_a)
        x_b = self.idEn_SVHN(x_b)
        x_c = self.idEn_MNISTM(x_c)
        
        z_a = self.fuEn(x_a)
        z_b = self.fuEn(x_b)
        z_c = self.fuEn(x_c)
        
        return z_a, z_b, z_c, x_a, x_b, x_c  
        
    def build_Indep_Encoder(self, channel_in=3, channel_out=32, init_padding="same"):
        """ Build Independent encoder for each dataset """
        idEn = nn.Sequential()
        idEn.add_module('conv1', nn.Conv2d(channel_in, 32, kernel_size=5, padding=init_padding))
        idEn.add_module('pool1', nn.MaxPool2d(2))
        idEn.add_module('relu1', nn.ReLU(True))
        idEn.add_module('conv2', nn.Conv2d(32, channel_out, kernel_size=3, padding="same"))
        idEn.add_module('drop1', nn.Dropout2d(p=0.1))
        idEn.add_module('pool2', nn.MaxPool2d(2))
        idEn.add_module('relu2', nn.ReLU(True))
        
        return idEn
        
    def build_Multi_Encoder(self, channel_in=32, channel_out=32):
        """ Build Multi Encoder """
        fuEn = nn.Sequential()
        fuEn.add_module('conv3', nn.Conv2d(channel_in, channel_out, kernel_size=3, padding="same"))
        fuEn.add_module('pool3', nn.MaxPool2d(2))
        fuEn.add_module('relu3', nn.ReLU(True))
        
        return fuEn
    

class IndepClassifier(nn.Module):
    """ Independent Classification model for each image dataset """
    def __init__(self):
        super(IndepClassifier, self).__init__()
        self.idCl_MNIST = self.build_Indep_Classifier(dim_in = 32 * 3 * 3)
        self.idCl_SVHN = self.build_Indep_Classifier(dim_in = 32 * 3 * 3)
        self.idCl_MNISTM = self.build_Indep_Classifier(dim_in = 32 * 3 * 3)
    
    def forward(self, z_a, z_b, z_c):
        """ Inference """
        z_a = z_a.view(-1, 3 * 3 * 32)
        z_b = z_b.view(-1, 3 * 3 * 32)
        z_c = z_c.view(-1, 3 * 3 * 32)
        
        y_a = self.idCl_MNIST(z_a)
        y_b = self.idCl_SVHN(z_b)
        y_c = self.idCl_MNISTM(z_c)
        
        return y_a, y_b, y_c
            
    def build_Indep_Classifier(self, dim_in):
        """ Build Independent encoder for each dataset """
        idCl = nn.Sequential()
        idCl.add_module('fc1', nn.Linear(dim_in, 128))
        idCl.add_module('fc2', nn.Linear(128, 64))
        idCl.add_module('fc3', nn.Linear(64, 10))
        idCl.add_module('softmax', nn.LogSoftmax(dim = 1))
        
        return idCl