"""
Pytorch Neural Net Classes
"""
from model.nnloader import *

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