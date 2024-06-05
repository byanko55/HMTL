"""
Image Loading modules
"""
import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, Subset, random_split
from torchvision import datasets, transforms


class ImgLoader():
    """
    # In
      - cuda: determine if our system supports CUDA (GPUs for computation)
    """
    def __init__(self, cuda:bool = True) -> None:
        self.rds = TensorDataset(self.x, self.y) # original(raw) data samples
        self.cuda = cuda

    """
    Build a training/eval/test batch

    # In
      - batch_size: 
      - train_ratio:
      - val_ratio: if val_ratio == 0, then it does not generate an eval batch
    # Out
      - train_batch, val_batch, test_batch: tuple of DataLoaders
    """
    def buildbatch(self, batch_size:int = 128, train_ratio:float = 0.7, val_ratio:float = 0.1) -> tuple:
        num_samples = len(self.rds)
        train_samples = int(train_ratio * num_samples)
        val_samples = int(val_ratio * num_samples)

        if val_samples > 0 :
            train_ds, val_ds, test_ds = random_split(self.rds, 
                [train_samples, val_samples, num_samples-train_samples-val_samples],
                generator=torch.Generator().manual_seed(42)
            )

            train_batch = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_batch = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
            test_batch = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

            return train_batch, val_batch, test_batch
        else :
            train_ds, test_ds = random_split(self.rds, 
                [train_samples, num_samples-train_samples],
                generator=torch.Generator().manual_seed(42)
            )

            train_batch = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            test_batch = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

            return train_batch, None, test_batch

    """
    Reconfigure dimension of image samples (this method is only utilized by 'showsamples')

    # In
      - x: input image data
    # Out
      - reshaped image data
    """
    def recoverimg(self, x:torch.tensor) -> torch.tensor:
        return x.permute(1, 2, 0).numpy()

    """
    Plot nrows x ncols images

    # In
      - nrows/ncols: determines how many image rows/columns to be represented
    """
    def showsamples(self, nrows:int = 2, ncols:int = 4) -> None:
        fig, axes = plt.subplots(nrows, ncols)

        for i, ax in enumerate(axes.flat):
            img = self.recoverimg(self.x[i,:])
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(self.y[i].numpy())

        plt.show()


# Sampling image data
def ImgSampling(data_set, sampling_rate):
    num_samples = int(sampling_rate * len(data_set))
    indices = torch.tensor(random.sample(range(len(data_set)), num_samples))
    return Subset(data_set, indices)


# MNIST dataset
# A large collection of monochrome images of handwritten digits
# It has a training set of 55,000 examples, and a test set of 10,000 examples
class MNISTset(ImgLoader):
    def __init__(self, data_path="datasets", transform=None, sampling_rate=1.0, **kwargs):
        if transform == None :
            transform = transforms.ToTensor()
            
        data_path = os.path.join(data_path, 'MNIST_data/')
            
        self.dset = datasets.MNIST(root=data_path,
                                   train=True,
                                   transform=transform,
                                   download=True)
        
        if sampling_rate < 1.0 :
            self.dset = ImgSampling(self.dset, sampling_rate)

        self.build()
        self.num_images = self.x.shape[0]
        
        print("MNIST : ", self.x.shape)
        
        # Calculate the total number of images
        print("MNIST : Total Number of Images", self.num_images)

        super().__init__()
        
    def build(self):
        """ Configure MNIST image data """
        indices = self.dset.indices
        self.x = self.dset.dataset.data[indices]/255
        self.x = self.x.type(torch.FloatTensor)
        self.y = self.dset.dataset.targets[indices]
        self.y = self.y.type(torch.LongTensor) 

        self.x.unsqueeze_(1)
        
    def dim(self):
        """ dimension of MNIST image sample """
        return 1, 28, 28
        
    def __len__(self):
        return self.num_images
        
    def __getitem__(self, item):
        imgs, labels = self.x[item], self.y[item]
        
        return imgs, labels


# SVHN dataset
# A digit classification benchmark dataset that contains the street view house number (SVHN) images
# This dataset includes 99,289 32×32 RGB images of printed digits cropped from pictures of house number plates.
class SVHNset(Dataset):
    def __init__(self, data_path="datasets", transform=None, sampling_rate=1.0):
        if transform == None :
            transform = transforms.ToTensor()

        data_path = os.path.join(data_path, 'SVHN_data/')
            
        self.dset = datasets.SVHN(root=data_path,
                                  transform=transform,
                                  download=True)
        
        self.build(sampling_rate)
            
        self.num_images = self.x.shape[0]
        
        print("SVHN : ", self.x.shape)
        
        # Calculate the total number of images
        print("SVHN : Total Number of Images", self.x.shape[0])
   
    def build(self, sampling_rate):
        """ Configure SVHN image data """
        if sampling_rate < 1.0 :
            self.dset = ImgSampling(self.dset, sampling_rate)
        
            indices = self.dset.indices

            self.x = torch.tensor(self.dset.dataset.data[indices]/255)
            self.y = torch.tensor(self.dset.dataset.labels[indices])
        else :
            self.x = torch.tensor(self.dset.data/255)
            self.y = torch.tensor(self.dset.labels)

        self.x = self.x.type(torch.FloatTensor)
        self.y = self.y.type(torch.LongTensor) 
        
    def dim(self):
        """ dimension of SVHN image sample """
        return 3, 32, 32
        
    def __len__(self):
        return self.num_images
        
    def __getitem__(self, item):
        imgs, labels = self.x[item], self.y[item]
        
        return imgs, labels


# MNIST-M dataset
# This dataset is created by combining MNIST digits with the patches 
# randomly extracted from color photos of BSDS500 as their background
# It contains 55,000 training and 10,000 test images as well
class MNISTMset(Dataset):
    def __init__(self, data_path="datasets", transform=None, sampling_rate=1.0):
        mnist_path = os.path.join(data_path, 'MNIST_data/')
            
        # Labels are included in MNIST dataset
        mnist = datasets.MNIST(root=mnist_path,
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
            
        data_path = os.path.join(data_path, 'MNISTM_data/mnistm.pkl')    
        mnistm = pickle.load(open(data_path, 'rb'), encoding='latin1')
        
        # Reshape Images: [-1, 28, 28, 3] -> [-1, 3, 28, 28]
        data = np.transpose(mnistm['train'], (0, 3, 2, 1))/255
        data = np.flip(data, axis=(2))
        data = np.rot90(data, -1, (2,3))
        
        if sampling_rate < 1.0 :
            num_samples = int(sampling_rate * len(data))
            indices = torch.tensor(random.sample(range(len(data)), num_samples))
            
            self.dset = data[indices]
        
            self.x = torch.tensor(data[indices])
            self.y = mnist.targets[indices].clone().detach()
        else :
            self.x = torch.tensor(data)
            self.y = torch.tensor(mnist.targets)

        self.x = self.x.type(torch.FloatTensor)
        self.y = self.y.type(torch.LongTensor) 
        
        self.num_images = self.x.shape[0]
        
        print("MNIST-M : ", self.x.shape)
        
        # Calculate the total number of images
        print("MNIST-M : Total Number of Images", self.x.shape[0])
        
    def dim(self):
        """ dimension of MNIST-M image sample """
        return 3, 28, 28
        
    def __len__(self):
        return self.num_images
        
    def __getitem__(self, item):
        imgs, labels = self.x[item], self.y[item]
        
        return imgs, labels