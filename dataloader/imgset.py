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