import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

from code.utils.transforms import *


class Mask_Generator(object):

    def __init__(self, mask_names):
        self.mask_dict = {
          't2':[False, False, False, True], 
          't1c':[False, True, False, False], 
          't1':[False, False, True, False], 
          'flair':[True, False, False, False],
          't1cet2':[False, True, False, True], 
          't1cet1':[False, True, True, False], 
          'flairt1':[True, False, True, False], 
          't1t2':[False, False, True, True], 
          'flairt2':[True, False, False, True], 
          'flairt1ce':[True, True, False, False],
          'flairt1cet1':[True, True, True, False], 
          'flairt1t2':[True, False, True, True], 
          'flairt1cet2':[True, True, False, True], 
          't1cet1t2':[False, True, True, True],
          'flairt1cet1t2':[True, True, True, True]}
        self.mask_names = mask_names

    def __call__(self):
        masks = []
        for mask_name in self.mask_names:
            masks.append(self.mask_dict[mask_name])
        return masks, self.mask_names


class BRATS2020(Dataset):

    """
    Parameters:
    root -- root directory of BRATS2020
    indexing_file -- file containing names list of train/test set.
    train -- True, train dataset; False, test dataset.
    transforms -- string, containing transforms info
    mask_generator -- mask generator function, only required when "train == True"

    Returns(Train == True):
    x -- tensor, images of 4 modalities, of shape (4, patch, patch, patch)
    y -- tensor, one-hot ground truth labels of 4 objects, of shape (4, patch, patch, patch)
    mask -- tensor, modality mask of shape (4)
    name -- string, the sample's name

    Returns(Train == False):
    x -- tensor, images of 4 modalities, of shape (4, H, W, Z)
    y -- tensor, ground truth labels of 4 objects, of shape (H, W, Z)
    name -- string, the sample's name
    """

    def __init__(self, 
                 root=None, 
                 indexing_file='train.txt', 
                 train=True, 
                 transforms='', 
                 mask_generator=None,
                 alpha=None):

        data_file_path = os.path.join(root, indexing_file) 
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()]
        datalist.sort()

        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))
        
        self.names = datalist      
        self.volpaths = volpaths
        self.train = train
        self.transforms = eval(transforms or 'Identity()')
        
        if self.train:
            masks, _ = mask_generator()
            self.mask_array = np.array(masks)
            self.num_cls = 4
            self.alpha = alpha

    def __getitem__(self, index):

        name = self.names[index]
        volpath = self.volpaths[index]
        segpath = volpath.replace('vol', 'seg')
        
        x = np.load(volpath) 
        y = np.load(segpath).astype(np.uint8) 
        x, y = x[None, ...], y[None, ...] 
        x,y = self.transforms([x, y])

        if not self.train:
            x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))
            y = np.ascontiguousarray(y)
            x = torch.squeeze(torch.from_numpy(x), dim=0)
            y = torch.squeeze(torch.from_numpy(y), dim=0)
            return x, y, name

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))
        _, H, W, Z = np.shape(y)
        y = np.reshape(y, (-1))
        one_hot_targets = np.eye(self.num_cls)[y]
        y = np.reshape(one_hot_targets, (1, H, W, Z, -1))
        y = np.ascontiguousarray(y.transpose(0, 4, 1, 2, 3))
        x = torch.squeeze(torch.from_numpy(x), dim=0)
        y = torch.squeeze(torch.from_numpy(y), dim=0)

        # Generate mask with MD^M
        if self.alpha != None:
            probabilities = [(1-self.alpha)/(len(self.mask_array)-1) for i in range(len(self.mask_array)-1)] + [self.alpha]
            # print(probabilities)
            mask = random.choices(self.mask_array, probabilities, k=1)[0]
        else:
            num_masks = len(self.mask_array)
            mask_idx = int(np.random.choice(num_masks, 1)) 
            mask = torch.from_numpy(self.mask_array[mask_idx])

        return x, y, mask, name

    def __len__(self):
        return len(self.volpaths)

    def set_mc(self, mc):

        """
        Parameters:
        mc - name of the selected modality
        
        Notes:
            After set_mc(), data_iterator should be re-instanciated.
        """

        mask_generator = Mask_Generator([mc])
        masks, _ = mask_generator()
        self.mask_array = np.array(masks)


class MultiEpochsIterator(object):

    """
    Description: 
    Multi-Epoch Iterator can iterate the given dataloader any desired times

    Parameters:
    dataloader -- the given dataloader needed to be iterated
    """

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __next__(self):
        try:
            data = next(self.iterator)
        except:
            self.iterator = iter(self.dataloader)
            data = next(self.iterator)
        return data

    def update(self):
        self.iterator = iter(self.dataloader)


def init_fn(worker):

    """
    worker_init_fn (callable, optional)
    If not None, this will be called on each worker subprocess with 
    the worker id (an int in [0, num_workers - 1]) as input, 
    after seeding and before data loading. (default: None)
    """

    M = 2**32 - 1
    seed = torch.LongTensor(1).random_().item()
    seed = (seed + worker) % M
    np.random.seed(seed)
    random.seed(seed)

