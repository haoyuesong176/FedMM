import os
import torch
import random
import logging
import numpy as np


def equally_spaced_list(length, num):

    """
    Parameters:
    length -- int, total length
    num -- int, number of parts

    Returns:
    ans -- list, [part0, part1, ..., part_n-1]
    """
    
    q = length // num
    r = length % num
    ans = [q for i in range(num)]
    for j in range(r):
        ans[j] += 1
    return ans


class LR_Scheduler(object):
    
    """
    Description:
    learning rate scheduler, updating the optimizer's learning rate according to the current epoch
    """
    
    def __init__(self, base_lr, num_epochs):
        self.base_lr = base_lr
        self.num_epochs = num_epochs

    def __call__(self, optimizer, epoch):
        lr = round(self.base_lr * np.power(1 - np.float32(epoch)/np.float32(self.num_epochs), 0.9), 8) 
        self._adjust_learning_rate(optimizer, lr)
        return lr

    def _adjust_learning_rate(self, optimizer, lr):
        optimizer.param_groups[0]['lr'] = lr


def logging_setup(savepath, lfile='log', to_console=False):

    """
    Description: 
    python logging setup

    Params: 
    savepath -- string, path in which the log file will be saved
    lfile -- string, name of the log file
    to_console -- bool, controlling whether to direct a copy of the log to the console
    """

    ldir = savepath
    if not os.path.exists(ldir):
        os.makedirs(ldir)
    lfile = os.path.join(ldir, lfile)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', filename=lfile)

    if to_console:
      console = logging.StreamHandler()
      console.setLevel(logging.INFO)
      console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
      logging.getLogger('').addHandler(console)


def seed_setup(seed):
    
    """
    Description:
    setting the seed for torch, numpy and random.
    """
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


class AverageMeter(object):

    """
    Description:
    a simple meter for tracking a variable's average.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

def contaim(m, mask_names):
    if m == 'flair':
        if 'flair' in mask_names or 'flairt1' in mask_names or 'flairt2' in mask_names or 'flairt1ce' in mask_names or 'flairt1cet1' in mask_names or 'flairt1t2' in mask_names or 'flairt1cet2' in mask_names or 'flairt1cet1t2' in mask_names:
            return True
        else:
            return False
    if m == 't1c':
        if 't1c' in mask_names or 't1cet2' in mask_names or 't1cet1' in mask_names or 'flairt1ce' in mask_names or 'flairt1cet1' in mask_names or 't1cet1t2' in mask_names or 'flairt1cet2' in mask_names or 'flairt1cet1t2' in mask_names:
            return True
        else:
            return False
    if m == 't1':
        if 't1' in mask_names or 'flairt1' in mask_names or 't1t2' in mask_names or 't1cet1' in mask_names or 'flairt1cet1' in mask_names or 'flairt1t2' in mask_names or 't1cet1t2' in mask_names or 'flairt1cet1t2' in mask_names:
            return True
        else:
            return False
    if m == 't2':
        if 't2' in mask_names or 't1t2' in mask_names or 'flairt2' in mask_names or 't1cet2' in mask_names or 't1cet1t2' in mask_names or 'flairt1t2' in mask_names or 'flairt1cet2' in mask_names or 'flairt1cet1t2' in mask_names:
            return True
        else:
            return False


