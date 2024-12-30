import os
import sys
import argparse
import logging
import numpy as np

import torch
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 

sys.path.insert(0, os.path.dirname(__file__) + '/..')
from code.models.models import Model
from code.utils.utils import seed_setup, logging_setup, LR_Scheduler, AverageMeter
from code.utils.datasets import BRATS2020, Mask_Generator, MultiEpochsIterator, init_fn
from code.core.train import train_loop, tensorboard_record, model_save


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--patch_size', default=80, type=int)
    parser.add_argument('--masks', default=None, type=str)
    parser.add_argument('--datapath', default=None, type=str)
    parser.add_argument('--trainfile', default=None, type=str)
    parser.add_argument('--savepath', default=None, type=str)
    parser.add_argument('--logfile', default=None, type=str)
    parser.add_argument('--tensorboard_dirname', default=None, type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--iter_per_epoch', default=150, type=int)
    parser.add_argument('--seed', default=1024, type=int)
    args = parser.parse_args()
    return args
    

def run():

    ############## Setup ###############

    args = parse()
    seed_setup(args.seed)
    logging_setup(args.savepath, lfile=args.logfile, to_console=True)
    ckpts = args.savepath
    writer = SummaryWriter(os.path.join(args.savepath, args.tensorboard_dirname))

    cudnn.benchmark = False 
    cudnn.deterministic = True

    ########## Datasets ###########
    
    train_file = args.trainfile
    
    train_transforms = 'Compose([RandCrop3D(({},{},{})), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'.format(args.patch_size, args.patch_size, args.patch_size)

    mask_generator = Mask_Generator(args.masks.split('-'))
    
    train_set = BRATS2020(root=args.datapath, 
                          indexing_file=train_file,
                          train=True,
                          transforms=train_transforms, 
                          mask_generator=mask_generator)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=3,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn)

    ########## Setting Models ###########
    
    num_cls = 4
    model = Model(num_cls=num_cls).cuda()

    ########## Scheduler & Optimizer ##########

    lr_scheduler = LR_Scheduler(args.lr, args.num_epochs)
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
    optimizer = torch.optim.Adam(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)

    ########## Training #########

    train_iter = MultiEpochsIterator(train_loader)

    min_loss = 1000000

    for epoch in range(args.num_epochs):

        logging.info("Epoch {}/{}".format(epoch+1, args.num_epochs))

        step_lr = lr_scheduler(optimizer, epoch)

        train_avg_loss, train_avg_loss_list = train_loop(train_iter, model, optimizer, args.iter_per_epoch)

        tensorboard_record(train_avg_loss.avg, train_avg_loss_list.avg.tolist(), writer, epoch+1, lr=step_lr, prefix="epoch_")

        ### Logging ###
        msg = 'Avg Training Loss: Epoch {}/{}, Loss {:.4f}, '.format((epoch+1), args.num_epochs, train_avg_loss.avg)
        msg += 'fusecross:{:.4f}, fusedice:{:.4f},'.format(train_avg_loss_list.avg[0], train_avg_loss_list.avg[1])
        logging.info(msg)

        ### Saving model ###
        min_loss = model_save(model, ckpts, epoch, train_avg_loss.avg, min_loss)


if __name__ == '__main__':
    run()
