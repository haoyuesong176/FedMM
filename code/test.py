import argparse
import logging
import numpy as np
import sys
import os

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

sys.path.insert(0, os.path.dirname(__file__) + '/..')
from code.models.models import Model
from code.utils.utils import logging_setup, seed_setup, AverageMeter
from code.utils.datasets import BRATS2020, Mask_Generator
from code.core.test import test_loop, csv_write


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--patch_size', default=80, type=int)
    parser.add_argument('--masks', default=None, type=str)
    parser.add_argument('--datapath', default=None, type=str)
    parser.add_argument('--savepath', default=None, type=str)
    parser.add_argument('--testfile', default=None, type=str)
    parser.add_argument('--logfile', default=None, type=str)
    parser.add_argument('--csvfile', default=None, type=str)
    parser.add_argument('--checkpoint', default=None, type=str)              
    parser.add_argument('--seed', default=1024, type=int)
    args = parser.parse_args()
    return args


def run():

    ############## Setup ###############
    args = parse()
    seed_setup(args.seed)
    logging_setup(args.savepath, lfile=args.logfile, to_console=True)

    cudnn.benchmark = True 
    cudnn.deterministic = True

    ########## Datasets ###########

    test_file = args.testfile

    test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

    test_set = BRATS2020(root=args.datapath, 
                         indexing_file=test_file,
                         train=False,
                         transforms=test_transforms)

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True)

    ########## Loading Models ###########

    num_cls = 4
    model = Model(num_cls=num_cls).cuda()    
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint)

    ########## Testing ##############

    mask_generator = Mask_Generator(args.masks.split('-'))
    masks, mask_name = mask_generator()

    test_scores = AverageMeter()
    csv_records = []

    for i, mask in enumerate(masks):

        logging.info('mask | {}'.format(mask_name[i]))

        targets_names, avg_targets_dice_scores = test_loop(test_loader, 
                                                           model, 
                                                           feature_mask=mask, 
                                                           patch_size=args.patch_size)
        
        msg = 'Average scores:' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(targets_names, avg_targets_dice_scores)])
        logging.info(msg)

        csv_records.append(np.round(avg_targets_dice_scores[[0,1,3]]*100, 2))
        
        test_scores.update(avg_targets_dice_scores)

    logging.info('Overall Average scores: {}'.format(test_scores.avg))
    csv_records.append(np.round(test_scores.avg[[0,1,3]]*100, 2))
    csv_write(args.savepath, args.csvfile, csv_records)


if __name__ == '__main__':
    run()
