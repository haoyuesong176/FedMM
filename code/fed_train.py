import os
import sys
import argparse
import logging
import numpy as np
import copy

import torch
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 

sys.path.insert(0, os.path.dirname(__file__) + '/..')
from code.models.models import Model
from code.utils.utils import seed_setup, logging_setup, LR_Scheduler, AverageMeter, contaim
from code.utils.datasets import BRATS2020, Mask_Generator, MultiEpochsIterator, init_fn
from code.core.train import train_loop, tensorboard_record, model_save


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--patch_size', default=80, type=int)
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--masks', default=None, type=str)
    parser.add_argument('--datapath', default=None, type=str)
    parser.add_argument('--savepath', default=None, type=str)
    parser.add_argument('--logfile', default=None, type=str)
    parser.add_argument('--tensorboard_dirname', default=None, type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--num_rounds', default=300, type=int)
    parser.add_argument('--iter_per_round', default=150, type=int)
    parser.add_argument('--seed', default=1024, type=int)
    args = parser.parse_args()
    return args


def client_dataset_setup(datapath, indexing_file, mask_names, transforms, batch_size, alpha=None): 

    """
    Parameters: 
    datapath -- string, name of the root directory of dataset "BRATS2020"
    indexing_file -- string, name of the file containing training sample's indexes
    mask_names -- list, names list of avaliable modalities 
    transforms -- string, training transforms
    batch_size -- int, batch size

    Returns: 
    client_iterator -- object, multi-epoch iterator
    """    
    
    mask_generator = Mask_Generator(mask_names)

    client_dataset = BRATS2020(root=datapath, 
                               indexing_file=indexing_file,
                               train=True,
                               transforms=transforms, 
                               mask_generator=mask_generator,
                               alpha=alpha)

    client_dataloader = DataLoader(
        dataset=client_dataset,
        batch_size=batch_size,
        num_workers=3,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn)

    client_iterator = MultiEpochsIterator(client_dataloader)
    
    return client_iterator


def client_update(client_iterator, model, weights, optimizer, num_iterations):

    model.load_state_dict(weights)

    train_avg_loss, train_avg_loss_list = train_loop(client_iterator,
                                                     model, 
                                                     optimizer, 
                                                     num_iterations)
    updated_weights = copy.deepcopy(model.state_dict())

    return updated_weights, train_avg_loss, train_avg_loss_list


def weights_aggregate(model, weights_list, mask_names_list):

    updated_weights = model.state_dict()

    num_updated_flair_encoders = sum([contaim('flair', mask_names) for mask_names in mask_names_list])
    num_updated_t1ce_encoders = sum([contaim('t1c', mask_names) for mask_names in mask_names_list])
    num_updated_t1_encoders = sum([contaim('t1', mask_names) for mask_names in mask_names_list])
    num_updated_t2_encoders = sum([contaim('t2', mask_names) for mask_names in mask_names_list])
    print(num_updated_flair_encoders, num_updated_t1ce_encoders, num_updated_t1_encoders, num_updated_t2_encoders)

    for weight_layer_name in updated_weights:

        updated_weights[weight_layer_name] = torch.zeros_like(updated_weights[weight_layer_name])

        if "flair_encoder" in weight_layer_name:
            for k, client_weights in enumerate(weights_list):
                if contaim('flair', mask_names_list[k]):
                    updated_weights[weight_layer_name] += client_weights[weight_layer_name]
            updated_weights[weight_layer_name] /= num_updated_flair_encoders

        elif "t1ce_encoder" in weight_layer_name:
            for k, client_weights in enumerate(weights_list):
                if contaim('t1c', mask_names_list[k]):
                    updated_weights[weight_layer_name] += client_weights[weight_layer_name]
            updated_weights[weight_layer_name] /= num_updated_t1ce_encoders

        elif "t1_encoder" in weight_layer_name:
            for k, client_weights in enumerate(weights_list):
                if contaim('t1', mask_names_list[k]):
                    updated_weights[weight_layer_name] += client_weights[weight_layer_name]
            updated_weights[weight_layer_name] /= num_updated_t1_encoders

        elif "t2_encoder" in weight_layer_name:
            for k, client_weights in enumerate(weights_list):
                if contaim('t2', mask_names_list[k]):
                    updated_weights[weight_layer_name] += client_weights[weight_layer_name]
            updated_weights[weight_layer_name] /= num_updated_t2_encoders

        else:
            for client_weights in weights_list:
                updated_weights[weight_layer_name] += client_weights[weight_layer_name]
            updated_weights[weight_layer_name] /= len(weights_list)

    return updated_weights


def run():

    ############## Setup ###############

    args = parse()
    seed_setup(args.seed)
    logging_setup(args.savepath, lfile=args.logfile, to_console=True)
    ckpts = args.savepath
    writer = SummaryWriter(os.path.join(args.savepath, args.tensorboard_dirname))

    cudnn.benchmark = False 
    cudnn.deterministic = True

    ############## Dataset Preparation ##############

    num_clients = 4
    
    train_transforms = 'Compose([RandCrop3D(({},{},{})), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'.format(args.patch_size, args.patch_size, args.patch_size)

    client_masks = args.masks.split(':')
    mask_names_list = [client_mask.split('-') for client_mask in client_masks]

    client_iterators = []

    for k in range(num_clients):

        indexing_file = 'sites/sites{}/site{}'.format(num_clients, k)

        mask_names = mask_names_list[k]

        client_iterator = client_dataset_setup(args.datapath, 
                                               indexing_file, 
                                               mask_names, 
                                               train_transforms, 
                                               batch_size=args.batch_size,
                                               alpha=args.alpha)
        client_iterators.append(client_iterator)


    ########## Initializing Global Model ###########

    num_cls = 4
    model = Model(num_cls=num_cls).cuda()

    ######### Optimizers & Scheduler ###########

    lr_scheduler = LR_Scheduler(args.lr, args.num_rounds)

    optimizers = []

    for k in range(num_clients):
        train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
        optimizer = torch.optim.Adam(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
        optimizers.append(optimizer)

    ########## Training #########

    min_loss = 1000000

    for r in range(args.num_rounds):

        avg_train_loss = AverageMeter()
        avg_train_loss_list = AverageMeter()

        logging.info("Round: {}/{}".format(r+1, args.num_rounds))

        weights = copy.deepcopy(model.state_dict())

        updated_weights_list = []

        for k in range(num_clients):
            
            logging.info("Client {}/{}".format(k+1, num_clients))

            step_lr = lr_scheduler(optimizers[k], r)

            updated_weights, client_avg_loss, client_avg_loss_list = client_update(client_iterators[k], model, weights, optimizers[k], args.iter_per_round)

            updated_weights_list.append(updated_weights)

            avg_train_loss.update(client_avg_loss.avg)
            avg_train_loss_list.update(client_avg_loss_list.avg)

            tensorboard_record(client_avg_loss.avg, client_avg_loss_list.avg.tolist(), writer, r+1, lr=step_lr, prefix="client{}_".format(k+1))

            logging.info("Average Loss: {}".format(client_avg_loss.avg))

        weights = weights_aggregate(model, updated_weights_list, mask_names_list)

        tensorboard_record(avg_train_loss.avg, avg_train_loss_list.avg.tolist(), writer, r+1, prefix="round_")

        model.load_state_dict(weights)

        min_loss = model_save(model, ckpts, r, avg_train_loss.avg, min_loss)
  

if __name__ == '__main__':
    run()
