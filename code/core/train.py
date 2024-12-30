import torch

import os
import logging
import numpy as np

from code.utils.utils import AverageMeter
from code.models.criterions import loss_compute


def model_save(model, ckpts, step, loss, min_loss, freq=50):

    """
    Parameters:
    model -- RFNet model
    ckpts -- checkpoint savepath
    step -- current step
    loss -- loss of current communication round
    min_loss -- minimum loss traced
    freq -- frequency of model recording
    """

    if (step + 1) % freq == 0:
        file_name = os.path.join(ckpts, 'model_{}.pth'.format(step+1))
        torch.save(model.state_dict(), file_name)

    if loss < min_loss:
        file_name = os.path.join(ckpts, 'model_best.pth')
        torch.save(model.state_dict(), file_name)
    
    return loss if loss < min_loss else min_loss


def tensorboard_record(loss, loss_list, writer, step, lr=None, prefix=''):

    """
    Parameters: 
    loss -- float, overall loss
    loss_list -- list, sub-losses
    writer -- object, tensorboard summary writer
    step -- int, global step of tensorboard scalar 
    lr -- float, learning rate
    prefix -- string, adding prefix to names of scalars
    """

    writer.add_scalar(prefix + 'loss', loss, global_step=step)
    writer.add_scalar(prefix + 'fuse_cross_loss', loss_list[0], global_step=step)
    writer.add_scalar(prefix + 'fuse_dice_loss', loss_list[1], global_step=step)
    writer.add_scalar(prefix + 'sep_cross_loss', loss_list[2], global_step=step)
    writer.add_scalar(prefix + 'sep_dice_loss', loss_list[3], global_step=step)
    if lr is not None:
        writer.add_scalar(prefix + 'lr', lr, global_step=step)


def train_loop(iterator, model, optimizer, num_iterations):
    
    """
    Parameters:
    iterator -- object, multi-epochs iterator
    model -- object, Trans Model
    optimizer -- object, Adam Optimizer
    num_iterations -- number of iterations per training loop

    Returns:
    train_avg_loss -- float, training loss average
    train_avg_loss_list -- list, training sub-losses averages
    """

    train_avg_loss = AverageMeter()
    train_avg_loss_list = AverageMeter()
    
    model.train()

    for i in range(num_iterations):

        # DATA
        data = next(iterator)
        x, target, mask = data[:3]
        x = x.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        # print(mask)

        # MODEL
        fuse_pred, sep_preds = model(x, mask)
        loss, loss_list = loss_compute(fuse_pred, sep_preds, target, mask)

        # GD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        msg = 'Iter {}/{}, Loss {:.4f}, '.format((i+1), num_iterations, loss.item())
        msg += 'fusecross:{:.4f}, fusedice:{:.4f},'.format(loss_list[0].item(), loss_list[1].item())
        msg += 'sepcross:{:.4f}, sepdice:{:.4f},'.format(loss_list[2].item(), loss_list[3].item())
        logging.info(msg)        
        
        train_avg_loss.update(loss.item())
        train_avg_loss_list.update(np.array([l.item() for l in loss_list]))        

    return train_avg_loss, train_avg_loss_list
