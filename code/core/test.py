import torch
import numpy as np
import os
import logging
import csv
from code.utils.utils import AverageMeter


def dice(sep_output, sep_target, eps=1e-8):

    """
    Parameters:
    sep_output -- binary predictions of shape (b, H, W, Z)
    sep_target -- binary ground truth of shape (b, H, W, Z)
    
    Returns:
    ret -- dice scores of batch instances, of shape (b)
    """

    intersect = torch.sum(2 * (sep_output * sep_target), dim=(1,2,3)) + eps
    denominator = torch.sum(sep_output, dim=(1,2,3)) + torch.sum(sep_target, dim=(1,2,3)) + eps
    return intersect / denominator
    

def get_dice_score(output, target, labels):

    """
    Parameters:
    output -- predictions of shape (b, H, W, Z)
    target -- ground truth labels of shape (b, H, W, Z)
    labels -- labels defining regions to be evaluated

    Returns:
    ret -- dice scores of specific region (defined by labels) of batch instances, of shape (b)
    """

    sep_outputs = torch.zeros_like(output).float()
    sep_targets = torch.zeros_like(output).float()
    for label in labels:
        sep_outputs += (output == label).float()
        sep_targets += (target == label).float()
    return dice(sep_outputs, sep_targets)
    

def post_et_dice(output, target, et_label=3):

    """
    Parameters:
    output -- predictions of shape (b, H, W, Z)
    target -- ground truth labels of shape (b, H, W, Z)
    et_label -- label of enhancing

    ret -- dice score of post-processed ET
    """

    et_output = (output == et_label).float()
    et_target = (target == et_label).float()
    if torch.sum(et_output) < 500:
       sep_output = et_output * 0.0
    else:
       sep_output = et_output
    sep_target = et_target
    return dice(sep_output, sep_target)


def dice_compute(output, target):

    """
    Parameters:
    output -- predictions of shape (b, H, W, Z)
    target -- ground truth labels of shape (b, H, W, Z)

    Returns:
    ret1 -- names of targets given in the same order as targets_dice_scores
    ret2 -- ndarray, dice scores of combined regions of batch instances, of shape (b, 4)
    """

    targets_names = 'whole', 'core', 'enhancing', 'enhancing_postpro'

    enhancing_dice = get_dice_score(output, target, labels=[3])
    enhancing_dice_postpro = post_et_dice(output, target, et_label=3)
    dice_whole = get_dice_score(output, target, labels=[1, 2, 3])
    dice_core = get_dice_score(output, target, labels=[1, 3])

    targets_dice_scores = torch.cat((torch.unsqueeze(dice_whole, 1), torch.unsqueeze(dice_core, 1), torch.unsqueeze(enhancing_dice, 1), torch.unsqueeze(enhancing_dice_postpro, 1)), dim=1)

    return targets_names, targets_dice_scores.cpu().numpy()


def get_sliding_windows(H, W, Z, patch_size):

    """
    Parameters:
    H -- int, height of the input image 
    W -- int, width of the input image
    Z -- int, depth of the input image
    patch_size -- patch size

    h_idx_list - list, first indexes in the height dimension
    w_idx_list - list, first indexes in the width dimension
    z_idx_list - list, first indexes in the depth dimension
    """

    h_cnt = np.int(np.ceil((H - patch_size) / (patch_size * 0.5))) 
    h_idx_list = range(0, h_cnt)
    h_idx_list = [h_idx * np.int(patch_size * 0.5) for h_idx in h_idx_list]
    h_idx_list.append(H - patch_size)

    w_cnt = np.int(np.ceil((W - patch_size) / (patch_size * 0.5)))
    w_idx_list = range(0, w_cnt)
    w_idx_list = [w_idx * np.int(patch_size * 0.5) for w_idx in w_idx_list]
    w_idx_list.append(W - patch_size)

    z_cnt = np.int(np.ceil((Z - patch_size) / (patch_size * 0.5)))
    z_idx_list = range(0, z_cnt)
    z_idx_list = [z_idx * np.int(patch_size * 0.5) for z_idx in z_idx_list]
    z_idx_list.append(Z - patch_size)

    return h_idx_list, w_idx_list, z_idx_list


def predict(model, x, mask, patch_size):

    """
    Parameters:
    model -- object, pretrained RFNet model
    x -- tensor, model input of shape (b, 4, H, W, Z)
    mask -- tensor, modalities masks of shape (b, 4)
    patch_size -- patch size of the pretrained model

    Returns:
    pred -- tensor, model prediction output of shape (b, num_cls, H, W, Z)
    """

    B, _, H, W, Z = x.size()
    h_idx_list, w_idx_list, z_idx_list = get_sliding_windows(H, W, Z, patch_size)

    num_cls = model.num_cls
    one_tensor = torch.ones(1, 1, patch_size, patch_size, patch_size).float().cuda() 
    weight = torch.zeros(1, 1, H, W, Z).float().cuda()
    pred = torch.zeros(B, num_cls, H, W, Z).float().cuda()

    with torch.no_grad():

        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    x_input = x[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size]
                    pred_part = model(x_input, mask)
                    pred[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size] += pred_part
                    weight[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size] += one_tensor

    weight = weight.repeat(B, num_cls, 1, 1, 1)
    pred = pred / weight
    pred = torch.argmax(pred, dim=1)
    return pred


def csv_write(savepath, file_name, data):
    
    """
    Parameters:
    savepath -- path of the directory 
    file_name -- name of the csv file 
    data -- data to record
    """
    
    csv_path = os.path.join(savepath, file_name)

    with open(csv_path, mode='w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)


def test_loop(dataloader, model, feature_mask, patch_size):

    """
    Parameters:
    dataloader -- object, test dataloader
    model -- object, pretrained RFNet model
    feature_mask -- applied mask within this loop
    patch_size -- pretrained model's patch size

    Returns:
    ret1 -- list, names of targets given in the same order as ret2
    ret2 -- ndarray, targets' dice scores averaged over all dataset instances
    """

    avg_targets_dice_scores = AverageMeter()

    model.eval()

    for i, data in enumerate(dataloader):

        x = data[0].cuda()
        target = data[1].cuda()
        names = data[-1]

        mask = torch.from_numpy(np.array(feature_mask)) 
        mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
        mask = mask.cuda()

        pred = predict(model, x, mask, patch_size)

        targets_names, targets_dice_scores = dice_compute(pred, target)

        for k, name in enumerate(names):
            avg_targets_dice_scores.update(targets_dice_scores[k])
            
            
            msg = 'Subject {}/{}, {}/{}'.format((i+1), len(dataloader), (k+1), len(names))
            msg += '{:>20}, '.format(name)
            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(targets_names, targets_dice_scores[k])])
            logging.info(msg)
            
            
    return targets_names, avg_targets_dice_scores.avg

