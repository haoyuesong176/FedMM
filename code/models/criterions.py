import torch

def dice_loss(output, target, mask=None, eps=1e-7):
    
    """
    Params:
    output -- segmentation results of shape (b, num_cls, patch, patch, patch) 
    target -- one-hot ground truth labels of shape (b, num_cls, patch, patch, patch)
    num_cls -- number of classes of the target
    mask -- to mask irrelevant modalities from Decoder_sep

    Returns:
    loss -- it equals "1 - (averaged instance dice scores over num_cls & num_batch_instances)   
    """

    num_cls = output.shape[1]

    if mask == None:
        num_batch_instances = output.shape[0]
        mask = torch.ones(num_batch_instances, dtype=torch.bool).cuda()

    num_unmasked_instances = torch.sum(mask)
    if torch.sum(num_unmasked_instances) == 0:
        return 0.0

    target = target.float()
    for i in range(num_cls):
        num = torch.sum(output[:,i,:,:,:] * target[:,i,:,:,:], dim=[1,2,3])
        l = torch.sum(output[:,i,:,:,:], dim=[1,2,3])
        r = torch.sum(target[:,i,:,:,:], dim=[1,2,3])
        if i == 0:
            dice = (2.0 * num / (l+r+eps)) 
        else:
            dice += (2.0 * num / (l+r+eps)) 
    dice = torch.sum(dice * mask) / num_unmasked_instances
    loss = 1.0 - 1.0 * dice / num_cls
    return loss


def softmax_weighted_loss(output, target, mask=None):

    """
    Params:
    output -- segmentation results of shape (b, num_cls, patch, patch, patch) 
    target -- one-hot ground truth labels of shape (b, num_cls, patch, patch, patch)
    num_cls -- number of classes of the target
    mask -- to mask irrelevant modalities from Decoder_sep

    Returns:
    cross_loss -- weighted cross entropy loss 
    """

    if mask == None:
        num_batch_instances = output.shape[0]
        mask = torch.ones(num_batch_instances, dtype=torch.bool).cuda()

    num_unmasked_instances = torch.sum(mask)
    if torch.sum(num_unmasked_instances) == 0:
        return 0.0

    target = target.float()
    B, num_cls, H, W, Z = output.size()

    for i in range(num_cls):
        outputi = output[:, i, :, :, :] 
        targeti = target[:, i, :, :, :] 
        weighted = 1.0 - (torch.sum(targeti, (1,2,3)) * 1.0 / torch.sum(target, (1,2,3,4)))
        weighted = torch.reshape(weighted, (-1,1,1,1)).repeat(1,H,W,Z)
        if i == 0:
            cross_loss = -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
        else:
            cross_loss += -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
    cross_loss = torch.mean(cross_loss, dim=[1,2,3])
    cross_loss = torch.sum(cross_loss * mask) / num_unmasked_instances
    return cross_loss


def loss_compute(fuse_pred, sep_preds, target, mask):
    
    """
    Params:
    fuse_pred -- final predictions of Decoder_fuse, which are of shape (b, num_cls, patch, patch, patch)
    sep_preds -- tuple of predictions of Decoder_sep, whose elements are of shape (b, num_cls, patch, patch, patch)
    target -- one-hot ground truth labels of shape (b, num_cls, patch, patch, patch)
    mask -- to mask irrelevant modalities from Decoder_sep; mask is of shape (b, 4)

    Returns:
    loss -- overall loss
    fuse_cross_loss -- cross loss of fuse_pred
    fuse_dice_loss -- dice loss of fuse_pred
    sep_cross_loss -- sum over 4 sep_preds' cross losses
    sep_dice_loss -- sum over 4 sep_preds' dice losses
    """

    fuse_cross_loss = softmax_weighted_loss(fuse_pred, target)
    fuse_dice_loss = dice_loss(fuse_pred, target)
    fuse_loss = fuse_cross_loss + fuse_dice_loss

    sep_cross_loss = torch.zeros(1).cuda().float()
    sep_dice_loss = torch.zeros(1).cuda().float()
    for i, sep_pred in enumerate(sep_preds):
        modal_mask = mask[:, i]
        sep_cross_loss += softmax_weighted_loss(sep_pred, target, modal_mask)
        sep_dice_loss += dice_loss(sep_pred, target, modal_mask)
    sep_loss = sep_cross_loss + sep_dice_loss

    loss = fuse_loss + sep_loss

    return loss, (fuse_cross_loss, fuse_dice_loss, sep_cross_loss, sep_dice_loss)
