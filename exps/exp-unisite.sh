#!/bin/bash

#------------ Config -----------#

gpu=4

lr=2e-4
batch_size=2
patch_size=80
iterations=150
epochs=300

datapath=data
trainfile=sites/sites4/site3
masks=flairt2

output_root=output
exp_name=exp-brats-local-s2-site3
savepath=$output_root/$exp_name
logfile=train_log.txt
tensorboard_dirname=tensorboard


#------------ Script ------------#

CUDA_VISIBLE_DEVICES=$gpu python3 code/uni_train.py --batch_size $batch_size --patch_size $patch_size --iter_per_epoch $iterations --datapath $datapath --savepath $savepath --num_epochs $epochs --lr $lr --masks $masks --trainfile $trainfile --logfile $logfile --tensorboard_dirname $tensorboard_dirname

