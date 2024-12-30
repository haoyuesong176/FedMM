#!/bin/bash


#--------- Config ----------#

gpu=7

alpha=0.6 # alpha = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] + [uniform]

lr=2e-4
batch_size=2
patch_size=80
iterations=150
num_rounds=300

datapath=data
masks=flair-t1c-flairt1ce:t1c-t1-t1cet1:t1-t2-t1t2:t2-flair-flairt2

save_root=output
exp_name=exp-brats-fedmm-s2-alpha0.6

savepath=$save_root/$exp_name
logfile=train_log.txt
tensorboard_dirname=tensorboard


#--------- Script ----------#

CUDA_VISIBLE_DEVICES=$gpu python3 code/fed_train.py --batch_size $batch_size --patch_size $patch_size --iter_per_round $iterations --datapath $datapath --savepath $savepath --num_rounds $num_rounds --lr $lr --logfile $logfile --tensorboard_dirname $tensorboard_dirname --masks $masks --alpha $alpha

