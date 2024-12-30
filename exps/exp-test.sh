#!/bin/bash

#------------ Config -----------#

gpu=4
batch_size=1
patch_size=80

checkpoint=output/exp-brats-local-s2-site3/model_300.pth
datapath=data
testfile=test3.txt
masks=flairt2

save_root=output
exp_name=exp-brats-local-s2-site3
logfile=test_log_300_site3.txt
csvfile=test_csv_300_site3.csv
savepath=$save_root/$exp_name

#------------ Script ------------#

CUDA_VISIBLE_DEVICES=$gpu python3 code/test.py --batch_size $batch_size --patch_size $patch_size --masks $masks --datapath $datapath --savepath $savepath --checkpoint $checkpoint --logfile $logfile --csvfile $csvfile --testfile $testfile

