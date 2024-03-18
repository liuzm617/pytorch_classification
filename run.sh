#!/bin/bash
TIME=$(date "+%Y-%m-%d-%H-%M-%S")


OUTPUT_PATH=./outputs
TRAIN_LIST=./sample_files/imgs/listfile.txt
VAL_LIST=./sample_files/imgs/listfile.txt

export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=2
#export MASTER_ADDR=127.0.0.1
#train
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -u -m torch.distributed.launch --nproc_per_node 1  ./tools/train_val.py \
   --model_name=resnet18 \
   --lr  0.01 --epochs 70  --batch-size 128  -j 4 \
   --output=$OUTPUT_PATH/$TIME \
   --train_list=$TRAIN_LIST \
   --val_list=$VAL_LIST \
   --num_classes=2 \
   --is_pretrained



    
    
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python3 -u -m torch.distributed.launch --nproc_per_node 1  ./tools/evaluation.py \
#     --model_name=resnet18 \
#     --batch-size 64  -j 4 \
#     --output=$OUTPUT_PATH/$TIME \
#     --val_list=$VAL_LIST \
#     --tune_from='/home/lxztju/pytorch_classification/ouputs/xxx/epoch_4.pth' \
#     --num_classes=2
    

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python3 -u -m torch.distributed.launch --nproc_per_node 1  ./tools/predict.py \
#     --model_name=resnet18 \
#     --batch-size 64  -j 4 \
#     --output=$OUTPUT_PATH/$TIME \
#     --val_list=$VAL_LIST \
#     --tune_from='/home/lxztju/pytorch_classification/ouputs/xxx/epoch_4.pth' \
#     --num_classes=2
