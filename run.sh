#!/bin/bash
TIME=$(date "+%Y-%m-%d-%H-%M-%S")

#todo:
#1. 修改数据集路径
DATASET_PATH=/e/hug/opts/animals

OUTPUT_PATH=./outputs
#TRAIN_LIST=./sample_files/imgs/listfile.txt
#VAL_LIST=./sample_files/imgs/listfile.txt

TRAIN_LIST=$DATASET_PATH/train_labels.txt
VAL_LIST=$DATASET_PATH/test_labels.txt


export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=2
#export MASTER_ADDR=127.0.0.1
#训练
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -u -m torch.distributed.launch --nproc_per_node 1  ./tools/train_val.py \
   --model_name=resnet101 \
   --lr  0.01 --epochs 50  --batch-size 32  -j 4 \
   --output=$OUTPUT_PATH/$TIME \
   --train_list=$TRAIN_LIST \
   --val_list=$VAL_LIST \
   --num_classes=8 \
   --is_pretrained



    
#评测
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python3 -u -m torch.distributed.launch --nproc_per_node 1  ./tools/evaluation.py \
#     --model_name=resnet18 \
#     --batch-size 64  -j 4 \
#     --output=$OUTPUT_PATH/$TIME \
#     --val_list=$VAL_LIST \
#     --tune_from='/home/lxztju/pytorch_classification/ouputs/xxx/epoch_4.pth' \
#     --num_classes=2
    

#预测
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python3 -u -m torch.distributed.launch --nproc_per_node 1  ./tools/predict.py \
#     --model_name=resnet18 \
#     --batch-size 64  -j 4 \
#     --output=$OUTPUT_PATH/$TIME \
#     --val_list=$VAL_LIST \
#     --tune_from='/home/lxztju/pytorch_classification/ouputs/xxx/epoch_4.pth' \
#     --num_classes=2
