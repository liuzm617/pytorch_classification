#!/bin/bash
TIME=$(date "+%Y-%m-%d-%H-%M-%S")

#todo:
#数据集路径
DATASET_PATH=/e/hug/opts/animals

#输出模型及log路径
OUTPUT_PATH=./outputs

#TRAIN_LIST=./sample_files/imgs/listfile.txt
#VAL_LIST=./sample_files/imgs/listfile.txt

#训练labels文件
TRAIN_LIST=$DATASET_PATH/train_labels.txt

#评测及测试labels文件
VAL_LIST=$DATASET_PATH/test_labels.txt

#预训练模型
model_name=resnet101

#学习率
lr=0.01

#训练轮数
epochs=50

# 批大小，太大会超内存或者显存
batch_size=32

# workers数量
num_workers=4

# 类别数(labels去重数量)
num_classes=8

#评测或预测模型路径
model_path=./outputs/2024-03-23-18-59-16/precision_0.5600_num_40/epoch_40.pth


export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=2

#训练
# CUDA_VISIBLE_DEVICES=0,1,2,3 \

function train() {
      python -u -m torch.distributed.launch --nproc_per_node 1  ./tools/train_val.py \
     --model_name=$model_name \
     --lr  $lr --epochs $epochs  --batch-size $batch_size  -j $num_workers \
     --output=$OUTPUT_PATH/$TIME \
     --train_list=$TRAIN_LIST \
     --val_list=$VAL_LIST \
     --num_classes=$num_classes \
     --is_pretrained

}




#评测
function eval() {
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
 python -u -m torch.distributed.launch --nproc_per_node 1  ./tools/evaluation.py \
     --model_name=$model_name \
     --batch-size $batch_size  -j $num_workers \
     --output=$OUTPUT_PATH/$TIME \
     --val_list=$VAL_LIST \
     --tune_from=$model_path \
     --num_classes=$num_classes

}


#预测
function predict() {
   # CUDA_VISIBLE_DEVICES=0,1,2,3 \
 python -u -m torch.distributed.launch --nproc_per_node 1  ./tools/predict.py \
     --model_name=$model_name \
     --batch-size $batch_size  -j $num_workers \
     --output=$OUTPUT_PATH/$TIME \
     --val_list=$VAL_LIST \
     --tune_from=$model_path \
     --num_classes=$num_classes
}



case "$1" in
  train)
    echo "训练"
    train
    ;;
  eval)
    echo "评测"
    eval
    ;;
  predict)
    echo "预测"
    predict
    ;;
  *)
    echo "未知参数: $1"
    exit 1
    ;;
esac



#export MASTER_ADDR=127.0.0.1
