#!/bin/bash
#根据目录自动生成label文件
export PYTHONPATH=.

echo $PYTHONPATH

DATASET_PATH=/e/hug/opts/animals


python ./tools/label.py --dataset $DATASET_PATH
