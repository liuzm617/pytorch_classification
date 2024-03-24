## 简介

pytorch图像分类功能。

## 运行环境

* python3.7+
* pytorch 2.1.0(要使用gpu需要安装gpu驱动，并对应gpu包)
* torchvision 0.16.0
* opencv(libtorch cpp推理使用， 版本3.4.6)（可选）
* libtorch cpp推理使用（可选）

## 快速开始

### 数据集形式

数据集组织形式：

```
├── dataset
│   ├── train
│   │   ├── num_classes(如1_小鹿)
│   │   │   ├──1.png
│   │   │   ├──2.png
│   ├── test
│   │   ├── num_classes
│   │   │   ├──1.png
│   │   │   ├──2.png
│   ├── test_labels.txt (使用代码自动生成)
│   ├── train_labels.txt (使用代码自动生成)

```

todo:

数据集目录要求num_classes形式说明：

本身可以自动根据目录自动生成label,因其label值需要是数字类型，需要保证每次新增类别时，其他已有的label保持不变，比如训练了多个模型比如V1，V2版本，

V1使用数据集a1(label=0),b1(label=1),c1(label=2);V2版本可能新增了数据集类型b11，如果自动处理则分类顺序可能会是a1(0),b1(1)
,b11(2),c1(3);

如果预测的某个图片正好匹配到c1分类，此时V1版本输出为2，V2版本输出为3，不同版本结果(数字label)会不一致,

虽然可以通过生成的labels转换成对应的真实分类名，但是如果此时V1版本的labels文件被覆盖或者其他原因丢失，

那么V1模型使用V2的labels文件进行预测转换就会出问题，

相反，如果保证每次新加数据集，使得数字 label值累加，那么所有版本模型都可以用最后生成的labels文件进行预测转换。

### 自动生成labels.txt

修改DATASET_PATH为数据集路径

运行./generate_labels.sh 在数据集目录下下生成test_labels.txt、train_labels.txt

文件内容组织格式：

`文件路径 数字label 分类名`

### 训练 测试

修改`run.sh`中的参数，直接运行run.sh即可运行

训练：

run.sh train

评测：
run.sh eval

预测：
run.sh predict

主要修改的参数：

```
DATASET_PATH 数据集路径
OUTPUT_PATH 模型保存和log文件的路径
TRAIN_LIST 训练数据集的list文件
VAL_LIST  测试集合的list文件
model_name 默认是resnet50
lr 学习率
epochs 训练总的epoch
batch-size  batch的大小
j dataloader的num_workers的大小
num_classes 类别数

model_path 评测或预测模型路径
```

### libtorch inference

代码存储在`cpp_inference`文件夹中。

1.
利用[cpp_inference/traced_model/trace_model.py](https://github.com/lxztju/pytorch_classification/blob/master/cpp_inference/traced_model/trace_model.py)
将训练好的模型导出。
2. 编译所需的opencv和libtorch代码到`cpp_inference/third_party_library`

3. 编译

```
sh compile.sh
```

4. 可执行文件测试

```
./bin/imgCls imgpath
```


