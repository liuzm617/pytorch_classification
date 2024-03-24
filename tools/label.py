import os

from utils import parser


def generate_labels(dataset_path):
    if not os.path.isdir(dataset_path):
        raise Exception(f"Invalid dataset path:{dataset_path}")

    dirs = os.listdir(dataset_path)
    for d in dirs:
        d_path = os.path.join(dataset_path, d)
        if os.path.isdir(d_path):
            if d == "train":
                print("Generating train labels...")
                generate_label_file(d_path, os.path.join(dataset_path, "train_labels.txt"))
                print("done")
            elif d == "test":
                print("Generating test labels...")
                generate_label_file(d_path, os.path.join(dataset_path, "test_labels.txt"))
                print("done")


# 生成label文件，格式为 图片路径  标签（数字） 目录名（）
# 对于目录要求1_cat格式,前面数字作为训练label,因为label要求数字，要是自动生成这个label,那么后续每次增加目录可能导致混乱
def generate_label_file(path, label_file_name):
    with open(label_file_name, 'w', encoding='utf8') as f:
        for d in os.listdir(path):
            labels = d.split("_", maxsplit=1)
            if len(labels) < 2 or (not labels[0].isdigit()):
                raise ValueError(f"Invalid directory name:{d}, must be in the format of <number>_dirname")

            label = labels[0]

            d_path = os.path.join(path, d)
            if os.path.isdir(d_path):
                for file in os.listdir(d_path):
                    file_path = os.path.join(d_path, file)
                    if os.path.isfile(file_path):
                        f.write(f"{file_path}\t{label}\t{labels[1]}\n")


if __name__ == "__main__":
    args = parser.parse_args()
    if not args.dataset:
        raise Exception("Please specify the dataset path")

    generate_labels(args.dataset)
