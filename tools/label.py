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


def generate_label_file(path, label_file_name):
    with open(label_file_name, 'w', encoding='utf8') as f:
        for label_name in os.listdir(path):
            d_path = os.path.join(path, label_name)
            if os.path.isdir(d_path):
                for file in os.listdir(d_path):
                    file_path = os.path.join(d_path, file)
                    if os.path.isfile(file_path):
                        f.write(f"{file_path}\t{label_name}\n")


if __name__ == "__main__":
    args = parser.parse_args()
    if not args.dataset:
        raise Exception("Please specify the dataset path")

    generate_labels(args.dataset)
