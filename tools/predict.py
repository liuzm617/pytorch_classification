import json
import os
import sys
from pathlib import Path

import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import tqdm
from PIL import Image

FILE = Path(__file__).resolve()

ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from utils import init_logger
from utils import parser

from dataset import val_transform
from cls_models import ClsModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def __init_model(args):
    logger = init_logger(log_file=args.output + f'/log', rank=-1)

    model = ClsModel(args.model_name, args.num_classes)
    if args.tune_from and os.path.exists(args.tune_from):
        print(f'loading model from {args.tune_from}')
        sd = torch.load(args.tune_from, map_location='cpu')
        model.load_state_dict(sd)

    model.to(device)
    model.eval()

    cudnn.benchmark = True
    return logger, model


def predict(img_path):
    img = Image.open(img_path).convert('RGB')
    img_tensor = val_transform(size=args.input_size)(img).unsqueeze(0)

    with torch.no_grad():
        preds, labels, scores = [], [], []
        img_tensor = img_tensor.to(device)

        output = model(img_tensor)

        scores = torch.softmax(output, dim=1)
        score = torch.max(scores, dim=1)[0].item()
        pred = torch.max(scores, dim=1)[1].item()

    return pred, score


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.output]
    for folder in folders_util:
        os.makedirs(folder, exist_ok=True)


def counter(data):
    counter_dict = {}

    for k in data:
        num = len(data[k])
        counter_dict[k] = {
            "总数": num,
            "准确数": 0,
            "准确率": 0,
            "错误数": 0,
            "识别失败列表": [],
        }

        i = 0
        for r in data[k]:
            if k == r["result"]:
                i += 1
            else:
                counter_dict[k]["识别失败列表"].append(r["file"])

        counter_dict[k]["准确数"] = i
        counter_dict[k]["错误数"] = num - i
        counter_dict[k]["准确率"] = i / num

    return counter_dict


if __name__ == '__main__':

    args = parser.parse_args()
    check_rootfolders()
    logger, model = __init_model(args)

    for k, v in sorted(vars(args).items()):
        logger.info(f'{k} = {v}')

    pred_file = args.val_list

    datas = open(pred_file, 'r', encoding='utf8').readlines()

    label_map = {}

    for line in datas:
        line = line.strip()
        if not line:
            continue
        lines = line.split('\t')
        # class_name = lines[0].split('/')[-2]
        class_name = lines[2]
        label = int(lines[1])
        label_map[label] = class_name

    print(label_map)

    results = dict()

    for data in tqdm.tqdm(datas):
        lines = data.strip().split('\t')
        class_name = lines[2]
        label = lines[1]

        path = lines[0]
        pred, score = predict(path)

        print("path:", path, "pred:", pred, "score:", score)

        res = {
            "file": path,
            "result": label_map.get(pred, "-1"),
            "score": score,
            "err_msg": ""
        }

        if results.get(class_name, None) is None:
            results[class_name] = [
                res
            ]
        else:
            results[class_name].append(res)

    with open(os.path.join(args.output, 'predict_result.json'), 'w', encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    counter_result = counter(results)
    with open(os.path.join(args.output, "counter_result.json"), "w", encoding="utf-8") as f:
        json.dump(counter_result, f, ensure_ascii=False, indent=4)
