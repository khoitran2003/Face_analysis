import yaml
import argparse
import torch
import cv2
from src.classification.train import Classify
from src.classification.data_loader_clf import ClassifierData, Get_Loader
from src.classification.focal_loss import cal_weight


def get_args():
    parsers = argparse.ArgumentParser("Classifier task")
    parsers.add_argument(
        "--cfg",
        type=str,
        default="cfg/classifier.yaml",
    )
    parsers.add_argument(
        "--label_cfg",
        type=str,
        default="cfg/labels.yaml",
    )
    args = parsers.parse_args()
    return args


def main(args):
    with open(args.cfg) as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    with open(args.label_cfg) as label_cfg_file:
        label_cfg = yaml.safe_load(label_cfg_file)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clf = Classify(cfg, label_cfg, device)
    clf.train()

    # dataloader = Get_Loader(cfg, label_cfg)
    # train_loader, val_loader = dataloader.load_train_val()
    
    # for images, labels in val_loader:
    #     print(images.shape)





if __name__ == "__main__":
    args = get_args()
    main(args)
