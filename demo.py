import argparse
import os
import numpy as np
import yaml
import matplotlib.pyplot as plt

import cv2
import torch
from torch.autograd import Variable

from models.yolov3 import *
from utils.utils import *
from utils.parse_yolo_weights import parse_yolo_weights
from utils.vis_bbox import vis_bbox


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str)
    parser.add_argument('--data', type=str, choices=['coco', 'drone'], default='drone')
    parser.add_argument('--detect_thresh', type=float,
                        default=0.5, help='confidence threshold')
    parser.add_argument('--ckpt', type=str,
                        help='path to the checkpoint file')
    parser.add_argument('--weights_path', '-w', type=str,
                        default=None, help='path to weights file')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--window', action='store_true',
                        help='show result on matplotlib window. Otherwise, save as a PNG file')
    return parser.parse_args()


def main():
    args = parse_args()

    print("------------------------------------")
    print("    use {} dataset for demo.        ".format(args.data))
    print("------------------------------------")

    assert args.data in ['coco', 'drone']

    # [TBM] gen n_classes from coco-format json file..
    if args.data == 'coco':
        cfg_path = 'config/yolov3_default.cfg'
        n_classes = 80
    if args.data == 'drone':
        cfg_path = 'config/yolov3_visdrone_default.cfg'
        n_classes = 4

    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f)

    imgsize = cfg['TEST']['IMGSIZE']
    model = YOLOv3(n_classes=n_classes)
    confthre = cfg['TEST']['CONFTHRE'] 
    nmsthre = cfg['TEST']['NMSTHRE']

    if args.detect_thresh:
        confthre = args.detect_thresh

    img = cv2.imread(args.image)
    assert img is not None

    img_raw = img.copy()[:, :, ::-1].transpose((2, 0, 1))
    img, info_img = preprocess(img, imgsize)  # info = (h, w, nh, nw, dx, dy)
    img = torch.from_numpy(img).float().unsqueeze(0)

    if args.gpu >= 0:
        model.cuda(args.gpu)
        img = Variable(img.type(torch.cuda.FloatTensor))
    else:
        img = Variable(img.type(torch.FloatTensor))

    if args.weights_path:
        print("loading yolo weights %s" % (args.weights_path))
        parse_yolo_weights(model, args.weights_path)
    else:
        print("loading checkpoint %s" % (args.ckpt))
        model.load_state_dict(torch.load(args.ckpt))

    model.eval()

    with torch.no_grad():
        outputs = model(img)
        outputs = postprocess(outputs, n_classes, confthre, nmsthre)
    
    # [TBM] gen label_names from coco-format json file..
    if args.data == 'coco':
        class_names, class_ids, class_colors = get_coco_label_names()
    if args.data == 'drone':
        class_names, class_ids, class_colors = get_visdrone_label_names()

    bboxes = list()
    classes = list()
    colors = list()

    for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs[0]:

        cls_id = class_ids[int(cls_pred)]
        print(int(x1), int(y1), int(x2), int(y2), float(conf), int(cls_pred))
        print('\t+ Label: %s, Conf: %.5f' %
              (class_names[cls_id], cls_conf.item()))
        box = yolobox2label([y1, x1, y2, x2], info_img)
        bboxes.append(box)
        classes.append(cls_id)
        colors.append(class_colors[int(cls_pred)])
    
    vis_bbox(
        img_raw, bboxes, label=classes, label_names=class_names,
        instance_colors=colors, linewidth=2)
    
    if args.window:
        plt.show()
    else:
        out_path = './output.png'
        plt.savefig(out_path, bbox_inches=0, pad_inches=0, dpi=100)


if __name__ == '__main__':
    main()
