import argparse
import os
import numpy as np
import yaml
import time
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
    parser.add_argument('--in_dir', '-i', type=str,
                        help='VisDrone sequence images')
    parser.add_argument('--out_dir', '-o', type=str, default='det_results',
                        help='Output directory')
    parser.add_argument('--step', '-s', type=int, default=10, 
                        help='Step to sample input images')
    parser.add_argument('--data', '-d', type=str, choices=['coco', 'drone'], default='drone')
    parser.add_argument('--detect_thresh', '-t', type=float, default=0.5, 
                        help='confidence threshold')
    parser.add_argument('--ckpt', '-c', type=str,
                        help='path to the checkpoint file')
    parser.add_argument('--weights_path', '-w', type=str, default=None, 
                        help='path to weights file')
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--verbose', '-v', action='store_true')
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

    if args.gpu >= 0:
        model.cuda(args.gpu)

    if args.weights_path:
        print("loading yolo weights %s" % (args.weights_path))
        parse_yolo_weights(model, args.weights_path)
    else:
        print("loading checkpoint %s" % (args.ckpt))
        model.load_state_dict(torch.load(args.ckpt))

    model.eval()

    dir_name = os.path.basename(os.path.dirname(args.in_dir + '/'))
    out_dir = os.path.join(args.out_dir, dir_name)
    os.makedirs(out_dir, exist_ok=True)

    img_files = os.listdir(args.in_dir)
    img_files.sort()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    for i in range(0, len(img_files), args.step):

        filename = img_files[i]
        img_path = os.path.join(args.in_dir, filename)
        img = cv2.imread(img_path)
        assert img is not None

        start.record()

        img_raw = img.copy()[:, :, ::-1].transpose((2, 0, 1))
        img, info_img = preprocess(img, imgsize)  # info = (h, w, nh, nw, dx, dy)
        img = torch.from_numpy(img).float().unsqueeze(0)

        if args.gpu >= 0:
            img = Variable(img.type(torch.cuda.FloatTensor))
        else:
            img = Variable(img.type(torch.FloatTensor))

        with torch.no_grad():
            outputs = model(img)
            outputs = postprocess(outputs, n_classes, confthre, nmsthre)
        
        end.record()
        torch.cuda.synchronize()

        # [TBM] gen label_names from coco-format json file..
        if args.data == 'coco':
            class_names, class_ids, class_colors = get_coco_label_names()
        if args.data == 'drone':
            class_names, class_ids, class_colors = get_visdrone_label_names()
        

        bboxes, classes, colors = list(), list(), list()

        if outputs[0] is None:
            outputs[0] = []
        
        if args.verbose:
            print("=====================================")
        
        print("{}, {:.2f} [fps]".format(filename, 1000.0 / start.elapsed_time(end)))

        for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs[0]:
            cls_id = class_ids[int(cls_pred)]
            if args.verbose:
                print(int(x1), int(y1), int(x2), int(y2), float(conf), int(cls_pred))
                print('\t+ Label: %s, Conf: %.5f' %
                    (class_names[cls_id], cls_conf.item()))
            box = yolobox2label([y1, x1, y2, x2], info_img)
            bboxes.append(box)
            classes.append(cls_id)
            colors.append(class_colors[int(cls_pred)])
        
        if args.verbose:
            print()
    
        vis_bbox(
            img_raw, bboxes, label=classes, label_names=class_names,
            instance_colors=colors, linewidth=2)
        
        basename, _ = os.path.splitext(filename)
        out_path = os.path.join(out_dir, '{}.png'.format(basename))
        plt.savefig(out_path, bbox_inches=0, pad_inches=0, dpi=100)
        plt.close()
    
    print("Done!")


if __name__ == '__main__':
    main()
