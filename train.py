from __future__ import division

from utils.utils import *
from utils.cocoapi_evaluator import COCOAPIEvaluator
from utils.parse_yolo_weights import parse_yolo_weights
from models.yolov3 import *
from dataset.cocodataset import *

import os
import argparse
import yaml
import random
import math

import torch
from torch.autograd import Variable
import torch.optim as optim


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, choices=['coco', 'drone'], default='drone')
    parser.add_argument('--weights_path', type=str, default='weights/darknet53.conv.74',
                        help='darknet weights file')
    parser.add_argument('--n_cpu', type=int, default=0,
                        help='number of workers')
    parser.add_argument('--checkpoint', type=str,
                        help='pytorch checkpoint file path')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints',
                        help='directory where checkpoint files are saved')
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode where only one image is trained')
    parser.add_argument('--tboard', type=str, default='log_00', 
                        help='tensorboard path for logging')
    return parser.parse_args()


def main():
    """
    YOLOv3 trainer. See README for details.
    """
    args = parse_args()
    print("Setting Arguments.. : ", args)

    cuda = torch.cuda.is_available() and args.use_cuda
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Use `file_name` key to load images when train YOLO with VisDrone data
    print("------------------------------------")
    print("  use {} dataset for training.      ".format(args.data))
    print("------------------------------------")

    assert args.data in ['coco', 'drone']
    if args.data == 'coco':
        data_dir='./COCO'
        cfg_path = 'config/yolov3_default.cfg'
        use_filename_key = False
    if args.data == 'drone':
        data_dir='./VisDrone'
        cfg_path = 'config/yolov3_visdrone_default.cfg'
        use_filename_key = True

    # Parse config settings
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f)

    print("successfully loaded config file: ", cfg)

    lr = cfg['TRAIN']['LR']
    momentum = cfg['TRAIN']['MOMENTUM']
    decay = cfg['TRAIN']['DECAY']
    burn_in = cfg['TRAIN']['BURN_IN']
    iter_size = cfg['TRAIN']['MAXITER']
    steps = eval(cfg['TRAIN']['STEPS'])
    batch_size = cfg['TRAIN']['BATCHSIZE']
    subdivision = cfg['TRAIN']['SUBDIVISION']
    ignore_thre = cfg['TRAIN']['IGNORETHRE']
    random_resize = cfg['TRAIN']['RANDRESIZE']

    print('effective_batch_size = batch_size * iter_size = %d * %d' %
          (batch_size, subdivision))
    
    # Load data
    print("loading dataset...")
    imgsize = cfg['TRAIN']['IMGSIZE']
    dataset = COCODataset(model_type=cfg['MODEL']['TYPE'],
                  data_dir=data_dir,
                  img_size=imgsize,
                  min_size=cfg['TRAIN']['MINSIZE'], 
                  max_labels=cfg['TRAIN']['MAXOBJECTS'],
                  use_filename_key=use_filename_key,
                  debug=args.debug)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=args.n_cpu)
    dataiterator = iter(dataloader)

    evaluator = COCOAPIEvaluator(model_type=cfg['MODEL']['TYPE'],
                    data_dir=data_dir,
                    img_size=cfg['TEST']['IMGSIZE'],
                    confthre=cfg['TEST']['CONFTHRE'],
                    nmsthre=cfg['TEST']['NMSTHRE'],
                    min_size=cfg['TEST']['MINSIZE'],
                    max_labels=cfg['TEST']['MAXOBJECTS'],
                    use_filename_key=use_filename_key)
    
    # Learning rate setup
    base_lr = lr

    # Initiate model
    n_classes = len(dataset.class_ids)
    model = YOLOv3(n_classes=n_classes, ignore_thre=ignore_thre)

    if args.weights_path:
        print("loading darknet weights....", args.weights_path)
        parse_yolo_weights(model, args.weights_path)
    elif args.checkpoint:
        print("loading pytorch ckpt...", args.checkpoint)
        model.load_state_dict(torch.load(args.checkpoint))

    if cuda:
        print("using cuda") 
        model = model.cuda()

    if args.tboard:
        print("using tboard")
        from tensorboardX import SummaryWriter
        tblogger = SummaryWriter(os.path.join('logs', args.tboard))

    model.train()

    # optimizer setup
    # set weight decay only on conv.weight
    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if 'conv.weight' in key:
            params += [{'params':value, 'weight_decay':decay * batch_size * subdivision}]
        else:
            params += [{'params':value, 'weight_decay':0.0}]
    optimizer = optim.SGD(params, lr=base_lr, momentum=momentum,
                          dampening=0, weight_decay=decay * batch_size * subdivision)

    tmp_lr = base_lr

    def set_lr(tmp_lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = tmp_lr / batch_size / subdivision
    
    # start training loop
    eval_interval = cfg['TRAIN']['ITER_EVAL']
    checkpoint_interval = cfg['TRAIN']['ITER_CKPT']
    
    dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # random resizing..
    imgsize_max = imgsize
    imgsize_res = 32
    r_max = int(math.floor(imgsize_max / imgsize_res))
    r_min = int(math.ceil(imgsize_max / imgsize_res / 2.0))

    for iter_i in range(iter_size):

        # COCO evaluation
        if iter_i % eval_interval == 0 and iter_i > 0:
            ap50_95, ap50 = evaluator.evaluate(model)
            model.train()
            if args.tboard:
                tblogger.add_scalar('val/AP50', ap50, iter_i)
                tblogger.add_scalar('val/AP50_95', ap50_95, iter_i)

        # learning rate scheduling
        if iter_i < burn_in:
            tmp_lr = base_lr * pow(iter_i / burn_in, 4)
            set_lr(tmp_lr)
        elif iter_i == burn_in:
            tmp_lr = base_lr
            set_lr(tmp_lr)
        elif iter_i in steps:
            tmp_lr = tmp_lr * 0.1
            set_lr(tmp_lr)

        # subdivision loop
        optimizer.zero_grad()
        for inner_iter_i in range(subdivision):
            try:
                imgs, targets, _, _ = next(dataiterator)  # load a batch
            except StopIteration:
                dataiterator = iter(dataloader)
                imgs, targets, _, _ = next(dataiterator)  # load a batch
            imgs = Variable(imgs.type(dtype))
            targets = Variable(targets.type(dtype), requires_grad=False)
            loss = model(imgs, targets)
            loss.backward()

        optimizer.step()

        if iter_i % 10 == 0:
            # logging
            print('[Iter %d/%d] [lr %f] '
                  '[Losses: xy %f, wh %f, conf %f, cls %f, total %f, imgsize %d]'
                  % (iter_i, iter_size, tmp_lr,
                     model.loss_dict['xy'], model.loss_dict['wh'],
                     model.loss_dict['conf'], model.loss_dict['cls'], 
                     model.loss_dict['l2'], imgsize),
                  flush=True)

            if args.tboard:
                tblogger.add_scalar('train/total_loss', model.loss_dict['l2'], iter_i)

            # random resizing
            if random_resize:
                imgsize = random.randint(r_min, r_max) * imgsize_res
                dataset.img_shape = (imgsize, imgsize)
                dataset.img_size = imgsize
                dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=True, num_workers=args.n_cpu)
                dataiterator = iter(dataiterator)

        # save checkpoint
        if iter_i > 0 and (iter_i % checkpoint_interval == 0):
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir,
                       "snapshot_{}.ckpt".format(iter_i)))
    if args.tboard:
        tblogger.close()


if __name__ == '__main__':
    main()
