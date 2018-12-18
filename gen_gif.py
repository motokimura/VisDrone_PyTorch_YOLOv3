import argparse
import os
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', '-i', type=str,
                        help='VisDrone sequence images')
    parser.add_argument('--out_dir', '-o', type=str, default='det_results',
                        help='Output path')
    parser.add_argument('--duration', '-d', type=int, default=100, 
                        help='Duration per frame')
    parser.add_argument('--optim', action='store_true',  
                        help='Compress output gif')
    return parser.parse_args()


def main():
    args = parse_args()

    files = os.listdir(args.in_dir)
    files.sort()

    im_list = []
    for file in files:
        path = os.path.join(args.in_dir, file)
        im = Image.open(path)
        im_list.append(im)
    
    dir_name = os.path.basename(os.path.dirname(args.in_dir + '/'))
    out_path = os.path.join(args.out_dir, '{}.gif'.format(dir_name))

    im_list[0].save(out_path, save_all=True, append_images=im_list[1:], optimize=args.optim, duration=args.duration, loop=0)


if __name__ == '__main__':
    main()
