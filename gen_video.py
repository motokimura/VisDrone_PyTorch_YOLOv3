import argparse
import os
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', '-i', type=str,
                        help='VisDrone sequence images')
    parser.add_argument('--out_dir', '-o', type=str, default='det_results',
                        help='Output path')
    parser.add_argument('--fps', '-f', type=float, default=30, 
                        help='Frame per second')
    parser.add_argument('--step', '-s', type=int, default=1, 
                        help='Step to sample input images')
    parser.add_argument('--scale', type=float, default=1.0,  
                        help='Scale of the frame size')
    return parser.parse_args()


def main():
    args = parse_args()

    files = os.listdir(args.in_dir)
    files.sort()

    # dummy read to know the original image size
    path = os.path.join(args.in_dir, files[0])
    img = cv2.imread(path)
    h_ref, w_ref, _ = img.shape
    nh, nw = int(h_ref / args.scale), int(w_ref / args.scale)

    dir_name = os.path.basename(os.path.dirname(args.in_dir + '/'))
    out_path = os.path.join(args.out_dir, '{}.mp4'.format(dir_name))
    
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v') # MP4
    video = cv2.VideoWriter(out_path, fourcc, args.fps, (nw, nh))

    for i in range(0, len(files), args.step):
        file = files[i]
        path = os.path.join(args.in_dir, file)
        img = cv2.imread(path)
        
        h, w, _ = img.shape
        assert (h == h_ref) and (w == w_ref)

        img = cv2.resize(img, (nw, nh))
        video.write(img)
    
    video.release()


if __name__ == '__main__':
    main()
