import os
import cv2
import glob
import json
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

from utils.utils import visual_label_yolo_format

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize YOLO format label File')
    parser.add_argument('--data_root', type=str, default='data/custom_kpts', help='data directory')
    parser.add_argument('--data_file', type=str, default='train.txt', help='data list txt file')
    parser.add_argument('--visual_num', type=int, default=5, help='visualize number')
    parser.add_argument('--save_dir', type=str, default='runs/visual', help='visualize save dir')
    args = parser.parse_args()
    data_file = os.path.join(args.data_root,args.data_file)
    with open(data_file,'r',encoding='utf-8') as f:
        imgs_list = [os.path.join(args.data_root,line.strip()) for line in f.readlines()]
        imgs_list = np.array(imgs_list)
    visual_list = np.random.choice(imgs_list,args.visual_num,replace=False)
    Path(args.save_dir).mkdir(parents=True,exist_ok=True)
    visual_label_yolo_format(visual_list,args.save_dir)