import os
import cv2
import glob
import json
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

from utils.utils import Json2YOLO


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Joson File to YOLO format')
    parser.add_argument('--json_dir', type=str, default='data/custom', help='json file directory')
    parser.add_argument('--label_dir', type=int, default=None, help='yolo format label save directory')
    args = parser.parse_args()
    Json2YOLO(args.json_dir,args.label_dir)