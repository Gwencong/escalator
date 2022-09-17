import os
import cv2
import glob
import json
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

from utils.utils import MergeJson


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Joson File to YOLO format')
    parser.add_argument('--json_dir', type=str, default='data/custom', help='json file directory')
    parser.add_argument('--save_file', type=str, default='data/all.json', help='yolo format label save directory')
    args = parser.parse_args()
    MergeJson(args.json_dir,args.save_file)