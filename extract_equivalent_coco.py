import argparse

from utils.utils import Extra_equivalent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Joson File to YOLO format')
    parser.add_argument('--custom_data', type=str, default='data/custom_kpts/train.txt', help='json file directory')
    parser.add_argument('--coco_data', type=str, default='data/coco_kpts/train2017.txt', help='yolo format label save directory')
    args = parser.parse_args()
    with open(args.custom_data,'r',encoding='utf-8') as f:
        img_list = f.readlines()
    num_extract = len(img_list)
    Extra_equivalent(args.coco_data,num_extract)