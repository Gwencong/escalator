import os
import argparse
import numpy as np
from pathlib import Path

def split_files(out_path, file_name, prefix_path='',train=0.9, test=0.1, validate=0.0):  # split training data
    file_name = list(filter(lambda x: len(x) > 0, file_name))
    file_name = sorted(file_name)
    i, j, k = split_indices(file_name, train=train, test=test, validate=validate)
    datasets = {'train': i, 'test': j, 'val': k}
    for key, item in datasets.items():
        if item.any():
            with open(f'{out_path}/{key}.txt', 'a',encoding='utf-8') as file:
                for i in item:
                    file.write('%s%s\n' % (prefix_path, file_name[i]))

def split_indices(x, train=0.9, test=0.1, validate=0.0, shuffle=True):  # split training data
    n = len(x)
    v = np.arange(n)
    if shuffle:
        np.random.shuffle(v)

    i = round(n * train)  # train
    j = round(n * test) + i  # test
    k = round(n * validate) + j  # validate
    return v[:i], v[i:j], v[j:k]  # return indices

if __name__ == "__main__":
    # train test split
    parser = argparse.ArgumentParser(description='Joson File to YOLO format')
    parser.add_argument('--image_dir', type=str, default='data/custom_kpts/images', help='image file directory')
    parser.add_argument('--prefix_path', type=str, default='./images/', help='path prefix in txt')
    parser.add_argument('--out_path', type=str, default=None, help='path prefix in txt')
    parser.add_argument('--split', type=float, nargs=3,default=None, help='split rate')

    args = parser.parse_args()
    print(args)
    images = os.listdir(args.image_dir)
    out_path = Path(args.image_dir).parent.__str__()
    prefix_path = args.prefix_path
    train,valid,test = args.split
    split_files(out_path,images,prefix_path,train=train,test=test,validate=valid)
