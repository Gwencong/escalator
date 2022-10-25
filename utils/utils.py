from genericpath import isfile
import os
import cv2
import json
import glob
import shutil
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy

info = dict(description="COCO 2017 Dataset",
            url="http://cocodataset.org",
            version="1.0",
            year=2017,
            contributor="COCO Consortium",
            date_created= "2022/08/31")
licenses = [{}]
images = []
annotations = []
categories = [
    {
        "supercategory": "person",
        "id": 1,
        "name": "person",
        "keypoints": [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle"
        ],
        "skeleton": [
        [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],
        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]
        ]
    }
]


def plot_one_box(x, im, color=None, label=None, line_thickness=3, kpt_label=False, kpts=None, steps=2, orig_shape=None):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl*1//3, lineType=cv2.LINE_AA)
    if label:
        if len(label.split(' ')) > 1:
            label = label.split(' ')[-1]
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 6, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 6, [225, 255, 255], thickness=tf//2, lineType=cv2.LINE_AA)
    if kpt_label:
        plot_skeleton_kpts(im, kpts, steps, orig_shape=orig_shape)


def plot_skeleton_kpts(im, kpts, steps, orig_shape=None):
    #Plot the skeleton and keypointsfor coco datatset
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    num_kpts = len(kpts) // steps

    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf < 0.5:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)

    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        if steps == 3:
            conf1 = kpts[(sk[0]-1)*steps+2]
            conf2 = kpts[(sk[1]-1)*steps+2]
            if conf1<0.5 or conf2<0.5:
                continue
        if pos1[0]%640 == 0 or pos1[1]%640==0 or pos1[0]<0 or pos1[1]<0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0]<0 or pos2[1]<0:
            continue
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)

def visul_json(img_file,anno_file=None,color=None):
    if anno_file is None:
        anno_file = img_file.replace('images','annotations').replace('.jpg','.json')
        assert os.path.exists(anno_file),f'file not found: `{anno_file}`'
        assert os.path.exists(img_file),f'file not found: `{img_file}`'
    img = cv2.imdecode(np.fromfile(img_file,dtype=np.uint8),-1)
    with open(anno_file,'r',encoding='utf-8') as f:
        data = json.load(f)
        for person in data:
            kpts = [float(pt) for pt in person['keypoints']]
            bbox = person['bbox']
            if len(bbox) == 0:
                bbox = get_box_from_kpts(kpts)
                color = (0,0,255)
                # bbox = [0,0,1,1]
            bbox = [bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]]
            plot_one_box(bbox,img,color,kpt_label=True,kpts=kpts)
        cv2.imshow('img',img)
        cv2.waitKey(0)
    return img

def get_box_from_kpts(kpts,supply=0.06):
    kpts = deepcopy(kpts)
    kpts = np.asarray(kpts).reshape(17,2)
    kpts = kpts[np.bitwise_and(kpts[:,0]>0 , kpts[:,1]>0),:]
    xmin = np.min(kpts[:,0])
    xmax = np.max(kpts[:,0])
    ymin = np.min(kpts[:,1])
    ymax = np.max(kpts[:,1])
    h,w = ymax-ymin,xmax-xmin
    xmin -= supply*w
    xmax += supply*w
    ymin -= supply*h*2.8
    ymax += supply*h*2
    return [xmin,ymin,xmax-xmin,ymax-ymin]

def split_files(out_path, file_name, prefix_path='',train=0.9, test=0.1, validate=0.0):  # split training data
    file_name = list(filter(lambda x: len(x) > 0, file_name))
    file_name = sorted(file_name)
    i, j, k = split_indices(file_name, train=train, test=test, validate=validate)
    datasets = {'train': i, 'test': j, 'val': k}
    for key, item in datasets.items():
        if item.any():
            with open(f'{out_path}_{key}.txt', 'a',encoding='utf-8') as file:
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

def plot_box(img,box,idx=None):
    box = [int(i) for i in box]
    pt1 = tuple(box[:2])
    pt2 = tuple(box[2:])
    cv2.rectangle(img,pt1=pt1,pt2=pt2,color=(0,255,255))
    if idx is not None:
        cv2.putText(img,str(idx),pt1,cv2.LINE_AA,0.75,(255,255,0),thickness=2)


def plot_kpts(img,kpts,color=None):
    color = (0,0,255) if color is None else color
    for kpt in kpts:
        pt = (int(kpt[0]),int(kpt[1]))
        visible = kpt[2]
        if visible==1:
            cv2.circle(img,pt,5,(0,255,0),thickness=-1)
        elif visible==2:
            cv2.circle(img,pt,5,color,thickness=-1)
        


def visual_image_yolo_format(img_dir="coco_kpts/images",anno_dir="coco_kpts/labels",mode='train2017',img_name=''):
    path_img = Path(img_dir) / mode / f'{img_name}.jpg'
    path_anno = Path(anno_dir) / f'{mode}' / f'{img_name}.txt'
    # img = cv2.imread(path_img.__str__())
    img = cv2.imdecode(np.fromfile(path_img.__str__(),dtype=np.uint8),-1)
    assert img is not None,f'open img from file {path_img.__str__()} failed.'
    h,w = img.shape[:2]
    colors = np.random.randint(0,255,(50,3)).tolist()
    with open(path_anno.__str__(),'r') as f:
        for idx,line in enumerate(f.readlines()):
            line = line.strip()
            line = line.split(' ')
            label = line[0]
            box = np.array([float(i) for i in line[1:5]])
            xyxy = box
            
            xyxy[[0,2]] = box[[0,2]]*w
            xyxy[[1,3]] = box[[1,3]]*h
            xyxy[:2] = xyxy[:2]-xyxy[2:]/2
            xyxy[2:] = xyxy[:2]+xyxy[2:]
                       
            
            kpts = np.array([float(i) for i in line[5:]]).reshape(17,3)
            kpts[:,0] = kpts[:,0] * w
            kpts[:,1] = kpts[:,1] * h

            plot_box(img,xyxy,idx)
            plot_kpts(img,kpts,colors[idx])
    cv2.imshow('test',img)
    cv2.waitKey(0)
    return img

def visual_label_yolo_format(label_files,save_dir=''):
    for label_file in tqdm(label_files,desc='Visualization'):
        path_img = Path(label_file)
        path_anno = Path(str(path_img.with_suffix('.txt')).replace('images','labels'))
        img = cv2.imdecode(np.fromfile(path_img.__str__(),dtype=np.uint8),-1)
        assert img is not None,f'open img from file {path_img.__str__()} failed.'
        h,w = img.shape[:2]
        colors = np.random.randint(0,255,(50,3)).tolist()
        with open(path_anno.__str__(),'r') as f:
            for idx,line in enumerate(f.readlines()):
                line = line.strip()
                line = line.split(' ')
                label = line[0]
                box = np.array([float(i) for i in line[1:5]])
                xyxy = box
                
                xyxy[[0,2]] = box[[0,2]]*w
                xyxy[[1,3]] = box[[1,3]]*h
                xyxy[:2] = xyxy[:2]-xyxy[2:]/2
                xyxy[2:] = xyxy[:2]+xyxy[2:]
                        
                
                kpts = np.array([float(i) for i in line[5:]]).reshape(17,3)
                kpts[:,0] = kpts[:,0] * w
                kpts[:,1] = kpts[:,1] * h

                # plot_box(img,xyxy,idx)
                # plot_kpts(img,kpts,colors[idx])
                plot_one_box(xyxy,img,label=f' person-{idx}',color=(255,0,0),kpt_label=True,kpts=kpts.reshape(-1),steps=3)
        save_file = os.path.join(save_dir,path_img.name)
        cv2.imwrite(save_file,img)
    return img


def copy_data(src_dir,dst_dir,prefix='action_',postfix=''):
    # copy img
    for src_img_file in tqdm(glob.glob(os.path.join(src_dir,'*.jpg')), desc='copy data'):
        file_name = os.path.basename(src_img_file).split('.')[0]
        new_name = f'{prefix}{file_name}{postfix}'
        src_json_file = src_img_file.replace('images','annotations').replace('.jpg','.json')
        dst_img_file = os.path.join(dst_dir,new_name+'.jpg')
        dst_json_file = os.path.join(dst_dir.replace('images','annotations'),new_name+'.json')

        shutil.copyfile(src_img_file,dst_img_file)
        shutil.copyfile(src_json_file,dst_json_file)

def check_rate(file):
    with open(file,'r',encoding='utf-8') as f:
        kpt_num = 0
        action_num = 0
        for line in f.readlines():
            line = line.strip().split('/')[-1]
            if line.startswith('action'):
                action_num += 1
            else:
                kpt_num += 1
    print('without action label: {}\nwith action label:{}\n'.format(kpt_num,action_num))


def MergeJson(json_dir,save_file='data/all.json',area_weight=0.6):
    lack_box_file = []

    new_data = dict(info=info,licenses=licenses,images=images,annotations=annotations,categories=categories)
    image_id = 0
    anno_id = 0
    if os.path.isfile(json_dir):
        dir_name = os.path.dirname(json_dir)
        json_files = []
        with open(json_dir) as f:
            for line in f.readlines():
                line = line.strip().replace('images','annotations').replace('.jpg','.json')
                json_file = os.path.join(dir_name,line)
                if os.path.exists(json_file):
                    json_files.append(json_file)
                else:
                    print(f'annotation file `{json_file}` not found')
    else:
        json_file = os.path.join(json_dir,'*.json')
        json_files = glob.glob(json_file)
    
    for path in tqdm(json_files,desc='Merge json files to one file'):
        file_id = os.path.basename(path).replace('.json','')
        image_id = file_id
        img_file = path.replace('annotations','images').replace('.json','.jpg')
        assert os.path.exists(img_file),f'file not found: `{img_file}`'
        img = cv2.imdecode(np.fromfile(img_file,dtype=np.uint8),-1)
        height,width = img.shape[:2]
        with open(path,'r',encoding='utf-8') as f:
            data = json.load(f)
            for person in data:
                kpts = [float(pt) for pt in person["keypoints"]]
                bbox = [float(coord) for coord in person["bbox"]]
                if len(bbox) == 0:
                    bbox = get_box_from_kpts(kpts)
                    lack_box_file.append(path)
                    print('\n',path)
                bbox = [round(coord,2) for coord in bbox]
                num_kpt = int(np.count_nonzero(np.array(kpts))/2)
                area = round(bbox[2]*bbox[3]*area_weight,4)
                pts = np.array(kpts).reshape(17,2)
                vis_flag = np.where(np.bitwise_and(pts[:,0:1]==0 , pts[:,1:2]==0),0,2)
                kpts = np.concatenate([pts,vis_flag],axis=-1).reshape(-1)
                kpts = np.around(kpts)
                kpts = kpts.astype(np.int0).tolist()

                img_info = {"license": 0,
                            "file_name": os.path.basename(img_file),
                            "height": height,
                            "width": width,
                            "id": image_id}
                new_data['images'].append(img_info)

                anno_info = {"segmentation":[],
                            "num_keypoints": num_kpt,
                            "area": area,
                            "iscrowd": 0,
                            "keypoints": kpts,
                            "image_id": image_id,
                            "bbox": bbox,
                            "category_id": 1,
                            "id": anno_id}
                new_data['annotations'].append(anno_info)
                anno_id += 1
        # image_id += 1
    Path(save_file).parent.mkdir(parents=True,exist_ok=True)
    with open(save_file,'w',encoding='utf-8') as f:
        json.dump(new_data,f,ensure_ascii=False,indent=2)
    print('lack box file: {}'.format(len(lack_box_file)))
    
def Json2YOLO(json_dir,label_dir=None):
    lack_box_file = []

    label_dir = json_dir.replace('annotations','labels') if label_dir is None else label_dir
    fn = Path(label_dir)  # folder name
    fn.mkdir(exist_ok=True)
    
    json_file = os.path.join(json_dir,'*.json')
    for path in tqdm(glob.glob(json_file),desc='json files to YOLO labels'):
        img_file = path.replace('annotations','images').replace('.json','.jpg')
        label_file = path.replace('annotations','labels').replace('.json','.txt')
        assert os.path.exists(img_file),f'file not found: `{img_file}`'

        img = cv2.imdecode(np.fromfile(img_file,dtype=np.uint8),-1)
        height,width = img.shape[:2]
        with open(path,'r',encoding='utf-8') as f, \
                open(label_file, 'w') as file:
            data = json.load(f)
            
            for person in data:
                kpts = np.array([float(pt) for pt in person["keypoints"]])
                bbox = np.array([float(coord) for coord in person["bbox"]])
                if len(bbox) == 0:
                    bbox = get_box_from_kpts(kpts)
                    bbox = np.array(bbox)
                    lack_box_file.append(path)
                    print('\n',path)
                bbox[:2] += bbox[2:] / 2    # [x,y,w,h]->[cx,cy,w,h]
                bbox[[0,2]] /= width
                bbox[[1,3]] /= height
                

                pts = np.array(kpts).reshape(17,2)
                vis_flag = np.where(np.bitwise_and(pts[:,0:1]==0 , pts[:,1:2]==0),0,2)
                kpts = np.concatenate([pts,vis_flag],axis=-1)
                kpts[:,0] /= width
                kpts[:,1] /= height
                kpts = kpts.reshape(-1)
                kpts = kpts.tolist()

                # Write
                if bbox[2] > 0 and bbox[3] > 0:  # if w > 0 and h > 0
                    cls = 0  # class
                    line = cls, *bbox, *kpts  # cls, box or segments
                    file.write(('%g ' * len(line)).rstrip() % line + '\n')
    print('lack box file: {}'.format(len(lack_box_file)))
    print('json to yolo completed.')

def Extra_equivalent(txtfile,extract_num):
    print(f'Extract {extract_num} files from {txtfile}...')
    with open(txtfile,'r',encoding='utf-8') as f:
        img_all_list = f.readlines()
    if extract_num>len(img_all_list):
        print(f'expect number is {extract_num}, but get {len(img_all_list)}')
        extract_num = len(img_all_list)
    img_list = img_all_list[:extract_num]

    savefile = txtfile.replace('.txt','-equiv.txt')
    with open(savefile,'w',encoding='utf-8') as f:
        for img_path in tqdm(img_list):
            f.write(img_path)
    print(f'Extracted file has been saved in {savefile}')


if __name__ == "__main__":
    json_dir = r'D:\my file\project\扶梯项目\训练数据\姿态估计\annotations'
    img_dir = json_dir.replace('annotations','images')
    label_dir = json_dir.replace('annotations','labels')
    # MergeJson(json_dir)
    # Json2YOLO(json_dir)
    # visul_json(img_file=r"D:\my file\project\扶梯项目\训练数据\姿态估计\images\00411.jpg")
    # visual_image_yolo_format(img_dir=img_dir,anno_dir=label_dir,mode='',img_name='18301')

    # copy data
    # src_dir = r"D:\my file\project\扶梯项目\训练数据\动作分类\images"
    # dst_dir = r"D:\my file\project\扶梯项目\训练数据\姿态估计\images"
    # copy_data(src_dir,dst_dir,prefix='action_')

    # train test split
    # images = os.listdir(r'D:\my file\project\扶梯项目\训练数据\姿态估计\images')
    # prefix_path = ''
    # out_path = r'D:\my file\project\扶梯项目\训练数据\姿态估计\custom'
    # split_files(out_path,images,prefix_path,train=0.7,test=0.1,validate=0.2)

    # check rate
    check_rate(r'D:\my file\project\扶梯项目\训练数据\姿态估计\custom_train.txt')
    check_rate(r'D:\my file\project\扶梯项目\训练数据\姿态估计\custom_val.txt')
    check_rate(r'D:\my file\project\扶梯项目\训练数据\姿态估计\custom_test.txt')