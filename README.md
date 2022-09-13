# Scripts for Json to Yolo-Pose format

## 1. 准备数据集
将数据集组织为如下目录形式：
```
...
--custom_kpts
    |--images           # 原图片
    |    |---image1.jpg
    |    |---image2.jpg
    |    |---...
    |
    |--annotations      # 标注文件
    |    |---image1.json
    |    |---image2.json
    |    |---...
...
```
## 2. 转换为Yolo-Pose格式
运行如下脚本：
```bash
python json2yolo.py --json_dir custom_kpts/annotations
```
将在custom_kpts目录下生成对应的yolo-pose格式的label文件

## 3. 划分数据集
运行如下脚本划分训练集、验证集、测试集
```
python split.py \
    --image_dir custom_kpts/images \
    --prefix_path ./images/ \
    --split 0.7 0.2 0.1

# 参数含义：
# --image_dir:      图片路径
# --prefix_path:    txt中保存的图片路径的前缀，默认 `./images/`
# --out_path:       txt文件保存路径，默认为image_dir同级目录
# --split:          训练：验证：测试 的划分比例
```

准备完数据集后，custom_kpts目录如下
```
...
--custom_kpts
    |--images           
    |    |---image1.jpg
    |    |---image2.jpg
    |    |---...
    |
    |--annotations      
    |    |---image1.json
    |    |---image2.json
    |    |---...
    |
    |--labels           # Yolo-Pose格式的label文件  
    |    |---image1.txt
    |    |---image2.txt
    |    
    |--train.txt        # 训练集文件
    |--test.txt         # 测试集文件
    |--val.txt          # 验证集文件
...
```