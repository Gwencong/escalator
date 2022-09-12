# Script for training custom data with PaddleSeg

## 1. 准备数据集
将数据组织为如下目录形式
```
...
data
  |--images
  |    |--image1.jpg
  |    |--image2.jpg
  |    |--...
  |
  |--annotations
  |    |--image1.json
  |    |--image2.json
  |    |--...
...
```
运行脚本 **escalator/scripts/prepare.sh** ，将会根据图像标注的json文件生成对应的label图像以及可视化的label图像。
```bash
bash escalator/scripts/prepare.sh
```
之后运行脚本 **escalator/scripts/split.sh** 划分训练集、验证集和测试集，划分比例可在脚本中更改
```bash
bash escalator/scripts/split.sh
```
运行完上述命令后，data目录如下
```
...
data
  |--images             # 原图像
  |    |--image1.jpg
  |    |--image2.jpg
  |    |--...
  |
  |--annotations        # 标注文件
  |    |--image1.json
  |    |--image2.json
  |    |--...
  |
  |--labels             # label图像
  |    |--image1.png
  |    |--image2.png
  |    |--...
  |
  |--labels_color       # 伪彩色label图像，可视化
  |    |--image1.png
  |    |--image2.png
  |    |--...
  |
  |--train.txt
  |--val.txt
  |--test.txt
...
```

## 2. 训练
```bash
bash escalator/scripts/train.sh
```
## 3. 测试
```bash
bash escalator/scripts/eval.sh
```
## 4. 预测
```bash
bash escalator/scripts/predict.sh
```
## 5. 导出onnx
```bash
bash escalator/scripts/export.sh
```