import torch
import glob
import os
import csv
import cv2 as cv
import numpy as np
import random
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms

# 因为图片大小不一，所以要对图片进行transform
transform = transforms.Compose([transforms.Resize(256), # 重塑大小至256
                                transforms.RandomCrop(244),# 裁剪至244
                                transforms.RandomHorizontalFlip(), # 随机水平翻转
                                transforms.ToTensor(), # 转成张量
                                transforms.Normalize([0.5], [0.5])]) # 标准化

class CatVsDog(Dataset) :
    def __init__(self, filename, mode = 'train', transform = None):
        super(CatVsDog, self).__init__()
        # 读取图片及其标签
        with open(os.path.join(filename)) as file:
            reader = csv.reader(file)
            images, labels = [], []
            for img, label in reader:
                images.append(img)
                labels.append(int(label))
        # 数据划分，90% 作为训练集，10% 作为验证集
        if mode == 'train' :
            images = images[ : int(len(images) * 0.9)]
            labels = labels[ : int(len(labels) * 0.9)]

        if mode == 'val' :
            images = images[int(len(images) * 0.9) : ]
            labels = labels[int(len(labels) * 0.9) : ]

        self.images, self.labels = images, labels
        self.transform = transform

    def __getitem__(self, item):
        # 读取图片
        image = Image.open(os.path.join(self.images[item]))
        # 转换
        if self.transform :
            image = self.transform(image)
        label = torch.from_numpy(np.array(self.labels[item]))
        return image, label

    def __len__(self):
        return len(self.images)

if __name__ == '__main__' :
    train_dataset = CatVsDog('train.csv', mode = 'train', transform = transform)
    val_dataset = CatVsDog('train.csv', mode = 'val', transform = transform)
    # 返回 batch 的数据对象
    train_loader = DataLoader(train_dataset, shuffle = True, batch_size = 64)
    val_loader = DataLoader(val_dataset, shuffle = False, batch_size = 64)
    # 生成可迭代对象
    image_train, label_train = iter(train_loader).next()
    image_val, label_val = iter(val_loader).next()
    # 选择第一张图片
    image_train_sample, label_train_sample = image_train[0].squeeze(), label_train[0]
    image_val_sample, label_val_sample = image_val[0].squeeze(), label_val[0]
    # 进行轴转化，因为tensor的三通道为(C, H, W)，要转成(H, W, C)
    image_train_sample = image_train_sample.permute((1, 2, 0)).numpy()
    image_val_sample = image_val_sample.permute((1, 2, 0)).numpy()
    # 因为前面以标准差和均值都是0.5标准化了图片，所以要转回来
    image_train_sample = image_train_sample * 0.5
    image_train_sample = image_train_sample + 0.5
    image_val_sample = image_val_sample * 0.5
    image_val_sample = image_val_sample + 0.5
    # 限制像素值的大小
    image_train_sample = np.clip(image_train_sample, 0, 1)
    image_val_sample = np.clip(image_val_sample, 0, 1)
    # 显示
    plt.subplot(121)
    plt.imshow(image_train_sample)
    plt.title(label_train_sample.item())
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(image_val_sample)
    plt.title(label_val_sample.item())
    plt.axis('off'), plt.show()

# def load_data(root) :
#     name2label = {}
#     for name in sorted(os.listdir(os.path.join(root))) :
#         name = name.split(sep = '.')[0]
#         if name not in name2label.keys() :
#             name2label[name] = len(name2label.keys())
#
#     return name2label #out :　{'cat': 0, 'dog': 1}


# def create_csv(root, filename, name2label):
#     # 从 csv 文件返回 images,labels 列表
#     images = []
#     # glob.glob用于返回符合格式的路径
#     images += glob.glob(os.path.join(root, '*.png'))
#     images += glob.glob(os.path.join(root, '*.jpeg'))
#     images += glob.glob(os.path.join(root, '*.jpg'))
#     # 随机打乱照片顺序(路径顺序)
#     random.shuffle(images)
#     with open(os.path.join(filename), mode='w', newline='') as file:
#         writer = csv.writer(file)
#         for name in images:
#             # 得到照片得种类名字
#             img = name.split(os.sep)[-1][0 : 3]
#             print(img)
#             writer.writerow([name, name2label[img]])
#
#
# def load_csv(filename, name2label):
#     with open(os.path.join(filename)) as file:
#         reader = csv.reader(file)
#         images, labels = [], []
#         for img, label in reader:
#             images.append(img)
#             labels.append(int(label))
#
#     return images, labels
#
# def load_cat_dog(filename, mode = 'train') :
#     images, labels = load_csv(filename, name2label)
#     if mode == 'val' :
#         images = images[int(len(images) * 0.9) : ]
#         labels = labels[int(len(labels) * 0.9) : ]
#     else :
#         images = images[ : int(len(images) * 0.9)]
#         labels = labels[ : int(len(labels) * 0.9)]
#     print(len(images), len(labels))
#     return images, labels
#
#
# name2label = load_data('train')
# # print(name2label)
# create_csv('train', 'train.csv', name2label)
# images, labels = load_csv('train.csv', name2label)
# # load_cat_dog('train.csv', mode = 'val')