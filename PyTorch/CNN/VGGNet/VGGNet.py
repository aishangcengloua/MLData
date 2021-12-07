import torch.nn as nn
import copy
import torch.optim as optim
import torch
import os
import cv2 as cv
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision.models import vgg

transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(244),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()])

class dataset(Dataset) :
    def __init__(self, root_path = None, model = 'train', transform = None):
        super(dataset, self).__init__()
        self.model = model
        self.root = root_path
        self.transform = transform

        image_dir = list(sorted(os.listdir(root_path)))
        images = torch.zeros(len(image_dir), 3, 244, 244)
        target = torch.zeros(len(image_dir), dtype=torch.int64)
        if model == 'train' :
            for i, image in enumerate(image_dir) :
                img = cv.imread(os.path.join(root_path, image), cv.IMREAD_COLOR)
                img = cv.resize(img, (244, 244))

                images[i, :, :, :] = self.transform(img)
                name = image_dir[i].split('.')[0]
                if name == 'cat' :
                    target[i] = 0
                else :
                    target[i] = 1
            self.images = torch.FloatTensor(images)
            self.target = target
            print(len(target))
        else :
            for i, image in enumerate(image_dir) :
                img = cv.imread(os.path.join(root_path, image), cv.IMREAD_COLOR)
                img = cv.resize(img, (244, 244))
                img = img.transpose(2, 0, 1)
                images[i, :, :, :] = torch.tensor(img)
            self.images = torch.FloatTensor(images)
            # cv.imshow('img', images[0].numpy().transpose(1, 2, 0).astype(np.uint8))
            # cv.waitKey(0)
    def __getitem__(self, item):
        if self.model == 'train' :
            return self.images[item], self.target[item]
        else :
            return self.images[item]
    def __len__(self):
        return len(self.images)

data_train = dataset(root_path='train', model = 'train', transform = transform_train)
data_test = dataset(root_path='test', model = 'test', transform = transform_test)
classes = ['cat', 'dog']

loader_train = DataLoader(data_train, batch_size = 32, shuffle = True)
loader_test = DataLoader(data_test, batch_size = 1, shuffle = False)
#制作十一层VGG模型
feature = vgg.make_layers(vgg.cfgs['A'])
model = vgg.VGG(num_classes = 2, features = feature).cuda()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()
max_epoch = 3

for epoch in range(max_epoch) :
    train_loss = 0.0
    train_acc = 0.0
    for x, label in loader_train:

        x, label = x.cuda(), label.cuda()
        optimizer.zero_grad()
        out = model(x)

        loss = criterion(out, label)
        train_loss += loss.item() / label.shape[0]
        _, pred = torch.max(out, 1)
        # if epoch == 0 :
        #     for i in range(len(pred)) :
        #         print(classes[pred[i]])
        loss.backward()
        optimizer.step()

        num_correct = (pred == label).sum().item()
        acc = num_correct / label.shape[0]
        train_acc += acc

    print(f'epoch : {epoch}, train_loss : {train_loss / len(loader_train)},'
          f'train_acc : {train_acc / len(loader_train)} ')
    if epoch == 2 :
        plt.figure()
        with torch.no_grad() :
            for i, x in enumerate(loader_test) :
                image = copy.deepcopy(x)
                x = x.cuda()
                out = model(x)
                _, pred = torch.max(out, 1)
                image = image.numpy().astype(np.uint8).squeeze().transpose(1, 2, 0)

                plt.subplot(1, len(loader_test), i + 1)
                plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
                plt.title(f'{classes[pred]}')
            plt.show()