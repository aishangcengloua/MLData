import os
import torch
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from torchvision import transforms

# MAX_EPOCH = 100
# lr_learning = 0.001
# batch_size = 64
# image_size = 784
# os.makedirs('samples_AE', exist_ok = True)
# samples_dir = 'samples_AE'
#
# dataset = datasets.FashionMNIST(root = 'data2',
#                         train = False,
#                         download = True,
#                         transform = transforms.ToTensor())
#
# data_loader = DataLoader(dataset, shuffle = True, batch_size = batch_size, drop_last = True)
#
# class AE(nn.Module) :
#     def __init__(self):
#         super(AE, self).__init__()
#         self.fc1 = nn.Linear(784, 256)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(256, 128)
#         self.relu2 = nn.ReLU()
#         self.fc3 = nn.Linear(128, 20)
#         self.fc4 = nn.Linear(20, 128)
#         self.relu3 = nn.ReLU()
#         self.fc5 = nn.Linear(128, 256)
#         self.relu4 = nn.ReLU()
#         self.fc6 = nn.Linear(256, 784)
#
#     def Encoder(self, x):
#         h1 = self.relu1(self.fc1(x))
#         h2 = self.relu2(self.fc2(h1))
#         return self.fc3(h2)
#
#     def Decoder(self, z):
#         h1 = self.relu3(self.fc4(z))
#         h2 = self.relu4(self.fc5(h1))
#         return F.sigmoid(self.fc6(h2))
#
#     def forward(self, x):
#         z = self.Encoder(x)
#         # print(z.shape)
#         x_reconst = self.Decoder(z)
#         return x_reconst
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = AE().to(device)
# optimizer = optim.Adam(model.parameters(), lr = lr_learning)
#
# for epoch in range(MAX_EPOCH) :
#     model.train()
#     for i, (x, _) in enumerate(data_loader) :
#         x = x.to(device)
#         optimizer.zero_grad()
#         x = x.view(-1, image_size)
#         x_reconst = model(x)
#         # 重构损失，使用二元分类损失
#         reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
#
#         reconst_loss.backward()
#         optimizer.step()
#
#     with torch.no_grad() :
#         out = model(x)
#         #将与原图与重塑图像进行拼接，奇数列为原图像，偶数列为重塑图像
#         x_concat = torch.cat([x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim = 3)
#         save_image(x_concat, os.path.join(samples_dir, f'reconst-{epoch + 1}.png'))

img1 = cv.imread('samples_AE/reconst-1.png')
img2 = cv.imread('samples_AE/reconst-50.png')
img3 = cv.imread('samples_AE/reconst-100.png')

images = [img1, img2, img3]
xlabels = ['epoch : 1', 'epoch : 50', 'epoch : 100']
for i in range(3) :
    plt.subplot(1, 3, i + 1)
    plt.imshow(images[i], 'gray')
    plt.axis('off')
    plt.title(xlabels[i])
    plt.tight_layout()
plt.show()