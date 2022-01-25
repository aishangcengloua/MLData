import cv2 as cv
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from torchvision import transforms

MAX_EPOCH = 100
lr_learning = 0.001
batch_size = 64
hidden_size = 400
z_size = 20
image_size = 784
os.makedirs('samples', exist_ok = True)
samples_dir = 'samples'

dataset=datasets.MNIST( root = 'data',
                        train = False,
                        download = True,
                        transform = transforms.ToTensor())

data_loader = DataLoader(dataset, shuffle = True, batch_size = batch_size, drop_last = True)

class VAE(nn.Module) :
    def __init__(self, image_size, hidden_size, z_size):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, z_size)
        self.fc3 = nn.Linear(hidden_size, z_size)
        self.fc4 = nn.Linear(z_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc5 = nn.Linear(hidden_size, image_size)

    def Encoder(self, x):
        h = self.relu1(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def Reparameterize(self, mu, Log_var):
        std = torch.exp(Log_var / 2)
        eps = torch.randn_like((std))
        return mu + eps * std

    def Decoder(self, z):
        h = self.relu2(self.fc4(z))
        return F.sigmoid(self.fc5(h))

    def forward(self, x):
        mu, Log_var = self.Encoder(x)
        z = self.Reparameterize(mu, Log_var)
        x_reconst = self.Decoder(z)
        return x_reconst, mu, Log_var

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(image_size, hidden_size, z_size).to(device)
optimizer = optim.Adam(model.parameters(), lr = lr_learning)

for epoch in range(MAX_EPOCH) :
    model.train()
    for i, (x, _) in enumerate(data_loader) :
        x = x.to(device)
        optimizer.zero_grad()
        x = x.view(-1, image_size)
        x_reconst, mu, Log_var = model(x)
        # 计算重构损失和KL散度
        # 重构损失
        reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
        # KL散度
        kl_div = - 0.5 * torch.sum(1 + Log_var - mu.pow(2) - Log_var.exp())
        loss = reconst_loss + kl_div
        loss.backward()
        optimizer.step()

        # if i % 10 == 0 :
        #     print(f'reconst_loss : {reconst_loss : 0.3f}, kl_div : {kl_div : 0.3f}')

    with torch.no_grad() :
        #图片生成
        z = torch.randn(batch_size, z_size).to(device)
        out = model.Decoder(z).view(-1, 1, 28, 28)
        save_image(out, os.path.join(samples_dir, f'sampled-{epoch + 1}.png'))
        #图片重塑
        out, _, _ = model(x)
        print(x.shape)
        x_concat = torch.cat([x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim = 3)
        save_image(x_concat, os.path.join(samples_dir, f'reconst-{epoch + 1}.png'))

img1 = cv.imread('samples/sampled-1.png')
img2 = cv.imread('samples/sampled-50.png')
img3 = cv.imread('samples/sampled-100.png')

img4 = cv.imread('samples/reconst-1.png')
img5 = cv.imread('samples/reconst-50.png')
img6 = cv.imread('samples/reconst-100.png')

images = [img1, img2, img3, img4, img5, img6]
xlabels = ['images sample epoch : 1', 'epoch : 50', 'epoch : 100', 'images reconst epoch : 1', 'epoch : 50', 'epoch : 100']
plt.figure(figsize = (15, 10))
for i in range(6) :
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], 'gray')
    plt.axis('off')
    plt.title(xlabels[i])
    plt.tight_layout()
plt.show()