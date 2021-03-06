import torch
import torch.nn as nn
import torch.optim as optim
import os
import cv2 as cv
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

#设置超参数
MAX_EPOCH = 200
LR_RATE = 0.0002
BATCH_SIZE = 100
HIDDEN_SIZE = 256
IMAGE_SIZE = 784
Z_SIZE = 64

sample_dir = 'samples_GAN'
os.makedirs(sample_dir, exist_ok = True)

Dataset = datasets.MNIST(root = 'data',
                         download = False,
                         train = True,
                         transform = transforms.Compose([transforms.ToTensor(),
                                                         transforms.Normalize([0.5], [0.5])]))

Dataloader = DataLoader(Dataset, batch_size = BATCH_SIZE, shuffle = True, drop_last = True)

class Generator(nn.Module) :
    def __init__(self):
        super(Generator, self).__init__()
        self.G = nn.Sequential(nn.Linear(Z_SIZE, HIDDEN_SIZE),
                               nn.ReLU(),
                               nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                               nn.ReLU(),
                               nn.Linear(HIDDEN_SIZE, IMAGE_SIZE),
                               nn.Tanh())
    def forward(self, z) :
        return self.G(z)

class Discriminator(nn.Module) :
    def __init__(self) :
        super(Discriminator, self).__init__()
        self.D = nn.Sequential(nn.Linear(IMAGE_SIZE, HIDDEN_SIZE),
                               nn.LeakyReLU(0.2),
                               nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                               nn.LeakyReLU(0.2),
                               nn.Linear(HIDDEN_SIZE, 1),
                               nn.Sigmoid())

    def forward(self, x):
        return self.D(x)

#Clamp函数x限制在区间[min, max]内
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

D = Discriminator().cuda()
G = Generator().cuda()
d_optimizer = optim.Adam(D.parameters(), lr = LR_RATE)
g_optimizer = optim.Adam(G.parameters(), lr = LR_RATE)
criterion = nn.BCELoss()


for epoch in range(MAX_EPOCH) :
    for i, (images, labels) in enumerate(Dataloader) :

        images = images.reshape(BATCH_SIZE, -1).cuda()
        #真样本与生成样本的标签设置
        real_labels = torch.ones(BATCH_SIZE, 1).cuda()
        fake_labels = torch.zeros(BATCH_SIZE, 1).cuda()
        #训练判别器
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        out = D(images)
        real_score = out
        d_loss_real = criterion(out, real_labels)

        z = torch.randn(BATCH_SIZE, Z_SIZE).cuda()
        fake_images = G(z)
        out = D(fake_images)
        fake_score = out
        d_loss_fake = criterion(out, fake_labels)

        d_loss = d_loss_fake + d_loss_real

        d_loss.backward()
        d_optimizer.step()

        #训练生成器
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        z = torch.randn(BATCH_SIZE, Z_SIZE).cuda()
        fake_images = G(z)
        out = D(fake_images)
        g_loss = criterion(out, real_labels)

        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                  .format(epoch, MAX_EPOCH, i + 1, len(Dataloader), d_loss.item(), g_loss.item(),
                          real_score.mean().item(), fake_score.mean().item()))

            # 保存真图片
        if (epoch + 1) == 1:
            images = images.reshape(images.size(0), 1, 28, 28)
            save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))

            # 保存假图片
        fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
        save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch + 1)))

    # 保存模型
    torch.save(G.state_dict(), 'G.ckpt')
    torch.save(D.state_dict(), 'D.ckpt')

img1 = cv.imread('samples_GAN/fake_images-1.png')
img2 = cv.imread('samples_GAN/fake_images-100.png')
img3 = cv.imread('samples_GAN/fake_images-200.png')

imgs = [img1, img2, img3]
labels = ['Epoch : 1', 'Epoch : 100', 'Epoch : 200']

plt.figure(figsize = (10, 10))
for i in range(3) :
    plt.subplot(1, 3, i + 1)
    plt.imshow(imgs[i], 'gray')
    plt.title(labels[i])
    plt.axis('off')
    plt.tight_layout()

plt.show()