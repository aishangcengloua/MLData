import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

MAX_EPOCH = 50
LR_RATE = 0.0001
BATCH_SIZE = 100

writer = SummaryWriter(log_dir = 'logs')
sample_dir = 'samples_CGAN'
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
        self.embedding = nn.Embedding(10, 10)

        self.G = nn.Sequential(nn.Linear(110, 256),
                               nn.LeakyReLU(0.2),
                               nn.Linear(256, 512),
                               nn.LeakyReLU(0.2),
                               nn.Linear(512, 1024),
                               nn.LeakyReLU(0.2),
                               nn.Linear(1024, 784),
                               nn.Tanh())
    def forward(self, z, labels) :
        y = self.embedding(labels)
        x = torch.cat([z, y], dim = 1)
        out = self.G(x)
        return out.view(z.size(0), 28, 28)

class Discriminator(nn.Module) :
    def __init__(self) :
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(10, 10)
        self.D = nn.Sequential(nn.Linear(794, 1024),
                               nn.LeakyReLU(0.2),
                               nn.Dropout(0.4),
                               nn.Linear(1024, 512),
                               nn.LeakyReLU(0.2),
                               nn.Dropout(0.4),
                               nn.Linear(512, 256),
                               nn.LeakyReLU(0.2),
                               nn.Dropout(0.4),
                               nn.Linear(256, 1),
                               nn.Sigmoid())

    def forward(self, x, labels):
        x = x.view(x.size(0), -1)
        y = self.embedding(labels)
        x = torch.cat([x, y], dim = 1)
        out = self.D(x)
        return out

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
        step = epoch * len(Dataloader) + i + 1
        images, labels = images.reshape(BATCH_SIZE, -1).cuda(), labels.cuda()
        real_labels = torch.ones(BATCH_SIZE, 1).cuda()

        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        out = D(images, labels)
        real_score = out
        d_loss_real = criterion(out, real_labels)

        z = torch.randn(BATCH_SIZE, 100).cuda()
        fake_labels = torch.randint(0, 10, (BATCH_SIZE, )).cuda()

        fake_images = G(z, fake_labels)
        out = D(fake_images, fake_labels)
        fake_score = out
        d_loss_fake = criterion(out, torch.zeros(BATCH_SIZE, 1).cuda())

        d_loss = d_loss_fake + d_loss_real

        d_loss.backward()
        d_optimizer.step()

        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        z = torch.randn(BATCH_SIZE, 100).cuda()
        fake_images = G(z, fake_labels)
        out = D(fake_images, fake_labels)
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

        # 可视化损失值
        writer.add_scalars('scalars', {'d_loss': d_loss.item(), 'g_loss': g_loss.item()}, step)
    # 保存模型
    torch.save(G.state_dict(), 'G.ckpt')
    torch.save(D.state_dict(), 'D.ckpt')
z = torch.randn(100, 100).cuda()
labels = torch.LongTensor([i for i in range(10) for _ in range(10)]).cuda()
images = G(z, labels).unsqueeze(1)
grid = make_grid(images, nrow = 10, normalize = True)
fig, ax = plt.subplots(figsize = (10, 10))
ax.imshow(grid.permute(1, 2, 0).detach().cpu().numpy(), cmap = 'binary')
ax.axis('off')
plt.show()

def generate_digit(generator, digit) :
    z = torch.randn(1, 100).cuda()
    label = torch.LongTensor([digit]).cuda()
    img = generator(z, label).detach().cpu()
    img = 0.5 * img + 0.5
    return transforms.ToPILImage()(img)
generate_digit(G, 8)

