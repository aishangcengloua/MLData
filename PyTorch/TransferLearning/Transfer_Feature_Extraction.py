import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from datetime import datetime

trans_train = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                         std = [0.229, 0.224, 0.225])])

trans_vaild = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                         std = [0.229, 0.224, 0.225])])

trainset = torchvision.datasets.CIFAR10(
    root = 'data',
    download = False,
    train = True,
    transform = trans_train
)
trainloader = DataLoader(trainset, batch_size = 64, shuffle = True)

testset = torchvision.datasets.CIFAR10(
    root = 'data',
    download = False,
    train = False,
    transform = trans_vaild
)
testloader = DataLoader(testset, batch_size = 64, shuffle = False)

calsses = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

net = models.resnet18(pretrained = True)
#冻结参数
for param in net.parameters() :
    param.requires_grad = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#修改最后的全连接层
net.fc = nn.Linear(512, 10)

#查看总参数及训练参数
total_params = sum(p.numel() for p in net.parameters())
print(f'原参数的数量 : {total_params}') #11181642
total_params_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f'需要训练的参数 : {total_params_trainable}') #5130

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.fc.parameters(), lr = 0.001, weight_decay = 0.001, momentum = 0.9)

net = net.to(device)
for epoch in range(20) :
    prev_time = datetime.now()
    train_losses = 0.0
    train_acc = 0.0
    net.train()
    for x, label in trainloader :
        x, label = x.to(device), label.to(device)
        optimizer.zero_grad()
        out = net(x)
        loss = criterion(out, label)
        train_losses += loss.item()
        _, pred = torch.max(out, dim = 1)
        num_correct = (pred == label).sum().item()
        train_acc += num_correct / x.size(0)
        loss.backward()
        optimizer.step()

    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = "Time %02d:%02d:%02d" % (h, m, s)
    with torch.no_grad() :
        net.eval()
        test_losses = 0.0
        test_acc = 0.0
        for x, label in testloader :
            x, label = x.to(device), label.to(device)
            out = net(x)
            loss = criterion(out, label)
            test_losses += loss.item()
            _, pred = torch.max(out, dim = 1)
            num_correct = (pred == label).sum().item()
            test_acc += num_correct / x.size(0)

    print(f'Eopch {epoch}. Train Loss: {train_losses / len(trainloader)}, '
          f'Train Acc: {train_acc / len(trainloader)}, '
          f'Vaild Loss: {test_losses / len(testloader)}, '
          f'Vaild Acc: {test_acc / len(testloader)}, '
          f'Time: {time_str}')