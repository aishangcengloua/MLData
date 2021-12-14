import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision import models

train_set = datasets.MNIST(root = './data',
                           train = True,
                           download = True,
                           transform = transforms.Compose([transforms.ToTensor(),]) )

test_set = datasets.MNIST(root = './data',
                          train = False,
                          download = True,
                          transform = transforms.Compose([transforms.ToTensor()]))

class RNN(nn.Module) :
    def __init__(self):
        super(RNN, self).__init__()
        input_size, hidden_size, output_size = 28, 64, 10
        self.rnn = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = 1, batch_first = True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        input = input.view(-1, Seq_len, 28)
        rnn_out, hidden_state = self.rnn(input)
        # print(rnn_out.size(0))
        out = self.linear(rnn_out[ :, -1, :])
        return F.softmax(out)

Batch_size = 64
Seq_len = 28
Max_epoch = 10
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model = RNN().to(device)

train_loader = DataLoader(train_set, batch_size = Batch_size, shuffle = True, num_workers = 0)
test_loader = DataLoader(test_set, batch_size = Batch_size, shuffle = False, num_workers = 0)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

train_losses = []
train_acces = []

for epoch in range(Max_epoch) :
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    for input, label in train_loader :
        input, label = input.to(device), label.to(device)

        optimizer.zero_grad()
        outputs = model(input)
        # print(outputs.dtype, label.dtype)

        loss = criterion(outputs, label)
        train_loss += loss.item()
        loss.backward()

        _, pred = torch.max(outputs, dim = 1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / input.size(0)
        train_acc += acc

        optimizer.step()
    with torch.no_grad():
        model.eval()
        total_loss = 0.0
        test_acc = 0.0
        for input, label in test_loader:
            input, label = input.to(device), label.to(device)

            outputs = model(input)
            loss = criterion(outputs, label)
            total_loss += loss.item()

            _, pred = torch.max(outputs, dim=1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / input.size(0)
            test_acc += acc

    train_losses.append(train_loss / len(train_loader))
    train_acces.append(train_acc / len(train_loader))

    print(f'epoch : {epoch + 1}, train_loss : {train_loss / len(train_loader) : .3f}, '
          f'train_acc : {train_acc / len(train_loader) : .3f}, '
          f'test_loss : {total_loss / len(test_loader) : .3f}, '
          f'test_acc : {test_acc / len(test_loader) : .3f}')