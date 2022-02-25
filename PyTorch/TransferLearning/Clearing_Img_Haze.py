import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
import glob
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
#展示图片
# img = cv.imread('test_images/shanghai01.jpg', 1)
# plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
# plt.show()

#定义神经网络
class model(nn.Module) :
    def __init__(self):
        super(model, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.e_conv1 = nn.Conv2d(3, 3, 1, 1, 0, bias = True)
        self.e_conv2 = nn.Conv2d(3, 3, 3, 1, 1, bias = True)
        self.e_conv3 = nn.Conv2d(6, 3, 5, 1, 2, bias = True)
        self.e_conv4 = nn.Conv2d(6, 3, 7, 1, 3, bias = True)
        self.e_conv5 = nn.Conv2d(12, 3, 3, 1, 1, bias = True)

    def forward(self, x):
        source = []
        source.append(x)

        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        concat1 = torch.cat((x1, x2), dim = 1)
        x3 = self.relu(self.e_conv3(concat1))
        concat2 = torch.cat((x2, x3), dim = 1)
        x4 = self.relu(self.e_conv4(concat2))
        concat3 = torch.cat((x1, x2, x3, x4), dim = 1)
        x5 = self.relu(self.e_conv5(concat3))
        clean_img = self.relu((x5 * x) - x5 + 1)

        return clean_img


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = model().to(device)


def cl_image(image_path):
    data = Image.open(image_path)
    data = (np.asarray(data) / 255.0)  #(H, W, C)
    data = torch.from_numpy(data).float()
    data = data.permute(2, 0, 1)       #(C, H, W)
    data = data.to(device).unsqueeze(0)

    net.load_state_dict(torch.load('dehazer.pth'))
    clean_image = net.forward(data)
    torchvision.utils.save_image(torch.cat((data, clean_image), 0), "results/" + image_path.split("\\")[-1])


if __name__ == '__main__':
    test_list = glob.glob("test_images/*")
    print(test_list)
    for image in test_list:
        cl_image(image)
        print(image, "done!")