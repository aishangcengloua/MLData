import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter, ImageChops
from torchvision import models
from torchvision import transforms

#下载图片
def load_image(path) :
    img = Image.open(path)
    return img

#因为在图像处理过程中有归一化的操作，所以要"反归一化"
def deprocess(image, device):
    image = image * torch.tensor([0.229, 0.224, 0.225], device = device) + torch.tensor([0.485, 0.456, 0.406], device = device)
    return image

#传入输入图像，正 向传播到VGG19的指定层，然后，用梯度上升更新 输入图像的特征值。
def prod(image, feature_layers, iterations, lr, transform, device, vgg, modulelist) :
    input = transform(image).unsqueeze(0)         #对图像进行resize，转成tensor和归一化操作，要增加一个维度，表示一个样本，[1, C, H, W]
    input = input.to(device).requires_grad_(True) #对图片进行追踪计算梯度
    vgg.zero_grad()                               #梯度清零
    for i in range(iterations) :
        out = input
        for j in range(feature_layers) :          #遍历features模块的各层
            out = modulelist[j + 1](out)          #以上一层的输出特征作为下一层的输入特征
        loss = out.norm()                         #计算特征的二范数
        loss.backward()                           #反向传播计算梯度，其中图像的每个像素点都是参数

        with torch.no_grad() :
            input += lr * input.grad              #更新原始图像的像素值

    input = input.squeeze()                       #训练完成后将表示样本数的维度去除
    # 交互维度
    # input = input.transpose(0, 1)
    # input = input.transpose(1, 2)
    input = input.permute(1, 2, 0)                #维度转换，因为tensor的维度是(C, H, W)，而array是(H, W, C)
    input = np.clip(deprocess(input, device).detach().cpu().numpy(), 0, 1)#将像素值限制在(0, 1)之间
    image = Image.fromarray(np.uint8(input * 255))#将array类型的图像转成PIL类型图像，要乘以255是因为转成tensor时函数自动除以了255
    return image

#多次缩小图像，然后调用函数 prod。接着在放大结果，并与按一定比例图像混合在一起，最终得到与输入 图像相同大小的输出图像。
#octave_scale参数决定了有多少个尺度的图像, num_octaves参数决定一共有多少张图像
#octave_scale和num_octaves两个参数的选定对生成图像的影响很大。
def deep_dream_vgg(image, feature_layers, iterations, lr, transform, device, vgg, modulelist, octave_scale = 2, num_octaves = 100) :
    if num_octaves > 0 :
        image1 = image.filter(ImageFilter.GaussianBlur(2))#高斯模糊
        if (image1.size[0] / octave_scale < 1 or image1.size[1] / octave_scale < 1) :#当图像的大小小于octave_scale时图像尺度不再变化
            size = image1.size
        else :
            size = (int(image1.size[0] / octave_scale), int(image1.size[1] / octave_scale))

        image1 = image1.resize(size, Image.ANTIALIAS)#缩小图片
        image1 = deep_dream_vgg(image1, feature_layers, iterations, lr, transform, device, vgg, modulelist, octave_scale, num_octaves - 1)#递归
        size = (image.size[0], image.size[1])

        image1 = image1.resize(size, Image.ANTIALIAS)#放大图像
        image = ImageChops.blend(image, image1, 0.6) #按一定比例将图像混合在一起
        # PIL.ImageChops.blend(image1, image2, alpha)
        # out = image1 * (1.0 - alpha) + image2 * alpha
    img_result = prod(image, feature_layers, iterations, lr, transform, device, vgg, modulelist)
    img_result = img_result.resize(image.size)
    return img_result

if __name__ == '__main__':
    #对图像进行预处理
    tranform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), #将PIL类型转成tensor类型，注意再次过程中像素值已经转到了[0, 1]之间，方式是除以255
        transforms.Normalize(mean = [0.485, 0.456, 0.406], #归一化
                             std = [0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vgg = models.vgg19(pretrained = True).to(device)

    modulelist = list(vgg.features.modules())#要注意网络层转成列表元素之后，第一个元素是全部的网络层，下标从1开始迭代网络层,这也是后面是modulelist[j + 1]的原因
    night_sky = load_image('starry_night.jpg')
    # night_sky_32 = deep_dream_vgg(night_sky, 30, 6, 0.2, tranform, device, vgg, modulelist)
    plt.imshow(night_sky)
    plt.show()