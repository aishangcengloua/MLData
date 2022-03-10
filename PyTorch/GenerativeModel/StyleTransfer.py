import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision import transforms, models
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
CNN = models.vgg19(pretrained = True).features.to(device).eval()

mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).to(device)

transform = transforms.Compose([
    transforms.Resize((512, 600)),
    transforms.ToTensor()
])
untransform = transforms.ToPILImage()

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def image_loader(path) :
    image = Image.open(path)
    image = transform(image).unsqueeze(0)

    return image.to(device, torch.float)

style_img = image_loader("starry_night.jpg")
content_img = image_loader("test1.jpg")
input_img = content_img.clone()

def imshow(tensor, title) :
    image = tensor.cpu().clone().squeeze(0)
    save_image(image, 'OutImage.png')
    image = untransform(image)
    plt.imshow(image)
    plt.title(title)
    plt.pause(0.01)
    plt.show()

def gram_matrix(input) :
    B, C, H, W = input.size()
    features = input.view(B * C, H * W)
    G = torch.mm(features, features.t())
    return G.div(B * C * H * W)

class ContentLoss(nn.Module) :
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module) :
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class Normalization(nn.Module) :
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

def get_style_model_and_losses(CNN, mean, std, style_img, content_img, content_layers=content_layers_default, style_layers=style_layers_default):
    CNN =copy.deepcopy(CNN)
    normalization = Normalization(mean, std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in CNN.children() :
        if isinstance(layer, nn.Conv2d) :
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU) :
            name = f'relu_{i}'
            layer = nn.ReLU(inplace = False)
        elif isinstance(layer, nn.MaxPool2d) :
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d) :
            name = f'bn_{i}'
        else :
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)
        # print(model)
        if name in content_layers :
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f'content_loss_{i}', content_loss)
            content_losses.append(content_loss)

        if name in style_layers :
            target = model(style_img).detach()
            style_loss = StyleLoss(target)
            model.add_module(f'style_loss_{i}', style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1) :
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss) :
            break
    model = model[ : i + 1]

    return model, content_losses, style_losses

def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def run_style_transfer(CNN, mean, std, content_img, style_img, input_img, num_steps=1000, style_weight=1000000, content_weight=1):
    model, content_losses, style_losses = get_style_model_and_losses(CNN, mean, std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)
    run = [0]

    while run[0] < num_steps :
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses :
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()
            return style_score + content_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    return input_img

out = run_style_transfer(CNN, mean, std, content_img, style_img, input_img)
imshow(out, title='Output Image')