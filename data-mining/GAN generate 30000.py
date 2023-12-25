import argparse
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torchvision

## 超参数配置
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)
img_area = np.prod(img_shape)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_area, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        ## 模型中间块儿
        def block(in_feat, out_feat, normalize=True):  ## block(in， out )
            layers = [nn.Linear(in_feat, out_feat)]  ## 线性变换将输入映射到out维
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))  ## 正则化
            layers.append(nn.LeakyReLU(0.2, inplace=True))  ## 非线性激活函数
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, img_area),
            nn.Tanh()
        )

    def forward(self, z):
        imgs = self.model(z)
        imgs = imgs.view(imgs.size(0), *img_shape)
        return imgs

generator = Generator()
discriminator = Discriminator()

criterion = torch.nn.BCELoss()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

## 如果有显卡，都在cuda模式中运行
if torch.cuda.is_available():
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion = criterion.cuda()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

generator1=Generator().to(device)
generator1.load_state_dict(torch.load("./save/gan/generator2.pth",map_location=device))
generator1.eval()
for i in range(1,30001):
    z = torch.randn(1, 100).to(device)
    fake_image = generator1(z)
    torchvision.utils.save_image(fake_image, "E:/GAN1/imaged/generator_image/generated%d_image.png" % i)
