import torch
import torch.nn as nn
import torch.optim as optim

from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet

from utils import *


# TODO: this is a COPY from the notebook, WE NEED TO ACTUALLY WRITE THIS
class UnetBlock(nn.Module):
    def __init__(self,filter1,filter2,submod=None,input=None,dropout=False,inermost=False,outermost=False):
        super().__init__()
        if input is None: input=filter1
        Dconv = nn.Conv2d(input,filter2,kernel_size=4,stride=2,padding=1,bias=False)
        self.outermost=outermost
        Drlu = nn.LeakyReLU(0.2,True)
        Dnorm = nn.BatchNorm2d(filter2)
        Urlu = nn.ReLU(True)
        Unorm = nn.BatchNorm2d(filter1)
        Uconv = nn.ConvTranspose2d(filter2*2, filter1, kernel_size=4, stride=2, padding=1)  # can change ker,str,padd
        self.x=0
        if(outermost):
            # define an up and down conv
            down_meathod = [Dconv]
            up_meathod = [Urlu,Uconv,nn.Tanh()]
            self.x = 1
        elif (inermost):
            Uconv = nn.ConvTranspose2d(filter2, filter1, kernel_size=4, stride=2, padding=1)
            down_meathod = [Drlu,Dconv]
            up_meathod = [Urlu, Uconv, Unorm]
        else:
            down_meathod = [Drlu,Dconv,Dnorm]
            up_meathod = [Urlu, Uconv, Unorm]
            self.x = 1
            if dropout:
                up_meathod += [nn.Dropout(0.5)]#maby change the val
        model = down_meathod
        if self.x:
            model += [submod]
        model+= up_meathod
        self.model = nn.Sequential(*model)

    def forward(self,x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x,self.model(x)],1)

class Unet(nn.Module):
    def __init__(self,input=1,output=2,hight=8,numfilters=64):
        super().__init__()
        unet_block = UnetBlock(numfilters*8,numfilters*8,inermost=True)
        for _ in range(hight - 5):
            unet_block = UnetBlock(numfilters * 8, numfilters * 8, submod=unet_block, dropout=True)
        outfilters = numfilters*8
        for _ in range(3):
            unet_block = UnetBlock(outfilters//2,outfilters,submod=unet_block)
            outfilters //= 2
        self.model = UnetBlock(output,outfilters,input=input,submod=unet_block,outermost=True)
    def forward(self,x):
        return self.model(x)


def get_generator(cin=1, cout=2, size=256):
    body = create_body(resnet18(), pretrained=True, n_in=cin, cut=-2)
    gen = DynamicUnet(body, cout, (size, size)).to(device)
    return gen


def get_discriminator(in_channel, n_filters=64, depth=3):
    l1 = [nn.Conv2d(in_channel, n_filters, 4, 2, 1, bias=True), nn.LeakyReLU(0.2, True)]
    l2 = []
    for i in range(depth):
        l2 += [nn.Conv2d(n_filters * 2**i, n_filters * 2**(i+1), 4, 1 + int(i < depth-1), 1, bias=True), nn.BatchNorm2d(n_filters * 2**(i+1)), nn.LeakyReLU(0.2, True)]
    l3 = [nn.Conv2d(n_filters * 2**depth, 1, 4, 1, 1, bias=True)]
    all = l1 + l2 + l3
    return nn.Sequential(*all)


# Final model
class ImColModel(nn.Module):
    """
    Main model with generator and discriminator
    """
    def __init__(self, lr_g, lr_d, lam, gen=None, load_pretrain_path=None):
        super().__init__()
        self.lam = lam
        if gen is None:
            self.gen = Unet().to(device)
            init_model(self.gen)
        else:
            self.gen = gen.to(device)
        self.dis = get_discriminator(3).to(device)
        init_model(self.dis)
        self.opt_g = optim.Adam(self.gen.parameters(), lr=lr_g)
        self.opt_d = optim.Adam(self.dis.parameters(), lr=lr_d)
        self.gan_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        if load_pretrain_path is not None:
            self.gen.load_state_dict(torch.load(load_pretrain_path))


    def forward(self, x):
        return self.gen(x)

    def loss_dis(self, inputs, labels, outputs):
        out_img = torch.cat([inputs, outputs], dim=1)
        out_preds = self.dis(out_img.detach())
        loss_d_fake = self.gan_loss(out_preds, torch.zeros(out_preds.shape, device=device))
        img = torch.cat([inputs, labels], dim=1)
        img_preds = self.dis(img)
        loss_d_real = self.gan_loss(img_preds, torch.ones(img_preds.shape, device=device))
        loss_d = (loss_d_fake + loss_d_real) / 2
        return loss_d

    def loss_gen(self, inputs, labels, outputs):
        out = torch.cat([inputs, outputs], dim=1)
        dis_pred = self.dis(out)
        loss_gan = self.gan_loss(dis_pred, torch.ones(dis_pred.shape, device=device))
        loss_l1 = self.l1_loss(outputs, labels)
        loss_g = loss_gan + self.lam * loss_l1
        return loss_g

    def backprop(self, inputs, labels, outputs):
        self.dis.train()
        for param in self.dis.parameters():
            param.requires_grad = True
        self.opt_d.zero_grad()
        loss_d = self.loss_dis(inputs, outputs, labels)
        loss_d.backward(retain_graph=True)
        self.opt_d.step()
        self.dis.eval()

        self.gen.train()
        for param in self.dis.parameters():
            param.requires_grad = False
        self.opt_g.zero_grad()
        loss_g = self.loss_gen(inputs, outputs, labels)
        loss_g.backward()
        self.opt_g.step()
        return loss_g.item()

    def save(self, path):
        torch.save(self.gen.state_dict(), path + 'gen.pt')
        torch.save(self.dis.state_dict(), path + 'dis.pt')

    def load(self, path):
        self.gen.load_state_dict(torch.load(path + 'gen.pt'))
        self.dis.load_state_dict(torch.load(path + 'dis.pt'))


def init_model(model, gain=0.02):
    def _init_weight(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)
    model.apply(_init_weight)
    return model


if __name__ == '__main__':
    model = ImColModel(lr_g=1e-4, lr_d=1e-4, lam=100).to(device)
    # inputs = torch.randn(16, 1, 256, 256, device=device)
    # labels = torch.randn(16, 2, 256, 256, device=device)
    # outputs = model(inputs)
    # model.backprop(inputs, labels, outputs)
    # model.save('../model/')
    img = torch.randn(1, 3, 256, 256, device=device) * 256
    result = test_img(model, img)[0, :, :, :]
    result = result.cpu().T
    plt.imshow(result)
    plt.show()
