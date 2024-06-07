import os
import time
import torch
import torch.nn as nn
from dataset import get_dataloader, get_img_shape
# from ddpm_process import DDPM
from ddim_process import DDIM
from network2 import build_unet
# from network import build_network, unet_res_cfg
from utils import sample
from tqdm import tqdm

def train(ddpm, net, ckpt_path, device):
    
    n_epochs = 5000      
    batch_size = 256
    lr = 1e-4
    ddpm_T = ddpm.ddpm_T    
    dataloader = get_dataloader(batch_size)     
    loss_fn = nn.MSELoss()                                  # nn里有自带的loss function
    optimizer = torch.optim.Adam(net.parameters(), lr)      # 优化器，用于优化神经网络模型中的可学习参数，简单来说就是自动把梯度减回去


    for epoch_i in range(n_epochs):
        tic = time.time()
        for x,_ in tqdm(dataloader):                              # 这里x充当每次的输入x0
            x = x.to(device)
            eps = torch.randn_like(x).to(device)             # 获取正态分布噪声，形状和x一致。新建数据记得todevice
            # print(x.shape[0])
            t = torch.randint(0, ddpm_T,(x.shape[0],)).to(device)         # 从0到ddpm_T-1随机选取t
            x_t = ddpm.ddpm_sample_forward(x, t, eps)       # 因为xt和eps都是已知参数，已经在device了，所以不用todevice
            eps_theta = net(x_t,t)
            loss = loss_fn(eps, eps_theta)
            optimizer.zero_grad()                           # 设置0梯度
            loss.backward()                                 # 开始后向传播，计算出的梯度会累加在optimizer中
            optimizer.step()                                # 根据梯度更新参数
        toc = time.time()
        print(f'epoch {epoch_i} loss: {loss.item()} time: {(toc - tic):.2f}s')
        if(epoch_i%60==0):
            sample(ddpm, net, device)
        torch.save(net.state_dict(), ckpt_path)


if __name__ == '__main__':

    # ckpt_path = './model_unet2.pth'
    ckpt_path = './data/number/model_unet_res.pth'
    device = 'cuda:0'
    ddpm_T = 1000
    # net = build_network(unet_res_cfg, ddpm_T)
    net = build_unet()
    net = net.to(device)
    ddpm = DDIM(ddpm_T = ddpm_T, device=device)       # 前向和后向网络

    # net.load_state_dict(torch.load(ckpt_path))
    # train(ddpm, net, ckpt_path, device=device)
    net.load_state_dict(torch.load(ckpt_path))
    sample(ddpm, net, device)
