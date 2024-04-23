import os
import time
import torch
import torch.nn as nn
from dataset import get_dataloader, get_img_shape
import network
from ddpm_process import DDPM
from network import (build_network, convnet_big_cfg,
                                  convnet_medium_cfg, convnet_small_cfg,
                                  unet_1_cfg, unet_res_cfg)

import cv2
import einops
import numpy as np

def train(ddpm, net, ckpt_path, device):
    
    n_epochs = 100      
    batch_size = 512
    lr = 1e-3
    ddpm_T = ddpm.ddpm_T    
    dataloader = get_dataloader(batch_size)     
    loss_fn = nn.MSELoss()                                  # nn里有自带的loss function
    optimizer = torch.optim.Adam(net.parameters(), lr)      # 优化器，用于优化神经网络模型中的可学习参数，简单来说就是自动把梯度减回去


    for epoch_i in range(n_epochs):
        tic = time.time()
        for x,_ in dataloader:                              # 这里x充当每次的输入x0
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
    torch.save(net.state_dict(), ckpt_path)

def sample(ddpm, net, device):
    
    i_n = 5
    # for i in range(i_n*i_n):
    net = net.to(device)
    net = net.eval()
    with torch.no_grad():
        eps_T = torch.randn((i_n * i_n,) + get_img_shape()).to(device)
        x_new = ddpm.ddpm_sample_backward(net, eps_T,save_flag = True)

        x_new = einops.rearrange(x_new, '(n1 n2) c h w -> (n2 h) (n1 w) c', n1 = i_n)
        x_new = (x_new.clip(-1, 1) + 1) / 2 * 255
        x_new = x_new.cpu().numpy().astype(np.uint8)
        # print(x_new.shape)
        cv2.imwrite('diffusion_sample.jpg', x_new)

if __name__ == '__main__':

    ckpt_path = 'model_unet_res.pth'
    device = 'cpu'
    ddpm_T = 1000
    net = build_network(unet_res_cfg, ddpm_T)
    net = net.to(device)
    ddpm = DDPM(ddpm_T = ddpm_T, device=device)       # 前向和后向网络

    train(ddpm, net, ckpt_path, device=device)
    net.load_state_dict(torch.load(ckpt_path))
    sample(ddpm, net, device)

