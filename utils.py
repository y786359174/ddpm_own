import os
import torch
import torch.nn as nn
from dataset import get_img_shape
import cv2
import einops
import numpy as np

sample_i = 0

def sample(ddpm, net, device):
    global sample_i
    sample_i+=1
    
    i_n = 5
    # for i in range(i_n*i_n):
    net = net.to(device)
    net = net.eval()
    with torch.no_grad():
        eps_T = torch.randn((i_n * i_n,) + get_img_shape()).to(device)
        x_new = ddpm.sample_backward(net, eps_T,save_flag = True)
        x_new = einops.rearrange(x_new, '(n1 n2) c h w -> (n2 h) (n1 w) c', n1 = i_n)
        
        x_new = (x_new.clip(-1, 1) + 1) / 2 * 255
        x_new = x_new.cpu().numpy().astype(np.uint8)
        # print(x_new.shape)
        if(x_new.shape[2]==3):
            x_new = cv2.cvtColor(x_new, cv2.COLOR_RGB2BGR)
        cv2.imwrite('diffusion_sample_%d.jpg' %(sample_i), x_new)