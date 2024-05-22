import os
import time
import torch
import torch.nn as nn
from dataset import get_dataloader, get_img_shape
from ddpm_process import DDPM
from ddim_process import DDIM
from dpmsolver_process import DPMSOLVER
from network2 import build_unet
from utils import sample
import cv2
import einops
import numpy as np


if __name__ == '__main__':

    ckpt_path = './data/faces/model_unet2.pth'
    device = 'cuda:0'
    ddpm_T = 1000
    net = build_unet()
    net = net.to(device)
    dpm = DPMSOLVER(ddpm_T = ddpm_T, device=device)       # 前向和后向网络

    net.load_state_dict(torch.load(ckpt_path))
    sample(dpm, net, device)

