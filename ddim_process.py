import torch
# from dataset import get_dataloader
from tqdm import tqdm
from ddpm_process import DDPM

class DDIM(DDPM):       # DDIM继承DDPM类
    def __init__(self, 
                 device = 'cuda',
                 ddpm_T = 1000,     # 训练DDPM时的步数，不是之后DDIM采样步数
                 min_beta = 0.0001,
                 max_beta = 0.02):
        super().__init__(device, ddpm_T, min_beta, max_beta)    #因为继承类了嘛，就过一下父类的初始化
    
    def sample_backward(self, net, eps_T, n_step=40, eta=1, save_flag = True):
        print("change ddpm_sample_backward to ddim_sample_backward")
        return self.ddim_sample_backward(net, eps_T, n_step, eta, save_flag)
    # 别的都不用动，就改一个逆向过程采样就行。

    def ddim_sample_backward(self, net, eps_T, n_step, eta, save_flag = False):   
        x_t = eps_T                                 
        ts = torch.linspace(0, self.ddpm_T ,
                            (n_step+1)).to(self.device).to(torch.long)
        if(save_flag):
            # ts = torch.linspace(0,self.ddpm_T-1,10).to(torch.int)  # index需要整型
            ts_i = n_step
            img_save = []
        for i in tqdm(reversed(range(n_step))):      #1 ddim首先需要修改时间序列，现在不再是循环时间，而是循环下角标了，就改成i了
            t1 = ts[i] - 1
            t2 = ts[i + 1] - 1
            print("t1=",t1.item(),"t2=",t2.item())
            x_t = self.sample_backward_step2(x_t, t1, t2, net)
            if(save_flag and (i <= ts_i)):
                img_save.append(x_t)
                ts_i = ts_i-int(n_step/10)
        if(save_flag):
            import cv2
            import einops
            import numpy as np
            img_save.append(x_t)
            img_save = einops.rearrange(img_save, 'n1 n2 c h w -> (n2 h) (n1 w) c' )
            img_save = (img_save.clip(-1, 1) + 1) / 2 * 255
            img_save = img_save.cpu().numpy().astype(np.uint8)
            if(img_save.shape[2]==3):
                img_save = cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR)
            cv2.imwrite('diffusion_backward.jpg', img_save)
        return x_t
    
    def sample_backward_step2(self, x_t, t1, t2, net):   #2 ddim的每步采样
        device = self.device
        # alphas = self.alphas
        alpha_bars = self.alpha_bars
        # betas = self.betas
        ab_1 = alpha_bars[t1] if t1 >= 0 else 1 # t1 代表小的时间,-1的时候其实是计算alpha_0用的，所以置1就行
        if t1 < 0: t1 = 0 
        ab_2 = alpha_bars[t2]                   # t2 代表大的时间    

        eps = torch.randn_like(x_t).to(device)                             
        # sigma_ddpm = (1-alpha_bars[t-1])/(1-alpha_bars[t])*betas[t]     # 观察DDPM方差，可以看出就是把一些改变时间后会被影响的符号(beta)给代入了一下
        # sigma_t2 = (1-ab_1)/(1-ab_2)*(1 - ab_2/ab_1)    
        # sigma_ddpm = betas[t]                                           # 这个简化var其实就是DDIM论文提出的，但是由于后面根号计算会变负所以不能直接换
        # noise = torch.sqrt((1 - ab_2/ab_1))*eps                         # DDIM论文中是只换了noise这部分的sigma，后面根号保持不变
        sigma_t2 = torch.tensor(0).to(device)                             # 最后一种，置0时的计算，也就是没有随机性的确定性计算。
        noise = torch.sqrt(sigma_t2)*eps


        t_tensor = torch.full((x_t.shape[0],),t2).to(torch.long).to(device)            
        eps_theta = net(x_t,t_tensor)    
        # mean = 1/torch.sqrt(alphas[t]) * (x_t -
        #                                     (1-alphas[t])/torch.sqrt(1-alpha_bars[t]) * eps_theta )                                
        mean = torch.sqrt(ab_1/ab_2)*x_t + \
                                            eps_theta*( torch.sqrt(1 - ab_1 - sigma_t2) - torch.sqrt( ab_1*(1-ab_2)/ab_2))
        x_t = mean + noise    

        return x_t
    