import torch
# from dataset import get_dataloader
from tqdm import tqdm
from ddpm_process import DDPM

class DPMSOLVER(DDPM):       # DDIM继承DDPM类
    def __init__(self, 
                 device = 'cuda',
                 ddpm_T = 1000,     # 训练DDPM时的步数，不是之后DDIM采样步数
                 min_beta = 0.0001,
                 max_beta = 0.02):
        super().__init__(device, ddpm_T, min_beta, max_beta)    #因为继承类了嘛，就过一下父类的初始化
    
    def sample_backward(self, net, eps_T, n_step=20, eta=1, save_flag = True):
        print("change ddpm_sample_backward to dpmsolver_sample_backward")
        return self.dpmsolver_sample_backward(net, eps_T, n_step, eta, save_flag)
    # 别的都不用动，就改一个逆向过程采样就行。

    def dpmsolver_sample_backward(self, net, eps_T, n_step, eta, save_flag = False):   
        x_t = eps_T                                 
        if(save_flag):
            # ts = torch.linspace(0,self.ddpm_T-1,10).to(torch.int)  # index需要整型
            ts_i = n_step
            img_save = []
        for i in tqdm(reversed(range(n_step))):      #1 ddim首先需要修改时间序列，现在不再是循环时间，而是循环下角标了，就改成i了
            x_t = self.sample_backward_step2(x_t, i, n_step, net)
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
    
    def sample_backward_step2(self, x_t, i, n_step, net, k = 2):   #2 ddim的每步采样
        device = self.device
        alpha_bars = self.alpha_bars
        if k == 1:
            ts = torch.linspace(0, self.ddpm_T ,
                                (n_step+1)).to(self.device).to(torch.long)
            t1 = ts[i] - 1
            t2 = ts[i + 1] - 1
            # print("t1=",t1.item(),"t2=",t2.item())
            # 根据原论文的符号
            ab_t1 = torch.sqrt(alpha_bars[t1]) if t1 >= 0 else torch.tensor(1-(1e-5)) # t1 代表小的时间(在这里代表t的下角标更大的数),-1的时候其实是计算alpha_0用的，所以置1就行
            ab_t2 = torch.sqrt(alpha_bars[t2])                   # t2 代表大的时间  (下角标更小的数)  
            sigma_t1 = torch.sqrt(1-alpha_bars[t1]) if t1 >= 0 else torch.tensor(1e-5)
            sigma_t2 = torch.sqrt(1-alpha_bars[t2])
            lambda_t1 = torch.log(ab_t1/sigma_t1)
            lambda_t2 = torch.log(ab_t2/sigma_t2)
            h_1 = lambda_t1 - lambda_t2
            t_tensor_2 = torch.full((x_t.shape[0],),t2).to(torch.long).to(device)

            x_t=(ab_t1/ab_t2)*x_t - sigma_t1*(torch.exp(h_1)-1) * net(x_t,t_tensor_2)
            
        elif k == 2:
            lambda_0 = torch.log(torch.sqrt(alpha_bars[0])/torch.sqrt(1-alpha_bars[0]))
            lambda_T = torch.log(torch.sqrt(alpha_bars[self.ddpm_T-1])/torch.sqrt(1-alpha_bars[self.ddpm_T-1]))
            lambda_t1 = (lambda_T-lambda_0)/n_step*(i) + lambda_0
            lambda_t2 = (lambda_T-lambda_0)/n_step*(i+1) + lambda_0
            h_1 = lambda_t1 - lambda_t2

            t1 = self.lambda2t(lambda_t1).to(device) 
            t2 = self.lambda2t(lambda_t2).to(device) 
            if t2>999: t2 = 999
            ab_t1 = torch.sqrt(alpha_bars[t1]) if t1 >= 0 else torch.tensor(1-(1e-5)) # t1 代表小的时间(在这里代表t的下角标更大的数),-1的时候其实是计算alpha_0用的，所以置1就行
            ab_t2 = torch.sqrt(alpha_bars[t2])                   # t2 代表大的时间  (下角标更小的数)  
            sigma_t1 = torch.sqrt(1-alpha_bars[t1]) if t1 >= 0 else torch.tensor(1e-5)
            sigma_t2 = torch.sqrt(1-alpha_bars[t2])
            lambda_mid = (lambda_t1+lambda_t2)/2
            # lambda_mid = lambda_t1

            s_1 = self.lambda2t(lambda_mid).to(device)
            ab_s1 = torch.sqrt(alpha_bars[s_1])
            sigma_s1 = torch.sqrt(1-alpha_bars[s_1])

            t_tensor_2 = torch.full((x_t.shape[0],),t2).to(torch.long).to(device)
            u_1 = (ab_s1/ab_t2)*x_t - sigma_s1*(torch.exp(h_1/2)-1) * net(x_t,t_tensor_2)
            t_tensor_s1 = torch.full((x_t.shape[0],),s_1).to(torch.long).to(device)
            x_t = (ab_t1/ab_t2)*x_t - sigma_t1*(torch.exp(h_1)-1) * net(u_1,t_tensor_s1)
        
        return x_t


    def lambda2t(self, lambda_t):

        beta_0 = 0.1
        beta_1 = 20
        part1 = torch.log(torch.exp(-2*lambda_t)+1)
        s_1 = 2*part1/  \
                                ( torch.sqrt(beta_0**2+2*(beta_1-beta_0)*part1) + beta_0)
        # s_1 = 1/(beta_1-beta_0)*(torch.sqrt(beta_0**2+2*(beta_1-beta_0)*part1)-beta_0)

        s_1 = torch.round(1000*s_1).int()
        # print(s_1)
        return s_1
    