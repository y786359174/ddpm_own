import torch
from dataset import get_dataloader



class DDPM():
    def __init__(self, 
                 device = 'cuda',
                 ddpm_T = 1000,
                 min_beta = 0.0001,
                 max_beta = 0.02) :
        betas = torch.linspace(min_beta, max_beta, ddpm_T, device = device)
        alphas = 1-betas
        alpha_bars = torch.empty_like(alphas)
        product = 1
        for i, alpha in enumerate(alphas):      # 在迭代过程中同时获取索引和元素
            product *= alpha
            alpha_bars[i] = product

        self.ddpm_T = ddpm_T
        self.alpha_bars = alpha_bars.to(device)
        self.alphas = alphas.to(device)
        self.betas = betas.to(device)
        self.device = device
        pass

    def ddpm_sample_forward(self, x, t, eps):       # 没有考虑batch_size
        # alpha_bar = torch.index_select(self.alphas_bar,0,t)  # 输入index也是向量
        # alpha_bar = self.alphas_bar[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)       # 假设x是四维张量
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)                         # 假设x是四维张量
        x_t = x * torch.sqrt(alpha_bar)  + eps * torch.sqrt(1-alpha_bar) 
        return x_t
    
    def ddpm_sample_backward(self, net, eps_T, save_flag = False):      
        x_t = eps_T                                 # 此时的eps_T是xT，第一个噪声
        if(save_flag):
            ts = torch.linspace(0,self.ddpm_T-1,10).to(torch.int)  # index需要整型
            ts_i = 9
            img_save = []
        for t in reversed(range(self.ddpm_T)):      # 倒序循环实现逆扩散
            print("t=",t)
            x_t = self.sample_backward_step(x_t, t, net, simple_var=True)
            if(save_flag and (t == ts[ts_i])):
                img_save.append(x_t)
                ts_i = ts_i-1
        if(save_flag):
            import cv2
            import einops
            import numpy as np
            img_save.append(x_t)
            img_save = einops.rearrange(img_save, 'n1 n2 c h w -> (n2 h) (n1 w) c' )
            img_save = (img_save.clip(-1, 1) + 1) / 2 * 255
            img_save = img_save.cpu().numpy().astype(np.uint8)
            cv2.imwrite('diffusion_backward.jpg', img_save)
        return x_t
    
    def sample_backward_step(self, x_t, t, net):
        device = self.device
        alphas = self.alphas
        alpha_bars = self.alpha_bars
        betas = self.betas


        # xdm 血泪教训啊rand_like和randn_like千万别用错啊，rand_like是均匀分布，randn_like是标准正态
        eps = torch.randn_like(x_t).to(device)                              # 这里eps指的是逆向过程的x_t-1在重参数技巧下的方差eps部分
        if(t == 0):
            sigma_t2 = torch.tensor(0).to(device)                           # 因为计算逆向过程x_t-1的sigma2时需要用到t-1下角标，所以要分类讨论
        else:
            # sigma_t2 = (1-alpha_bars[t-1])/(1-alpha_bars[t])*betas[t]     # 计算逆向过程x_t-1时使用重参数技巧把正态分布分成均值和方差，这里在计算方差
            sigma_t2 = betas[t]                                             # 我记得是好像是哪个文章说经验来看这个结果更好
        noise = torch.sqrt(sigma_t2)*eps


        t_tensor = torch.full((x_t.shape[0],),t).to(torch.long)             # 记得这里要to(torch.long)，因为t之前是整型（我也不确定有没有影响）
        eps_theta = net(x_t,t_tensor)    
        mean = 1/torch.sqrt(alphas[t]) * (x_t -
                                            (1-alphas[t])/torch.sqrt(1-alpha_bars[t]) * eps_theta )                                
        
        x_t = mean + noise    #严格来说等号左边的x_t其实是x_t-1

        return x_t
def module_test():
    import cv2
    import einops
    import numpy as np
    device = 'cpu'
    net = None
    batch_size = 4
    ddpm = DDPM(net, device=device)       # 前向和后向网络
    ddpm_T = ddpm.ddpm_T   

    dataloader = get_dataloader(batch_size) 
    data_iter = iter(dataloader)                # 测试，只提取一次dataloader，把它放入迭代器，next读取一个
    x,_ = next(data_iter)
    x = x.to(device)
    ts = torch.linspace(0,ddpm_T-1,10).to(torch.int)  # index需要整型
    x_ts = []
    for t in ts:
        eps = torch.rand_like(x).to(device)             # 获取正态分布噪声，形状和x一致。新建数据记得todevice
        x_t = ddpm.ddpm_sample_forward(x,t,eps)
        x_ts.append(x_t)

    x_ts = einops.rearrange(x_ts, 'n1 n2 c h w -> (n2 h) (n1 w) c')
    x_ts = (x_ts.clip(-1, 1) + 1) / 2 * 255
    x_ts = x_ts.cpu().numpy().astype(np.uint8)
    if x_ts.shape[0]>8:
        x_ts = x_ts[0:7,:]
    cv2.imwrite('diffusion_forward.jpg', x_ts)

if __name__ == '__main__':
    module_test()