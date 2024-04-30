import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import get_img_shape

# DDPM下的Unet，先写个死板的，之后再补充带配置的

class DownSample(nn.Module):
    # 下采样

    def __init__(self, in_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, in_c, (3,3), (2,2), (1,1) ) # 输入和输出的channal数不变，wh减半
                                                                # 写成3,2,1也行,都一样的
                                                                # 卷积核3*3对应pad1可以让卷积前后图片保持不变（卷积核3会减2，pad1加2）
    def forward(self, x:torch.Tensor, t: Optional[torch.Tensor] = None):                          # 好习惯，带上类别:torch.Tensor
        return self.conv(x)

class UpSample(nn.Module):
    # 上采样

    def __init__(self, in_c):
        super().__init__()
        self.convT = nn.ConvTranspose2d(in_c, in_c, (4,4), (2,2), (1,1))    # 输入和输出的channal数不变，wh加倍
                                                                            # 二倍图片wh常用的配置，我看DCGAN也这么用
                                                                            # 哦对遇到这就在说一嘴，忘了哪个大佬说的，在卷积时用小的卷积核多次比大的效果更好，好像是实验证明的，所以现在就都用卷积33或者反卷积44了
    def forward(self, x:torch.Tensor, t: Optional[torch.Tensor] = None):                          # 好习惯，带上类别:torch.Tensor
        return self.convT(x)
    

class ResidualBlock(nn.Module):
    # 残差块，用在DownBlock和UpBlock和MiddleBlock里面的那个小的块
    # 为什么不在这输入x和t呢？因为这里是网络init，输入是在使用的时候输入。

    def __init__(self, in_c:int, out_c:int, time_c:int, n_groups: int = 32, dropout: float = 0.1):
        super().__init__()

        # 第一步要把时间和图片做变换以对齐channal
        
        self.time_act = nn.SiLU()                   # 处理前先激活，Swish激活函数
        self.time_linear = nn.Linear(time_c, out_c) # 时间用线性层处理
        

        # 图片用卷积实现，但是卷积前要norm+activ
        self.norm1 = nn.GroupNorm(n_groups, in_c)                           # 卷积的norm用GroupNorm，激活函数用Swish，后面同理
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size = 3, padding = 1)   

        # 第二步两个加和，不用新建层，加和后要再做卷积
        self.norm2 = nn.GroupNorm(n_groups, out_c)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size = 3, padding = 1)


        # 残差部分的残差链接Residual Connection
        if in_c != out_c:
            self.shortcut = nn.Conv2d(in_c, out_c, kernel_size = 1)         # 简单卷积一下，channal能对上就行，尽可能不让他变化
        else:
            self.shortcut = nn.Identity()                                   # 等于他自己的意思
        
        self.dropout = nn.Dropout(dropout)
    def forward(self, x:torch.Tensor, t:torch.Tensor):
        # 输入数据x 大小为(batch_size, in_c, height, width)
        # 输入数据t，经过embedding后的结果 大小为(batch_size, time_c)
        # 新建一个h，因为x要保留最后做残差链接

        h = self.conv1(self.act1(self.norm1(x)))
        h += self.time_linear(self.time_act(t))[:, :, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))              # dropout层在norm和act后，卷积前
        h += self.shortcut(x)

        return h 

class AttentionBlock(nn.Module):
    """
    Attention模块
    和Transformer中的multi-head attention原理及实现方式一致
    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        """
        Params:
            n_channels：等待做attention操作的特征图的channel数
            n_heads：   attention头数
            d_k：       每一个attention头处理的向量维度
            n_groups：  Group Norm超参数
        """
        super().__init__()

        # 一般而言，d_k = n_channels // n_heads，需保证n_channels能被n_heads整除
        if d_k is None:
            d_k = n_channels
        # 定义Group Norm
        self.norm = nn.GroupNorm(n_groups, n_channels)
        # Multi-head attention层: 定义输入token分别和q,k,v矩阵相乘后的结果
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # MLP层
        self.output = nn.Linear(n_heads * d_k, n_channels)
        
        self.scale = d_k ** -0.5
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        """
        Params:
            x: 输入数据xt，尺寸大小为（batch_size, in_channels, height, width）
            t: 输入数据t，尺寸大小为（batch_size, time_c）
        
        【配合图例进行阅读】
        """
        # t并没有用到，但是为了和ResidualBlock定义方式一致，这里也引入了t
        _ = t
        # 获取shape
        batch_size, n_channels, height, width = x.shape
        # 将输入数据的shape改为(batch_size, height*weight, n_channels)
        # 这三个维度分别等同于transformer输入中的(batch_size, seq_length, token_embedding)
        # (参见图例）
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # 计算输入过矩阵q,k,v的结果，self.projection通过矩阵计算，一次性把这三个结果出出来
        # 也就是qkv矩阵是三个结果的拼接
        # 其shape为：(batch_size, height*weight, n_heads, 3 * d_k)
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # 将拼接结果切开，每一个结果的shape为(batch_size, height*weight, n_heads, d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # 以下是正常计算attention score的过程，不再做说明
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attn = attn.softmax(dim=2)
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        # 将结果reshape成(batch_size, height*weight,, n_heads * d_k)
        # 复习一下：n_heads * d_k = n_channels
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # MLP层，输出结果shape为(batch_size, height*weight,, n_channels)
        res = self.output(res)

        # 残差连接
        res += x

        # 将输出结果从序列形式还原成图像形式，
        # shape为(batch_size, n_channels, height, width)
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        return res
    

class TimeEmbedding(nn.Module):
    """
    TimeEmbedding模块将把整型t，以Transformer函数式位置编码的方式，映射成向量，
    其shape为(batch_size, time_channel)
    """

    def __init__(self, n_channels: int):
        """
        Params:
            n_channels：即time_channel
        """
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        """
        Params:
            t: 维度（batch_size），整型时刻t
        """
        # 以下转换方法和Transformer的位置编码一致
        # 【强烈建议大家动手跑一遍，打印出每一个步骤的结果和尺寸，更方便理解】
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        # 输出维度(batch_size, time_channels)
        return emb
    

class DownUpBlock(nn.Module):
    # DownBlock和UpBlock层（应该是能共用的）

    def __init__(self, in_c: int, out_c: int, time_c: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_c, out_c, time_c)
        if has_attn:                                # attention 可有可无
            self.attn = AttentionBlock(out_c)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x,t)
        x = self.attn(x)
        return x

class MiddleBlock(nn.Module):
    # MiddleBlock层，其输入和输入channal一样都等于n_c

    def __init__(self, n_c: int, time_c: int):
        super().__init__()
        self.res1 = ResidualBlock(n_c, n_c, time_c)     # 因为middle层输入和输出的channal维数都是一样的，所以只需要一个n_c
        self.attn = AttentionBlock(n_c)
        self.res2 = ResidualBlock(n_c, n_c, time_c)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x
    
class UNet(nn.Module):
    # DDPM中的Unet结构

    def __init__(self, image_c: int = 3, 
                 n_chs: Union[Tuple[int, ...], List[int]]= (64, 64, 128, 256, 1024),    # 这里意思是，这个值可以是tuple(元组)或者列表(list)
                 is_attn: Union[Tuple[bool, ...], List[int]] = (False, False, False, True, True),  
                 n_blocks: int = 2 ):
        
        # image_c 原有的图片channal
        # n_chs 每一层Encoder的卷积的输出channal数都是这个
        # is_attn ,每一层的Down_UpSample是否要attention，Middle固定有atten，就没写上面
        # n_blocks 每一层sample后跟几个Down_UpSample，默认2

        super().__init__()

        layer_n = len(n_chs) # 有多少层（这里的层指的是图画的层，sample一次加一个的）

        in_channal = image_c          # 后面的网络就会有多层，所以每次都提前准备好in_c和out_c，并在过一层后更新，提前习惯一下这种写法
        out_channal = n_chs[0]          # 定义成全称而不是out_c是因为看起来好看
        self.conv_pre = nn.Conv2d(in_channal, out_channal, kernel_size=3, padding=1)
        in_channal = out_channal
        time_channal = n_chs[0] * 4
        self.time_embed = TimeEmbedding(time_channal)  # 设置提取t向量的函数，我也不知道为什么设置成第一层channal的4倍，可能是经验值把
        # Encoder部分
        down_BlockList = []
        for i in range(1, layer_n):     # 因为第0层已经跑过了，从1开始跑
            # 根据找规则，可以看出来，规律就是先做n_blocks次downblock，然后做一次downsample。最后一次箭头是空的，不做downsample
            out_channal = n_chs[i]      # 每层的输出channal
            for _ in range(n_blocks):
                down_BlockList.append(DownUpBlock(in_channal, out_channal, time_channal, is_attn[i] ))
                in_channal = out_channal
            
            if i < layer_n - 1:
                down_BlockList.append(DownSample(in_channal)) # 因为sample不变channal，所以这不用重新赋值channal

        self.down = nn.ModuleList(down_BlockList)       # 最后塞进ModuleList组成一个可运行函数

        self.middle = MiddleBlock(in_channal, time_channal) # 同样不改变channal数
        # Decoder部分
        up_BlockList = []
        for i in reversed(range(1, layer_n)):     # up是从后往前跑
            # 找规律，以n_block为默认2，每三个UpBlock一个UpSample
            # 三个UpBlock前都要cat一个Encoder的图做残差链接
            # 前两个cat的是Decoder这层的输入的大小的channal(Decoder的输入就是Encoder的输出就是n_chs[i]，最后一个cat的是Encoder上一层的输出channal
            out_channal =  n_chs[i]
            for _ in range(n_blocks):

                in_channal = in_channal + n_chs[i]  # 前两次拼接，拼的都是Encoder经过一次卷积的块（可以对应结构图看一下）
                up_BlockList.append(DownUpBlock(in_channal, out_channal, time_channal, is_attn[i] ))
                in_channal = out_channal
            
            in_channal = in_channal + n_chs[i-1] # 第三次拼接，拼的是刚sample的块，他的channal等于上一层的Encoder的输出channal
            out_channal =  n_chs[i-1]           # 第三次拼接后的输出要用上一层的channal数，这个是规定/经验值
            up_BlockList.append(DownUpBlock(in_channal, out_channal, time_channal, is_attn[i] ))
            in_channal = out_channal

            if i > 1:
                up_BlockList.append(UpSample(in_channal)) # 因为sample不变channal ，但是要注意不是最后一层不用sample，而是第一层不用

        self.up = nn.ModuleList(up_BlockList)
        # 最后还有一个卷积，得顺便归一化激活一下（经验）因为最后一层的输出关系到最终的图片（估计噪声），操作一下结果会更稳定些。
        # 为什么不在最后加tanh？额，我觉得可能是因为这里是噪声生成不是图片生成，也可以加一个试试看看效果
        # in_channal = n_chs[0] # 此时的in_channal已经是第0层的大小了
        out_channal = image_c   # 最后一个卷积，输出原始图片大小的channal
        self.norm = nn.GroupNorm(8, in_channal)
        self.act = nn.SiLU()
        self.conv_final = nn.Conv2d(in_channal, out_channal, kernel_size=3, padding=1)

    def forward(self, x:torch.Tensor, t: torch.Tensor):

        # 提取t的向量表示
        t = self.time_embed(t)
        # 第一层卷积
        x = self.conv_pre(x)

        h=[]
        h.append(x) # 这个其实也可以当栈用，额，简单来说就是可以塞进去一个，等需要再拿出来

        # Encoder
        for m in self.down:         # m这里就是遍历down里的函数了。
            x = m(x,t)              # '很巧'每一层都是x和t的输入，这里如果有一些不是就可以用optional让他假装是一下
            # print(x.shape)
            h.append(x)             # 存一下之后残差链接
        # Middle
        x = self.middle(x,t)
        # Decoder
        for m in self.up:
            if isinstance(m, UpSample): # isinstance()用于检查一个对象 m 是否是某个指定类型（class） UpSample 的实例（instance）。
                x = m(x,t)
            else:                       # 不是upsample都要拼接
                s = h.pop()             # 把最近塞进h的东西弹粗来（从结果来看，弹出来包括赋值和删除）
                # print(x.shape,s.shape)
                x = torch.cat((x, s), dim = 1) # 沿着维度1进行拼接，dim=1就是channal维，就是对准其他维度，把channal维拼起来
                x = m(x, t)
        x = self.conv_final(self.act(self.norm(x)))
        return x

def build_unet():
    image_c = get_img_shape()[0]
    n_chs = (64, 64, 128, 256, 512, 1024)
    is_attn = (False, False, False, False, True, True)
    # n_chs = (32, 32, 64, 64, 128)
    # is_attn = (False, False, False, True, True)
    n_blocks = 2
    net = UNet(image_c, n_chs, is_attn, n_blocks)
    return net