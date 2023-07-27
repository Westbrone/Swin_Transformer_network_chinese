import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

#第一步完成实验中第一步位置编码
#取消与之前代码中patch_emd,修改成与论文网络结构图中的Patch_Parition_and_Linear_embedding
#相比于vision_trasnformer中添加了图像的padding操作，因此可以输入多尺度大小图像
class Patch_Parition_and_Linear_embedding():
    #对于模型的第一步输入了Batch_size张大小为img_size的RGB图像，这里使用的96维的，与模型图的48有不同
    def __init__(self, patch_size=4, in_channel=3, embed_dim=96, norm_layer=None):
        super.__init__()
        patch_size = (patch_size,patch_size)#后面的卷积需要输入元组类型的kernel_size和stride
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.Patch_Parition = nn.Conv2d(in_channel, embed_dim, kernel_size=patch_size, stride=patch_size)
        #使用nn.Conv2d需要注意它的使用输入数据的格式是(B,C,H,W)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self,x):
        _, _, H, W =x.shape#(B,C,H,W)
        #了图像的padding操作，因此可以输入多尺度大小图像
        pad_input = (H % self.patch_size[0] != 0 or W % self.patch_size[0] != 0)
        #填充最后 3 个维度，请使用（左填充,右填充,顶部填充,填充底部,前边距,后边距）,也就是同时填充C,H,W三个维度，B维度不需要填充
        if pad_input:
            x = F.pad(x,(self.patch_size[1] - W % self.patch_size[1], 0,
                         self.patch_size[1] - H % self.patch_size[1], 0,
                         0, 0))
        x = self.Patch_Parition(x)#进行卷积下采样4倍
        _, _, H, W =x.shape#H，W已经变为原始四分之一
        x = x.flatten(2).transpose(1,2)#X---(B,,H/4*W/4,96)
        x = self.norm(x)#最后一维（通道）进行归一化
        return x, H, W

#第二步定义接下来的一个单独的swin_trasnformer_block
#不同的swin_trasnformer_block的区别就在有无存在移动窗口计算，通过shift_size来控制判别是否需要移动窗口
#两个连续的swin_trasnformer_block的必有 W-MSA和SW-MSA
class Swin_Transformer_Block(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=0.,
                 qkv_bias=True, drop=0, attn_drop=0, drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super.__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "移动窗口距离必须小于窗口大小"

        self.norm1 = norm_layer(dim)#定义block中的第一个归一层
        self.norm2 = norm_layer(dim)#定义block中的第二个归一层
        self.attn = WindowAttention()#计算注意力得分
        self.drop = nn.Identity()#不采用drop，不做处理
        mlp_hidden_dim = int(mlp_ratio * dim)#定义中间隐藏层的维度
        self.mlp=Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)#普通的MLP

    def forward(self,x,attn_mask):
        H ,W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W ,"输入特征大小，错误"

        shortcut = x#保存原始x，后面残差连接
        x = self.norm1(x)
        x = x.view(B,H,W,C)
        # 把feature map给pad到window size的整数倍，为了方便后面的窗口划分window_partition
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))#?????有些问题
        _, Hp, Wp, _ = x.shape

        # 移动窗口通过循环图像进行
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))#进行SW-MSA
        else:
            shifted_x = x
            attn_mask = None#只进行W-MSA


        # 窗口分割成[nW*B, Wh, Ww, C]一张图像包括nW个窗口， Wh，Ww----窗口高--窗口宽，C--通道数
        x_windows = Window_partition(shifted_x, self.window_size)  # [nW*B, Wh, Ww, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Wh*Ww, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Wh*Ww, C]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Wh, Ww, C]
        shifted_x = Window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        # 窗口反转回去
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉，只保留最后的H维度和W维度的数据。contiguous保证内存连续性
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.shift_size = window_size // 2

        # 建立一个block列表
        self.blocks = nn.ModuleList([
            Swin_Transformer_Block(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = Window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]
        for blk in self.blocks:
            x = blk(x, attn_mask)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2

        return x, H, W



class SwinTransformer(nn.Module):

    def __init__(self, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # stage4输出特征矩阵的channels
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = Patch_Parition_and_Linear_embedding(
            patch_size=patch_size, in_channel=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 注意这里构建的stage和论文图中有些差异
            # 这里的stage不包含该stage的patch_merging层，包含的是下个stage的
            layers = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=Patch_Merging if (i_layer < self.num_layers - 1) else None,#最后一个block不需要在进行Patch_Merging
                                )
            self.layers.append(layers)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: [B, L, C]
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x, H, W = layer(x, H, W)

        x = self.norm(x)  # [B, L, C]
        x = self.avgpool(x.transpose(1, 2))  # [B, C, 1]
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x




#输入的数据为[nW*B, Wh*Ww, C]，同时传入self.window_size
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Wh, Ww]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5#就是公式里面的d

        # 定义一个可训练的位置偏置，多头注意力每一头都有单独的位置偏置
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Wh-1 * 2*Ww-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])#默认以零为起点，生成一个连续的自然数序列张量长度为Wh
        coords_w = torch.arange(self.window_size[1])#默认以零为起点，生成一个连续的自然数序列张量长度为Ww
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Wh, Ww]
        """举个例子：grid[0] 是一个 5x5 的网格，表示高度坐标的网格，grid[1] 是一个 5x5 的网格，
        表示宽度坐标的网格。这两个网格的值分别对应了 [0, 1, 2, 3, 4] 这个一维张量的值
        grid：
        tensor([[[0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1],
         [2, 2, 2, 2, 2],
         [3, 3, 3, 3, 3],
         [4, 4, 4, 4, 4]],

        [[0, 1, 2, 3, 4],
         [0, 1, 2, 3, 4],
         [0, 1, 2, 3, 4],
         [0, 1, 2, 3, 4],
         [0, 1, 2, 3, 4]]])
        """
        coords_flatten = torch.flatten(coords, 1)  # 从第一维之后所有的维度都展平[2, Wh*Ww]
        """
        tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
        [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]])
        """
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Wh*Ww, Wh*Ww]
        '''
        tensor([[[ 0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  3,  3,
           3,  3,  3,  4,  4,  4,  4,  4],
         [ 0, -1, -2, -3, -4,  0, -1, -2, -3, -4,  0, -1, -2, -3, -4,  0, -1,
          -2, -3, -4,  0, -1, -2, -3, -4]],

        [[ 0,  1,  2,  3,  4,  0,  1,  2,  3,  4,  0,  1,  2,  3,  4,  0,  1,
           2,  3,  4,  0,  1,  2,  3,  4],
         [ 0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -2, -2, -2, -2, -2, -3, -3,
          -3, -3, -3, -4, -4, -4, -4, -4]]])
         '''
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Wh*Ww, Wh*Ww, 2]置换维度顺序之后进行。contiguous()保证在内存中的连续性
        '''
        tensor([[[ 0,  0],
         [-1,  0],
         [-2,  0],
         [-3,  0],
         [-4,  0]],

        [[ 1,  0],
         [ 0,  0],
         [-1,  0],
         [-2,  0],
         [-3,  0]]])
        '''
        relative_coords[:, :, 0] += self.window_size[0] - 1  # 高度坐标+=Wh-1
        '''
        tensor([[[ 4,  0],
         [ 3,  0],
         [ 2,  0],
         [ 1,  0],
         [ 0,  0]],

        [[ 5,  0],
         [ 4,  0],
         [ 3,  0],
         [ 2,  0],
         [ 1,  0]]])
        '''
        relative_coords[:, :, 1] += self.window_size[1] - 1
        '''
        tensor([[[ 4,  4],
         [ 3,  4],
         [ 2,  4],
         [ 1,  4],
         [ 0,  4]],

        [[ 5,  4],
         [ 4,  4],
         [ 3,  4],
         [ 2,  4],
         [ 1,  4]]])
        '''
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1 # [Wh*Ww, Wh*Ww, 2]
        relative_position_index = relative_coords.sum(-1)  # [Wh*Ww, Wh*Ww]最后一个维度相加，也就是行列相对坐标相加
        self.register_buffer("relative_position_index", relative_position_index)#读入内存，这个是固定的，不进行参数学习

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)#可以省略浮点数小数点之前的零
        self.softmax = nn.Softmax(dim=-1)#对每行进行softmax

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        # [nW*B, Wh*Ww, C]
        B_, N, C = x.shape
        # qkv(): -> [nW*B, Wh*Ww, 3*C]
        # reshape: -> [nW*B, Wh*Ww, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, nW*B , num_heads, Wh*Ww, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) #将 qkv 沿着第一个维度拆分为三个张量 q, k, v


        q = q * self.scale#[nW * B, num_heads, Wh * Ww, embed_dim_per_head]
        attn = (q @ k.transpose(-2, -1))# [nW * B, num_heads, Wh * Ww, Wh * Ww]
        #self.relative_position_index--[Wh*Ww, Wh*Ww]
        # relative_position_bias_table.view: [Wh*Ww*Wh*Ww,nH] -> [Wh*Ww,Wh*Ww,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_feature, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_feature, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    # 定义一个熟悉的Mlp结果
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def Window_partition(x, window_size: int):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Wh, Wh, W//Ww, Ww, C] -> [B, H//Mh, W//Wh, Ww, Ww, C]
    # view: [B, H//Mh, W//Ww, Wh, Ww, C] -> [B*num_windows, Wh, Ww, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def Window_reverse(windows, window_size: int, H: int, W: int):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Wh, Ww, C] -> [B, H//Wh, W//Ww, Wh, Ww, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Wh, W//Ww, Wh, Ww, C] -> [B, H//Wh, Mh, W//Ww, Ww, C]
    # view: [B, H//Wh, Wh, W//Ww, Ww, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x



class Patch_Merging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        # 如果输入feature map的H，W不是2的整数倍，需要进行padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x

#这里使用这个最小的网络结构，为的是匹配下一个网络swin-unet同样采用swin_tiny_patch4_window7_224的结构
def swin_tiny_patch4_window7_224(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=96,
                            depths=(2, 2, 6, 2),
                            num_heads=(3, 6, 12, 24),
                            num_classes=num_classes,
                            **kwargs)
    return model