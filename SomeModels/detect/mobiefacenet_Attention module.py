import cv2
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module
import torch
import numpy as np
import onnx


#  yt add start se_模块
def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    # int(v + divisor / 2) // divisor * divisor：四舍五入到8
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class SqueezeExcitation(nn.Module):
    # squeeze_factor: int = 4：第一个FC层节点个数是输入特征矩阵的1/4
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        # 第一个FC层节点个数，也要是8的整数倍
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        # print("yttest     %d" %squeeze_c)
        # 通过卷积核大小为1x1的卷积替代FC层，作用相同
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x有很多channel，通过output_size=(1, 1)实现每个channel变成1个数字
        scale = nn.functional.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = nn.functional.relu(scale, inplace=True)
        scale = self.fc2(scale)
        # 此处的scale就是第二个FC层输出的数据
        scale = nn.functional.hardsigmoid(scale, inplace=True)
        return scale * x        # 和原输入相乘，得到SE模块的输出


#   yt add start end
#  yt add start，3Dse moudle，空间注意力模块
class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x


class SqueezeAttentionBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SqueezeAttentionBlock, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = conv_block(ch_in, ch_out)
        self.conv_atten = conv_block(ch_in, ch_out)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        print(x.shape)
        x_res = self.conv(x)
        print(x_res.shape)
        y = self.avg_pool(x)
        print(y.shape)
        y = self.conv_atten(y)
        print(y.shape)
        y = self.upsample(y)
        print(y.shape[0])
        # yt add, 解决奇数大小特征图下采样、上采样后无法恢复原始特征图大小的问题
        if y.shape != x_res.shape:
            y = torch.nn.functional.interpolate(y, size=[y.shape[2]+1,y.shape[3]+1], scale_factor=None, mode='nearest', align_corners=None,
                                            recompute_scale_factor=None)
        print(y.shape, x_res.shape)
        return (y * x_res) + y


#注意力机制2，空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  #
        print(x.shape)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        print(avg_out.shape)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        print(max_out.shape)
        x_1 = torch.cat([avg_out, max_out], dim=1)
        print(x_1.shape)
        x_2 = self.conv1(x_1)
        print(x_2.shape)
        x = self.sigmoid(x_2) * x
        print(x.shape)
        return x


#注意力机制3
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1 )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        print(x.shape)
        x_compress = self.compress(x)
        print(x_compress.shape)
        x_out = self.spatial(x_compress)
        print(x_out.shape)
        scale = nn.functional.sigmoid(x_out)
        print(scale.shape)
        x = x * scale
        print(x.shape)
        return x


#cbam 即通道注意力+空间注意力
class CBAM(nn.Module):
    def __init__(self,in_channel,reduction=16,kernel_size=7):
        super(CBAM, self).__init__()
        #通道注意力机制
        self.max_pool=nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.mlp=nn.Sequential(
            nn.Linear(in_features=in_channel,out_features=in_channel//reduction,bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel//reduction,out_features=in_channel,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
        #空间注意力机制
        self.conv=nn.Conv2d(in_channels=2,out_channels=1,kernel_size=kernel_size ,stride=1,padding=kernel_size//2,bias=False)

    def forward(self,x):
        #通道注意力机制
        maxout=self.max_pool(x)
        maxout=self.mlp(maxout.view(maxout.size(0),-1))
        avgout=self.avg_pool(x)
        avgout=self.mlp(avgout.view(avgout.size(0),-1))
        channel_out=self.sigmoid(maxout+avgout)
        channel_out=channel_out.view(x.size(0),x.size(1),1,1)
        channel_out=channel_out*x
        #空间注意力机制
        max_out,_=torch.max(channel_out,dim=1,keepdim=True)
        mean_out=torch.mean(channel_out,dim=1,keepdim=True)
        out=torch.cat((max_out,mean_out),dim=1)
        out=self.sigmoid(self.conv(out))
        out=out*channel_out
        return out


#原来基础的mobiefacenet模块
class Flatten(Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvBlock(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential(
            Conv2d(in_c, out_c, kernel, groups=groups, stride=stride, padding=padding, bias=False),
            BatchNorm2d(num_features=out_c),
            PReLU(num_parameters=out_c)
        )

    def forward(self, x):
        return self.layers(x)


class LinearBlock(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(LinearBlock, self).__init__()
        self.layers = nn.Sequential(
            Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False),
            BatchNorm2d(num_features=out_c)
        )

    def forward(self, x):
        return self.layers(x)


class DepthWise(Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(DepthWise, self).__init__()
        self.residual = residual
        self.layers = nn.Sequential(
            ConvBlock(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1)),
            ConvBlock(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride),
            LinearBlock(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        )

    def forward(self, x):
        short_cut = None
        if self.residual:
            short_cut = x
        x = self.layers(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class DepthWise_Se(Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(DepthWise_Se, self).__init__()
        self.residual = residual
        self.layers = nn.Sequential(
            ConvBlock(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1)),
            ConvBlock(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride),
            SqueezeExcitation(groups),
            LinearBlock(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        )

    def forward(self, x):
        short_cut = None
        if self.residual:
            short_cut = x
        x = self.layers(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(DepthWise(c, c, True, kernel, stride, padding, groups))
        self.layers = Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


class Residual_Se(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual_Se, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(DepthWise_Se(c, c, True, kernel, stride, padding, groups))
        self.layers = Sequential(*modules)

    def forward(self, x):
        return self.layers(x)

class GDC(Module):
    def __init__(self, embedding_size):
        super(GDC, self).__init__()
        self.layers = nn.Sequential(
            LinearBlock(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0)),
            Flatten(),
            Linear(512, embedding_size, bias=False),
            BatchNorm1d(embedding_size))

    def forward(self, x):
        return self.layers(x)


class MobileFaceNet_se(Module):
    def __init__(self, fp16=False, num_features=512, blocks=(1, 4, 6, 2), scale=2, use_se = 0, use_sa = 0):
        super(MobileFaceNet_se, self).__init__()
        self.scale = scale
        self.fp16 = fp16
        self.use_se = use_se
        self.use_sa = use_sa
        self.layers = nn.ModuleList()
        self.layers.append(
            ConvBlock(3, 64 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        )
        if self.use_sa == 1:
            self.layers.append(SqueezeAttentionBlock(ch_in=64 * self.scale, ch_out=64 * self.scale))
        elif self.use_sa == 2:
            self.layers.append(SpatialAttention())
        elif self.use_sa == 3:
            self.layers.append(SpatialGate())
        elif self.use_sa == 4:
            self.layers.append(CBAM(in_channel=64 * self.scale))

        if blocks[0] == 1:
            self.layers.append(
                ConvBlock(64 * self.scale, 64 * self.scale, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
            )
        else:
            self.layers.append(
                Residual(64 * self.scale, num_block=blocks[0], groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1)),
            )
#yt change for add se moudle
        if self.use_se == 1:
            self.layers.extend(
                [
                    DepthWise_Se(64 * self.scale, 64 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1),groups=128),
                    Residual(64 * self.scale, num_block=blocks[1], groups=128, kernel=(3, 3), stride=(1, 1),padding=(1, 1)),
                    DepthWise_Se(64 * self.scale, 128 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256),
                    Residual(128 * self.scale, num_block=blocks[2], groups=256, kernel=(3, 3), stride=(1, 1),padding=(1, 1)),
                    DepthWise_Se(128 * self.scale, 128 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1),groups=512),
                    Residual(128 * self.scale, num_block=blocks[3], groups=256, kernel=(3, 3), stride=(1, 1),padding=(1, 1)),
                ])
        elif self.use_se == 2:
            self.layers.extend(
                [
                    DepthWise(64 * self.scale, 64 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1),groups=128),
                    Residual_Se(64 * self.scale, num_block=blocks[1], groups=128, kernel=(3, 3), stride=(1, 1),padding=(1, 1)),
                    DepthWise(64 * self.scale, 128 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256),
                    Residual_Se(128 * self.scale, num_block=blocks[2], groups=256, kernel=(3, 3), stride=(1, 1),padding=(1, 1)),
                    DepthWise(128 * self.scale, 128 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1),groups=512),
                    Residual_Se(128 * self.scale, num_block=blocks[3], groups=256, kernel=(3, 3), stride=(1, 1),padding=(1, 1)),
                ])
        else:
            self.layers.extend(
                [
                    DepthWise(64 * self.scale, 64 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1),groups=128),
                    Residual(64 * self.scale, num_block=blocks[1], groups=128, kernel=(3, 3), stride=(1, 1),padding=(1, 1)),
                    DepthWise(64 * self.scale, 128 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256),
                    Residual(128 * self.scale, num_block=blocks[2], groups=256, kernel=(3, 3), stride=(1, 1),padding=(1, 1)),
                    DepthWise(128 * self.scale, 128 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1),groups=512),
                    Residual(128 * self.scale, num_block=blocks[3], groups=256, kernel=(3, 3), stride=(1, 1),padding=(1, 1)),
                ])
        if self.use_sa == 1:
            self.layers.append(SqueezeAttentionBlock(ch_in=128 * self.scale, ch_out=128 * self.scale))
        elif self.use_sa == 2:
            self.layers.append(SpatialAttention())
        elif self.use_sa == 3:
            self.layers.append(SpatialGate())
        elif self.use_sa == 4:
            self.layers.append(CBAM(in_channel=128 * self.scale))
        self.conv_sep = ConvBlock(128 * self.scale, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.features = GDC(num_features)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            for func in self.layers:
                x = func(x)
        x = self.conv_sep(x.float() if self.fp16 else x)
        x = self.features(x)
        return x


def get_mbf_se(fp16, num_features, blocks=(1, 4, 6, 2), scale=2, use_se=0, use_sa=0):
    return MobileFaceNet_se(fp16, num_features, blocks, scale=scale, use_se=use_se, use_sa=use_sa)


def get_mbf_se_large(fp16, num_features, blocks=(2, 8, 12, 4), scale=4, use_se=0, use_sa=0):
    return MobileFaceNet_se(fp16, num_features, blocks, scale=scale, use_se=use_se, use_sa=use_sa)
cv2.getGaussianKernel(ksize=31,sigma=31)
if __name__ == "__main__":
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    model  = get_mbf_se(fp16=False, num_features=512,use_se=0, use_sa=4)
    input_names = ['input']
    output_names = ['output']
    img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
    img = img.astype(np.float)
    img = (img / 255. - 0.5) / 0.5  # torch style norm
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    model.eval()
    out = model(img)
    print(out.shape)
    torch.onnx.export(model, img, 'test_mobiefacenet.onnx', input_names=input_names, output_names=output_names, verbose='True')
