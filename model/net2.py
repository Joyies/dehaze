import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    # default
    # bn = False relu=true bias = False
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=(0,0), dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, dilation = 1, bn=False):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation= dilation)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.bn:
            x = self.bn(out)
        out = self.relu(out)
        return out

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(Upsample, self).__init__()
      reflection_padding = kernel_size // 2
      self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
      self.relu = nn.ReLU()

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return self.relu(out)

class Blockbranch1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Blockbranch1,self).__init__()
        # self.Conv1x1 = ConvLayer(in_channels, out_channels, kernel_size=1, stride=1)
        # self.Conv3x3 = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1)
        self.Convkx1 = BasicConv(out_channels, out_channels, kernel_size=(1,1), stride=1)
        self.Convkx1_3x3 = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1)
            # BasicConv(out_channels, out_channels, kernel_size=3, stride=1, padding=3, dilation=3,
            #                          relu=False)
        self.Conv1xk = BasicConv(out_channels, out_channels, kernel_size=(1, 1), stride=1)
        self.Conv1xk_3x3 = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        # x = self.Conv1x1(x)
        # x = self.Conv3x3(x)
        x_kx1 = self.Convkx1_3x3(self.Convkx1(x))
        x_1xk = self.Conv1xk_3x3(self.Conv1xk(x))

        return torch.add(x_1xk, x_kx1)

class Blockbranch2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Blockbranch2, self).__init__()

        # self.Conv1x1 = ConvLayer(in_channels, out_channels, kernel_size=1, stride=1)
        # self.Conv3x3 = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1)
        self.Convkx1 = BasicConv(out_channels, out_channels, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.Convkx1_3x3 = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1)
        # BasicConv(out_channels, out_channels, kernel_size=3, stride=1, padding=3, dilation=3,
        #                          relu=False)
        self.Conv1xk = BasicConv(out_channels, out_channels, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.Conv1xk_3x3 = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = self.Conv1x1(x)
        # x = self.relu(self.Conv3x3(x))
        x_kx1 = self.Convkx1_3x3(self.Convkx1(x))
        x_1xk = self.Conv1xk_3x3(self.Conv1xk(x))

        return torch.add(x_1xk, x_kx1)

class Blockbranch3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Blockbranch3, self).__init__()

        # self.Conv1x1 = ConvLayer(in_channels, out_channels, kernel_size=1, stride=1)
        # self.Conv3x3 = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1)
        self.Convkx1 = BasicConv(out_channels, out_channels, kernel_size=(7, 1), stride=1, padding=(3, 0))
        self.Convkx1_3x3 = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1)
        # BasicConv(out_channels, out_channels, kernel_size=3, stride=1, padding=3, dilation=3,
        #                          relu=False)
        self.Conv1xk = BasicConv(out_channels, out_channels, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.Conv1xk_3x3 = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = self.relu(self.Conv1x1(x))
        # x = self.relu(self.Conv3x3(x))
        x_kx1 = self.Convkx1_3x3(self.Convkx1(x))
        x_1xk = self.Conv1xk_3x3(self.Conv1xk(x))

        return torch.add(x_1xk, x_kx1)

class Block(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(Block, self).__init__()
        self.Conv1x1 = ConvLayer(in_channels, out_channels, kernel_size=1, stride=1)
        self.Conv3x3 = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1)
        self.branch0 = ConvLayer(in_channels, out_channels, kernel_size=1, stride=1)  # 1x1
        self.branch1 = Blockbranch1(in_channels, out_channels) # 1x1
        self.branch2 = Blockbranch2(in_channels, out_channels) # 3x3
        self.branch3 = Blockbranch3(in_channels, out_channels) # 7x7

        self.conv = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1)
        self.relu = nn.ReLU()

    def forward(self,x):

        x0 = self.branch0(x)
        x = self.Conv1x1(x)
        x = self.Conv3x3(x)
        # x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        # x = torch.cat((x0,x1,x2,x3),1)
        x = torch.add(x0, x1)
        x = torch.add(x2, x)
        x = torch.add(x3, x)
        x = self.conv(x)

        return self.relu(x)


## Channels Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class CAmodule(nn.Module):

    def __init__(self, in_channels, reduction=16):
        super(CAmodule, self).__init__()

        self.calayer = CALayer(in_channels, reduction=reduction)  # branch1: channel attention
        # --branch2: map attention
        self.downsample = ConvLayer(in_channels, 2*in_channels, stride=2, kernel_size=3)
        self.middle = ConvLayer(2*in_channels, 2*in_channels, kernel_size=3, stride=1)
        self.upsample = Upsample(2*in_channels, in_channels, kernel_size=3, stride=1)

    def forward(self, x, y):
        # branch1
        out1 = self.calayer(x)
        # branch2
        out = self.downsample(x)
        out = torch.add(out, y)  # cat
        out = self.middle(out)
        out = self.upsample(out)
        out = F.upsample(out, x.size()[2:], mode='bilinear')
        # out = F.interpolate(out, x.size()[2:], mode='bilinear', align_corners=True)

        return torch.add(out, out1)

class CAmodule2(nn.Module):

    def __init__(self, in_channels, reduction=16):
        super(CAmodule2, self).__init__()

        self.calayer = CALayer(in_channels, reduction=reduction)  # branch1: channel attention
        # --branch2: map attention
        self.downsample = ConvLayer(in_channels, 2*in_channels, stride=2, kernel_size=3)
        self.middle = ConvLayer(2*in_channels, 2*in_channels, kernel_size=3, stride=1)
        self.upsample = Upsample(2*in_channels, in_channels, kernel_size=3, stride=1)

    def forward(self, x):
        # branch1
        out1 = self.calayer(x)
        # branch2
        out = self.downsample(x)
        out = self.middle(out)
        out = self.upsample(out)
        out = F.upsample(out, x.size()[2:], mode='bilinear')
        # out = F.interpolate(out, x.size()[2:], mode='bilinear', align_corners=True)

        return torch.add(out, out1)

class BasicBlock_Residual2(nn.Module):

    def __init__(self, in_planes, planes, stride=1):

        super(BasicBlock_Residual2, self).__init__()
        self.pading = nn.ReflectionPad2d(1)
        self.conv_init = ConvLayer(in_planes, in_planes, kernel_size=1, stride = stride)
        self.conv1 = ConvLayer(in_planes, planes, kernel_size=3, stride = stride)
        self.conv2 = ConvLayer(planes, planes, kernel_size=3, stride = stride)
        self.relu = nn.ReLU()

    def forward(self, x):

        residul = self.conv_init(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.add(out,residul)

        return out


class net(nn.Module):
    def __init__(self, n=1):

        super(net, self).__init__()
        # layer1
        self.pading = nn.ReflectionPad2d(2)
        self.conv2_init_1 = nn.Conv2d(3, 8*n, kernel_size=5)
        self.conv2_init_2 = nn.Conv2d(8*n, 16*n, kernel_size=5)
        self.layer1_E = Block(16*n, 16*n)
        # self.layer1_E = BasicBlock_Residual2(16*n, 16*n)
        self.layer1_CA = CAmodule(16*n, 1)

        # layer2
        self.layer2_down = ConvLayer(16*n, 32*n, kernel_size=3, stride=2)
        self.layer2_E = Block(32*n, 32*n)
        # self.layer2_E = BasicBlock_Residual2(32*n, 32*n)
        self.layer2_CA = CAmodule(32*n, 2)

        # layer3
        self.layer3_down = ConvLayer(32*n, 64*n, kernel_size=3, stride=2)
        self.layer3_E = Block(64*n, 64*n)
        # self.layer3_E = BasicBlock_Residual2(64*n, 64*n)
        self.layer3_CA = CAmodule(64*n, 4)

        # layer4
        self.layer4_down = ConvLayer(64*n, 128*n, kernel_size=3, stride=2)
        self.layer4_E = Block(128*n, 128*n)
        # self.layer4_E = BasicBlock_Residual2(128*n, 128*n)
        self.layer4_CA = CAmodule(128*n, 8)

        #layer5
        self.layer5_down = ConvLayer(128*n, 256*n, kernel_size=3, stride=2)
        self.layer5_E = Block(256*n, 256*n)
        # self.layer5_E = BasicBlock_Residual2(256*n, 256*n)
        self.layer5_CA = CAmodule2(256*n, 16)


        self.center = nn.Sequential(
            BasicBlock_Residual2(256 * n, 256 * n),
            BasicBlock_Residual2(256 * n, 256 * n),
            BasicBlock_Residual2(256 * n, 256 * n)
        )


        #layer 5D
        self.layer5_D = BasicBlock_Residual2(256*n, 256*n)
        # self.layer5_D = Block(256, 256)
        self.layer5_O = nn.Sequential(
            ConvLayer(256*n, 64*n, kernel_size=3, stride=1),
            ConvLayer(64*n, 3, kernel_size=3, stride=1)
        )
        self.layer5_U = Upsample(256*n, 128*n, kernel_size=3, stride=2)


        # layer 4D
        self.layer4_D = BasicBlock_Residual2(128*n, 128*n)
        # self.layer4_D = Block(128, 128)
        self.layer4_O = nn.Sequential(
                ConvLayer(128*n, 32*n, kernel_size=3, stride=1),
                ConvLayer(32*n, 3, kernel_size=3, stride=1)
        )
        self.layer4_U = Upsample(128*n, 64*n, kernel_size=3, stride=2)

        # layer 3D
        self.layer3_D = BasicBlock_Residual2(64*n, 64*n)
        # self.layer3_D = Block(64, 64)
        self.layer3_O = nn.Sequential(
            ConvLayer(64*n, 16*n, kernel_size=3, stride=1),
            ConvLayer(16*n, 3, kernel_size=3, stride=1)
        )
        self.layer3_U = Upsample(64*n, 32*n, kernel_size=3, stride=2)

        # layer 2D
        self.layer2_D = BasicBlock_Residual2(32*n, 32*n)
        # self.layer2_D = Block(32, 32)
        self.layer2_O = nn.Sequential(
            ConvLayer(32*n, 8*n, kernel_size=3, stride=1),
            ConvLayer(8*n, 3, kernel_size=3, stride=1)
        )
        self.layer2_U = Upsample(32*n, 16*n, kernel_size=3, stride=2)

        # layer 1D
        self.layer1_D = BasicBlock_Residual2(16*n, 16*n)
        # self.layer1_D = Block(16, 16)
        self.layer1_O = ConvLayer(16*n, 3, kernel_size=3, stride=1)

        self.relu = nn.ReLU()

    def forward(self,x):

        # E
        out = self.pading(x)
        out = self.relu(self.conv2_init_1(out))
        out = self.pading(out)
        out = self.relu(self.conv2_init_2(out))

        out1 = self.layer1_E(out)

        out = self.layer2_down(out1)
        out2 = self.layer2_E(out)
        out1 = self.layer1_CA(out1, out2)

        out = self.layer3_down(out2)
        out3 = self.layer3_E(out)
        out2 = self.layer2_CA(out2, out3)

        out = self.layer4_down(out3)
        out4 = self.layer4_E(out)
        out3 = self.layer3_CA(out3, out4)

        out = self.layer5_down(out4)
        out = self.layer5_E(out)
        out4 = self.layer4_CA(out4, out)
        out = self.layer5_CA(out)

        out = self.center(out)

        # D
        out = self.layer5_D(out)
        im5 = self.layer5_O(out)
        out = self.layer5_U(out)

        out = F.interpolate(out, out4.size()[2:], mode='bilinear', align_corners=True)
        # out = F.upsample(out, out4.size()[2:], mode='bilinear')
        out = torch.add(out, out4)
        out = self.layer4_D(out)
        im4 = self.layer4_O(out)
        out = self.layer4_U(out)

        out = F.interpolate(out, out3.size()[2:], mode='bilinear', align_corners=True)
        # out = F.upsample(out, out3.size()[2:], mode='bilinear')
        out = torch.add(out, out3)
        out = self.layer3_D(out)
        im3 = self.layer3_O(out)
        out = self.layer3_U(out)

        out = F.interpolate(out, out2.size()[2:], mode='bilinear', align_corners=True)
        # out = F.upsample(out, out2.size()[2:], mode='bilinear')
        out = torch.add(out, out2)
        out = self.layer2_D(out)
        im2 = self.layer2_O(out)
        out = self.layer2_U(out)

        out = F.interpolate(out, out1.size()[2:], mode='bilinear', align_corners=True)
        # out = F.upsample(out, out1.size()[2:], mode='bilinear')
        out = torch.add(out, out1)
        out = self.layer1_D(out)
        im1 = self.layer1_O(out)

        return im1, im2, im3, im4, im5


class Blockbranch(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super(Blockbranch, self).__init__()
        self.Conv1x1 = ConvLayer(in_channels, out_channels, kernel_size=1, stride=1)
        self.Conv3x3 = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1)
        self.Convkx1 = BasicConv(out_channels, out_channels, kernel_size=(k, 1), stride=1, padding=(k//2, 0))
        self.COnvkx1_3x3 = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1)
        # BasicConv(out_channels, out_channels, kernel_size=3, stride=1, padding=3, dilation=3,
        #                          relu=False)
        self.Conv1xk = BasicConv(out_channels, out_channels, kernel_size=(1, k), stride=1, padding=(0, k//2))
        self.Conv1xk_3x3 = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1)
        self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.Conv1x1(x))
            x = self.relu(self.Conv3x3(x))
            x_kx1 = self.Convkx1_3x3(self.Convkx1(x))
            x_1xk = self.Conv1xk_3x3(self.Conv1xk(x))

            return self.relu(x_kx1) + self.relu(x_1xk)
