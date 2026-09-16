import torch
import torch.nn as nn

from pytorch_wavelets import DWTForward





class WaveLetDown(nn.Module):
    def __init__(self, inchanel):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()

        self.wt = DWTForward(J=1, mode='zero', wave='haar')

        self.wtconv = nn.Conv2d(int(inchanel*2), inchanel, kernel_size=1, padding=0, )
        self.conv = nn.Conv2d( int(inchanel/2), inchanel, kernel_size=3, padding=1, )
        self.conv2 = nn.Conv2d( int(inchanel/2), inchanel,stride=2, kernel_size=3, padding=1, )

    def forward(self, x):
        x1, x2 = x.chunk(2, 1)

        yL, yH = self.wt(x1)
        HL = yH[0][:, :, 0, ::]
        LH = yH[0][:, :, 1, ::]
        HH = yH[0][:, :, 2, ::]
        wtx = torch.cat([yL, HL, LH, HH], dim=1)
        x1 = self.wtconv(wtx)

        x2 = self.conv2(x2)
        return torch.cat((x1, x2), 1)




class Encoder(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 dim=32,
                 num_blocks=None,
                 ):
        super(Encoder, self).__init__()
        if num_blocks is None:
            num_blocks = [2, 4, 4, 6]
        self.num_blocks = num_blocks
        self.feature_embed = nn.Conv2d(in_channels=inp_channels, out_channels=dim, kernel_size=3, padding=1, stride=1,
                                       groups=1, bias=True)
        self.b1 = nn.Sequential(*[MultiDWConv(dim) for _ in range(num_blocks[0])])
        self.down1 = WaveLetDown(dim)
        self.b2 = nn.Sequential(*[MultiDWConv(dim * 2) for _ in range(num_blocks[1])])
        self.down2 = WaveLetDown(dim * 2)
        self.b3 = nn.Sequential(*[MultiDWConv(dim * 2 ** 2) for _ in range(num_blocks[2])])
        self.down3 = WaveLetDown(dim * 2 ** 2)
        self.b4 = nn.Sequential(*[MultiDWConv(dim * 2 ** 3) for _ in range(num_blocks[3])])

    def forward(self, x):
        x = self.feature_embed(x)  # (1, 32, 256, 256)
        x1 = self.b1(x)  # (1, 32, 256, 256)

        x = self.down1(x1)  # (1, 64, 128, 128)
        x2 = self.b2(x)  # (1, 64, 128, 128)

        x = self.down2(x2)  # (1, 128, 64, 64)
        x3 = self.b3(x)  # (1, 128, 64, 64)

        x = self.down3(x3)
        x4 = self.b4(x)

        return x4, x3, x2, x1


class MultiDWConv(nn.Module):
    """Multi-order Features with Dilated DWConv Kernel.

    Args:
        embed_dims (int): Number of input channels.
        dw_dilation (list): Dilations of three DWConv layers.
        channel_split (list): The raletive ratio of three splited channels.
    """

    def __init__(self,
                 channel,
                ):
        super(MultiDWConv, self).__init__()




        self.channel = channel


        # basic DW conv
        self.DW_conv0 = nn.Conv2d(
            in_channels=self.channel,
            out_channels=self.channel,
            kernel_size=3,
            padding=3 // 2,
            groups=self.channel,
            stride=1,
        )
        # DW conv 1
        self.DW_conv1 = nn.Conv2d(
            in_channels=self.channel,
            out_channels=self.channel,
            kernel_size=3,
            padding=3 // 2,
            groups=self.channel,
            stride=1,
        )
        # DW conv 2
        self.DW_conv2 = nn.Conv2d(
            in_channels=self.channel,
            out_channels=self.channel,
            kernel_size=5,
            padding=5 // 2,
            groups=self.channel,
            stride=1,
        )
        # a channel convolution
        self.PW_conv = nn.Conv2d(  # point-wise convolution
            in_channels=self.channel*3,
            out_channels=self.channel,
            kernel_size=1)
        self.actg =nn.GELU()

        self.gateconv =  nn.Conv2d(in_channels=self.channel, out_channels=self.channel, kernel_size=1)
        self.outconv = nn.Conv2d(in_channels=self.channel, out_channels=self.channel, kernel_size=1)
        self.act = nn.SiLU()
        self.pa = PALayer(self.channel)
        self.ca = CALayer(self.channel)

    def forward(self, x):
        shortcut = x.clone()

        x_0 = self.actg(self.DW_conv0(x))
        x_1 = self.actg(self.DW_conv1(x_0))
        x_2 = self.actg(self.DW_conv2(x_0))
        x_dw = torch.cat([x_0, x_1, x_2], dim=1)
        x_dw = self.PW_conv(x_dw)

        x_gate = self.gateconv(x)

        x = self.outconv(self.act(x_gate) + self.act(x_dw))


        x = self.ca(x)
        x = self.pa(x)

        x = x + shortcut
        return x

class Fusion(nn.Module):
    def __init__(self,
                 dim=32,
                 num_blocks=[2, 4, 4, 6],
                 ):
        super(Fusion, self).__init__()
        self.num_blocks = num_blocks
        self.up43 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 3, dim * 2 ** 4, 1, bias=False),
            nn.PixelShuffle(2)
        )
        self.d3 = nn.Sequential(*[FusionBlock(dim * 2 ** 2) for _ in range(num_blocks[2])])
        self.up32 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 2, dim * 2 ** 3, 1, bias=False),
            nn.PixelShuffle(2)
        )
        self.d2 = nn.Sequential(*[FusionBlock(dim * 2) for _ in range(num_blocks[1])])

    def forward(self, x4, x3, x2, x1):
        x3_b = x3.contiguous()
        x = self.up43(x4) + x3
        x3 = self.d3(x)
        # deblur head x3(min) 128
        x2_b = x2.contiguous()
        x = self.up32(x3) + x2
        x2 = self.d2(x)

        return x4, x3, x3_b, x2, x2_b, x1

class Deblur_head(nn.Module):
    def __init__(self, num_in, num_mid, num_out):
        super().__init__()

        self.block = nn.Sequential(
            # nn.Conv2d(num_in, num_mid, kernel_size=1),
            # nn.BatchNorm2d(num_mid),
            # nn.GELU(),
            nn.Conv2d(num_in, num_out, kernel_size=3, stride=1, padding=1),


        )

    def forward(self, x):
        x = self.block(x)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 dim=64,
                 out_channels=3,
                 num_blocks=[2, 4, 4, 6],
                 ):
        super().__init__()
        self.num_blocks = num_blocks

        self.head4 = Deblur_head(int(dim * 2 ** 3), int(dim * 3), 3)
        self.up43 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 3, dim * 2 ** 4, 1, bias=False),
            EdgeEnhancer(dim * 2 ** 4),
            nn.PixelShuffle(2)
        )
        self.head3 = Deblur_head(int(dim * 2 ** 2), int(dim * 2 ** 1), out_channels)
        self.up32 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 2, dim * 2 ** 3, 1, bias=False),
            EdgeEnhancer(dim * 2 ** 3),
            nn.PixelShuffle(2)
        )

        self.head2 = Deblur_head(int(dim * 2 ** 1), int(dim), out_channels)
        self.up21 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 1, dim * 2 ** 2, 1, bias=False),
            EdgeEnhancer(dim * 2 ** 2),
            nn.PixelShuffle(2)
        )

        self.head1 = Deblur_head(dim, dim, out_channels)

        self.d4 = nn.Sequential(*[MultiDWConv(dim * 2 ** 3) for _ in range(num_blocks[3])])
        self.d3 = nn.Sequential(*[MultiDWConv(dim * 2 ** 2) for _ in range(num_blocks[2])])
        self.d2 = nn.Sequential(*[MultiDWConv(dim * 2) for _ in range(num_blocks[1])])
        self.d1 = nn.Sequential(*[MultiDWConv(dim) for _ in range(num_blocks[0])])


        self.alpha = nn.Parameter(torch.zeros((1, dim * 2, 1, 1)), requires_grad=True)

    def forward(self, x4, x3, x3_b, x2, x2_b, x1):
        # x = x4.contiguous()
        x = self.d4(x4)
        x4 = self.head4(x) if self.training else None

        x = self.up43(x) + x3
        x = self.d3(x)
        x3 = self.head3(x) if self.training else None

        x2_n = x2.contiguous()
        x = self.up32(x) + x2
        x = self.d2(x)
        x2 = self.head2(x) if self.training else None

        x = self.up21(x) + x1
        x = self.d1(x)
        x1 = self.head1(x)

        return x1, x2, x3, x4



class MLWNet(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=64,
                 ):

        super(MLWNet, self).__init__()
        # [False, True, True, False]
        # [False, False, False, False]
        self.encoder = Encoder(inp_channels=inp_channels,
                                 dim=dim,
                                 num_blocks=[6, 6, 6, 6],
                                 )
        self.fusion = Fusion(dim=dim,
                         num_blocks=[None, 2, 2, None],
                         )
        self.decoder = Decoder(dim=dim,
                         out_channels=out_channels,
                         num_blocks=[6, 6, 6, 6],
                         )

    def __repr__(self):
        return 'MLWNet'

    def forward(self, inp):
        x = self.encoder(inp)
        x = self.fusion(*x)
        x1, x2, x3, x4 = self.decoder(*x)
        return x1, x2, x3, x4




class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class FusionBlock(nn.Module):
    def __init__(self, c, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * 2
        self.ftbock = FourierUnit(c,c)

        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        ffn_channel = 2 * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.ftbock(x)

        x = x * self.ca(x)

        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        # gate
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * x2
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma

class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.norm = LayerNorm2d(in_channels*2)
        self.act = torch.nn.GELU()

    def forward(self, x):
        batch, c, h, w = x.size()

        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        # (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.act(self.norm(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.view_as_complex(ffted)

        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')

        return output


class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            LayerNorm2d(in_dim),
            nn.Sigmoid()
        )
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
