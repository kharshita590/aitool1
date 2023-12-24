import torch
from torch import nn

device = torch.device("cpu")


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 discriminator=False,
                 use_act=True,
                 use_bn=True,
                 **kwargs):
        super().__init__()
        self.use_act = use_act
        self.cnn = nn.Conv2d(in_channels, out_channels,
                             **kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = (
            nn.LeakyReLU(0.2, inplace=True)
            if discriminator
            else nn.PReLU(num_parameters=out_channels)
        )

    def forward(self, x):
        x = self.act(self.bn(self.cnn(x))
                     ) if self.use_act else self.bn(self.cnn(x))
        return x


class Upsample(nn.Module):
    def __init__(self, in_c, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_c, in_c * scale_factor**2, 3, 1, 1)
        self.ps = nn.PixelShuffle(scale_factor)
        self.act = nn.PReLU(num_parameters=in_c)

    def forward(self, x):
        x = self.act(self.ps(self.conv(x)))
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block1 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.block2 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_act=False,
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out + x


class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=3, num_blocks=16):
        super().__init__()
        self.initial = ConvBlock(
            in_channels, num_channels, kernel_size=9, stride=1, padding=4,
            use_bn=False)
        self.residuals = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.convblock = ConvBlock(
            num_channels, num_channels, kernel_size=3, stride=1, padding=1,
            use_act=False)
        self.upsamples = nn.Sequential(
            Upsample(num_channels, scale_factor=2), Upsample(num_channels,
                                                             scale_factor=2))
        self.final = nn.Conv2d(num_channels, in_channels,
                               kernel_size=9, stride=1, padding=4)

    def forward(self, noise):
        x = self.initial(noise)
        x = self.residuals(x)
        x = self.convblock(x) + noise
        x = self.upsamples(x)
        return torch.tanh(self.final(x))


class Discriminator(nn.Module):
    def __init__(self, in_channels, features=[64, 64, 128, 256,
                                              256, 512, 512]):
        super(Discriminator, self).__init__()
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels,
                    feature,
                    kernel_size=3,
                    stride=1 + idx % 2,
                    padding=1,
                    use_act=True,
                    discriminator=True,
                    use_bn=False if idx == 0 else True
                )
            )
            in_channels = feature
        self.blocks = nn.Sequential(*blocks)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(features[-1], 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.blocks(x)
        x = self.global_avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def test():
    with torch.cuda.amp.autocast():
        noise = torch.randn(6, 3, 64, 64)
        gen = Generator()
        gen_out = gen(noise)
        disc = Discriminator(in_channels=3)
        disc_out = disc(gen_out)

        print("gen", gen_out.shape)
        print("disc", disc_out.shape)


if __name__ == '__main__':
    test()
