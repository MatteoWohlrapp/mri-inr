import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, inp, out, paddi=1):
        super().__init__()

        self.conv1 = nn.Conv2d(inp, out, kernel_size=3, padding=paddi)
        self.conv2 = nn.Conv2d(out, out, kernel_size=3, padding=paddi)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)

        x = self.relu(x)

        x = self.conv2(x)

        x = self.relu(x)

        return x


class downsample_block(nn.Module):
    def __init__(self, inp, out):
        super().__init__()

        self.conv = conv_block(inp, out)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p


class upsample_block(nn.Module):
    def __init__(self, inp, out):
        super().__init__()

        self.up = nn.ConvTranspose2d(inp, out, kernel_size=2, stride=2)
        self.conv = conv_block(out + out, out)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.d1 = downsample_block(1, 64)
        self.d2 = downsample_block(64, 128)
        self.d3 = downsample_block(128, 256)
        self.d4 = downsample_block(256, 512)

        self.b = conv_block(512, 1024, 1)

        self.u1 = upsample_block(1024, 512)
        self.u2 = upsample_block(512, 256)
        self.u3 = upsample_block(256, 128)
        self.u4 = upsample_block(128, 64)

        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):

        s1, p1 = self.d1(inputs)
        s2, p2 = self.d2(p1)
        s3, p3 = self.d3(p2)
        s4, p4 = self.d4(p3)

        b = self.b(p4)

        u1 = self.u1(b, s4)
        u2 = self.u2(u1, s3)
        u3 = self.u3(u2, s2)
        u4 = self.u4(u3, s1)

        outputs = self.outputs(u4)

        return outputs