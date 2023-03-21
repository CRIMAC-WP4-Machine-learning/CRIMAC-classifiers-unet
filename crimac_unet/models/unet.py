"""
This script is based on
https://github.com/jaxony/unet-pytorch/blob/master/model.py
released under the MIT License:

### MIT License
###
### Copyright (c) 2017 Jackson Huang
###
### Permission is hereby granted, free of charge, to any person obtaining a copy
### of this software and associated documentation files (the "Software"), to deal
### in the Software without restriction, including without limitation the rights
### to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
### copies of the Software, and to permit persons to whom the Software is
### furnished to do so, subject to the following conditions:
###
### The above copyright notice and this permission notice shall be included in all
### copies or substantial portions of the Software.
###
### THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
### IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
### FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
### AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
### LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
### OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
### SOFTWARE.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups,
    )


def upconv2x2(in_channels, out_channels, mode="transpose"):
    if mode == "transpose":
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode="bilinear", scale_factor=2),
            conv1x1(in_channels, out_channels),
        )


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, stride=1)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.main = nn.Sequential(
            conv3x3(self.in_channels, self.out_channels),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            conv3x3(self.out_channels, self.out_channels),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
        )

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.main(x)
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(
        self, in_channels, out_channels, merge_mode="concat", up_mode="transpose"
    ):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)

        if self.merge_mode == "concat":
            self.conv1 = conv3x3(2 * self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

    def forward(self, from_down, from_up):
        """Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == "concat":
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class MetaPostProcessing(nn.Module):
    """ """

    def __init__(self, in_channels, out_channels):
        super(MetaPostProcessing, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels_1 = 32
        self.hidden_channels_2 = 32
        self.out_channels = out_channels

        self.main = nn.Sequential(
            nn.Linear(self.in_channels, self.hidden_channels_1),
            nn.ReLU(),
            nn.Linear(self.hidden_channels_1, self.hidden_channels_2),
            nn.ReLU(),
            nn.Linear(self.hidden_channels_2, self.out_channels)
            # Should have added: nn.ReLU()?
        )

    def forward(self, x):
        # Input: (N, C, H, W)
        x = x.permute(0, 2, 3, 1)
        # Now permuted to (N, H, W, C) - The linear layers now work on the channels dimension
        x = self.main(x).permute(0, 3, 1, 2)
        # After the linear layers, the output is now permuted back to (N, C, H, W)
        return x


class UNet(nn.Module):
    """`UNet` class is based on https://arxiv.org/abs/1505.04597

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).

    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the transpose convolution (specified by upmode='transpose')
    """

    valid = False
    pad = 0
    fow = [192, 192]
    dim = 2
    type = "seg"
    stride = 1
    increase_fow = 16

    def __init__(
        self,
        n_classes=2,
        in_channels=1,
        meta_in_channels=0,
        late_meta_inject=False,
        depth=5,
        start_filts=64,
        up_mode="transpose",
        merge_mode="concat",
    ):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet, self).__init__()

        if up_mode in ("transpose", "upsample"):
            self.up_mode = up_mode
        else:
            raise ValueError(
                '"{}" is not a valid mode for '
                'upsampling. Only "transpose" and '
                '"upsample" are allowed.'.format(up_mode)
            )

        if merge_mode in ("concat", "add"):
            self.merge_mode = merge_mode
        else:
            raise ValueError(
                '"{}" is not a valid mode for'
                "merging up and down paths. "
                'Only "concat" and '
                '"add" are allowed.'.format(up_mode)
            )

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == "upsample" and self.merge_mode == "add":
            raise ValueError(
                'up_mode "upsample" is incompatible '
                'with merge_mode "add" at the moment '
                "because it doesn't make sense to use "
                "nearest neighbour to reduce "
                "depth channels (by half)."
            )

        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self.n_classes = n_classes
        self.meta_in_channels = meta_in_channels

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2**i)
            pooling = True if i < depth - 1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode, merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        # add the list of modules to current module
        self.down_convs = nn.Sequential(*self.down_convs)
        self.up_convs = nn.Sequential(*self.up_convs)

        if late_meta_inject == False:
            self.conv_final = conv1x1(outs, n_classes)
        else:
            self.conv_final = conv1x1(outs + self.meta_in_channels, n_classes)
            self.post_processing_weights = MetaPostProcessing(
                in_channels=self.meta_in_channels, out_channels=1
            )

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            # nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            print(i, m)
            self.weight_init(m)


class UNet_Baseline(UNet):
    def __init__(
        self,
        n_classes,
        in_channels,
        meta_in_channels=0,
        late_meta_inject=False,
        depth=5,
        start_filts=64,
        up_mode="transpose",
        merge_mode="concat",
    ):
        super().__init__(
            n_classes,
            in_channels,
            meta_in_channels,
            late_meta_inject,
            depth,
            start_filts,
            up_mode,
            merge_mode,
        )

    def forward(self, x):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        return x


class UNet_LateMetInject(UNet):
    def __init__(
        self,
        n_classes,
        in_channels,
        meta_in_channels,
        late_meta_inject=True,
        depth=5,
        start_filts=64,
        up_mode="transpose",
        merge_mode="concat",
    ):
        super().__init__(
            n_classes,
            in_channels,
            meta_in_channels,
            late_meta_inject,
            depth,
            start_filts,
            up_mode,
            merge_mode,
        )

        # Line Added 15/04 at 10 pm
        self.conv_final = conv1x1(65, 3)

    def forward(self, x, meta_tensor):
        encoder_outs = []

        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.

        x = self.conv_final(
            torch.cat((x, self.post_processing_weights(meta_tensor)), 1)
        )

        return x


if __name__ == "__main__":
    pass
