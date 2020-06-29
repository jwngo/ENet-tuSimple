import torch.nn as nn
import torch
import torch.nn.functional as F

class InitialBlock(nn.Module):
    # TODO bias is not supposed to be true. It is supposed to return the bias
    def __init__(self,
                in_channels,
                out_channels,
                bias=True,
                relu=False):
        super().__init__()
            
        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU
            
        self.main_branch = nn.Conv2d(
            in_channels,
            out_channels - 3,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=bias
        )

        self.ext_branch = nn.MaxPool2d((2,2), (2,2))

        self.batch_norm = nn.BatchNorm2d(out_channels, 0.001, 0.1, True)

        self.out_activation = activation() 

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)

        # Concat Table; Concatenate branches
        # TODO make sure this concattable is working as intended
        # TODO this is supposedly both ConcatTable and JoinTable
        out = torch.cat((main,ext), 1)

        # SpatialBatchNormalization equivalent
        out = self.batch_norm(out)

        # PReLU activation
        out = self.out_activation(out)

        return out

class DownsamplingBottleneck(nn.Module):
    # TODO bias is not supposed to be True. It is supposed to return the bias
    def __init__(
        self,
        in_channels,
        out_channels,
        internal_ratio=4,
        return_indices=False,
        dropout_prob=0,
        bias=False,
        relu=False):
        super.__init__()
    
        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU
        self.out_activation = activation()
        if internal_ratio > 0:
            internal_channels = in_channels // internal_ratio
        else:
            internal_channels = in_channels

        # Main branch - max pooling followed by padding
        self.main_max1 = nn.MaxPool2d(
            (2,2),
            (2,2),
            return_indices=return_indices
        )
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                internal_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=bias
            ), 
            nn.BatchNorm2d(eps=0.001, momentum=0.1),
            activation()
        )
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                internal_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias
            ),
            nn.BatchNorm2d(eps=0.001, momentum=0.1),
            activation()
        )
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=bias
            ),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1),
            activation()
        )
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
    
    def forward(self, x): 
        # Main branch shortcut
        if self.return_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Main branch channel padding
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w)

        if main.is_cuda:
            padding = padding.cuda()
        
        # Concatenate (apply the padding)
        main = torch.cat((main,padding), 1)

        out = main + ext
        out = self.out_activation(out)
        return out, max_indices
        
class RegularBottleneck(nn.Module):
    
    def __init__(
        self,
        channels,
        internal_ratio=4,
        kernel_size=3,
        padding=0,
        dilation=1,
        asymmetric=False,
        dropout_prob=0,
        bias=False,
        relu=False
        ):
        super().__init__()
        if internal_ratio >= 0:
            internal_channels = channels // internal_ratio
        else:
            internal_channels = channels

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU
        
        # Main branch - shortcut connection
        
        # Extension branch 

        # 1x1 projection convolution
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                channels,
                internal_channels,
                kernel_size=1,
                stride=1,
                bias=bias,
            ),
            nn.BatchNorm2d(internal_channels, eps=0.001, momentum=0.1),
            activation()
        )
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                internal_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            ),
            nn.BatchNorm2d(eps=0.001, momentum=0.1),
            activation()
        )
    