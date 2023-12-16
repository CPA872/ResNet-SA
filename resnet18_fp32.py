import torch
import torch.nn as nn
import collections
import math
import pynq
import time
import numpy as np

from torch import Tensor
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torchvision.models import ResNet18_Weights

from itertools import repeat
from typing import Any, Callable, List, Optional, Type, Union, Tuple

from im2col import im2col_indices


# Module level declarations 
XCLBIN_PATH = "/mnt/sdb1/dpan/gemm_hls/build13_fp32_m64_n8_512/MatrixMultiplication_hw.xclbin"
ol = pynq.Overlay(XCLBIN_PATH)
mm_kernel = ol.MatrixMultiplicationKernel_1
print(f"using kernel: {mm_kernel}")


# Protected functions in nn
def _reverse_repeat_tuple(t, n):
    r"""Reverse the order of `t` and repeat each element for `n` times.

    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple(x for x in reversed(t) for _ in range(n))


def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse

_pair = _ntuple(2, "_pair")


def fpga_call(Ah, Bh):  # h for host
    MEM_GRANULARITY = 16
    # print(Ah.size, Bh.size)

    N, K = Ah.shape
    K, M = Bh.shape
    
    # padded device matrix dimensions
    # d for device
    Nd = N if N % MEM_GRANULARITY == 0 else (N // MEM_GRANULARITY + 1) * MEM_GRANULARITY
    Kd = K if K % MEM_GRANULARITY == 0 else (K // MEM_GRANULARITY + 1) * MEM_GRANULARITY
    Md = M if M % MEM_GRANULARITY == 0 else (M // MEM_GRANULARITY + 1) * MEM_GRANULARITY
    
    # print(f"FPGA call: A: {Nd}x{Kd}, B: {Kd}x{Md}, C:{Nd}x{Md}, pad={(0, Nd - N, 0, Kd - K)}")
    
    start = time.time()
    Ah = F.pad(Ah, (0, Kd - K, 0, Nd - N))
    Bh = F.pad(Bh, (0, Md - M, 0, Kd - K))
    padding_time = time.time() - start
    # print(Ah.shape, Bh.shape)
    
    start = time.time()
    Ad = pynq.allocate(shape=(Nd, Kd), dtype=np.float32)
    Bd = pynq.allocate(shape=(Kd, Md), dtype=np.float32)
    Cd = pynq.allocate(shape=(Nd, Md), dtype=np.float32)
    buffer_allocation_time = time.time() - start

    Ad[:] = Ah.detach().numpy()[:]
    Bd[:] = Bh.detach().numpy()[:]

    # print(A.shape, B.shape, C.shape)

    start = time.time()
    Ad.sync_to_device()
    Bd.sync_to_device()
    Cd.sync_to_device()
    buffer_copy_time = time.time() - start

    start = time.time()
    mm_kernel.call(Ad, Bd, Cd, Nd, Kd, Md)
    exec_time = time.time() - start

    # Cd = Ah @ Bh
    start = time.time()
    Cd.sync_from_device()
    result_copy_time = time.time() - start
    flops = 2 * Nd * Kd * Md / exec_time / (1 << 30)
    effective_flops = 2 * N * K * M / exec_time / (1 << 30)
    util = effective_flops / flops
    
    # print(f"Report: flops={flops:.2f}, eff_flops={effective_flops:.2f}, util={util:.2f}")
    
    Ad.freebuffer()
    Bd.freebuffer()
    # Cd.freebuffer()
    
    # print(padding_time, buffer_allocation_time, buffer_copy_time, exec_time, result_copy_time, sep=", ")
    return torch.Tensor(Cd[:N, :M])


# Conv2d with im2col
def conv2d_im2col(input, weight, bias=None, stride=1, padding=0, dilation=0, groups=1) -> Tensor:
    padding = _pair(padding) if isinstance(padding, int) else padding
    stride  = _pair(stride) if isinstance(stride, int) else stride

    # print(f"input: {input.shape}, weight: {weight.shape}")
    c_out = weight.shape[0]
    kernel_height = weight.shape[2]
    kernel_width  = weight.shape[3]
    N, C, H, W = input.shape
    out_height = int((H + 2 * padding[0] - kernel_height) / stride[0] + 1)
    out_width = int((W + 2 * padding[1] - kernel_width) / stride[1] + 1)

    # im2col
    x_cols = im2col_indices(input.detach().numpy(), kernel_height, kernel_width, 
                            padding=padding, stride=stride)
    
    # Matrix multiplication
    
    A = weight.reshape((c_out, -1))
    B = torch.tensor(x_cols)
    # print(A.shape, B.shape)
    
    res = weight.reshape((c_out, -1)).matmul(torch.tensor(x_cols))    # [C_o, (H_o * W_o * N)]

    if bias is not None:
        res += bias.unsqueeze(1).repeat(1, out_height * out_width * N)

    # Reshape & get output
    res = res.reshape(c_out, out_height, out_width, N).permute((3, 0, 1, 2))   # [C_o, H_o, W_o, N]
    # res = torch.transpose(res, 3, 0)    # [N, H, W, C]
    # res = torch.transpose(res, 1, 3)    # [N, C, W, H]
    # res = torch.transpose(res, 2, 3)    # [N, C, H, W]
    
    return res

def conv2d_im2col_fpga(input, weight, bias=None, stride=1, padding=0, dilation=0, groups=1) -> Tensor:
    start = time.time()
    padding = _pair(padding) if isinstance(padding, int) else padding
    stride  = _pair(stride) if isinstance(stride, int) else stride

    # print(f"input: {input.shape}, weight: {weight.shape}")
    c_out = weight.shape[0]
    kernel_height = weight.shape[2]
    kernel_width  = weight.shape[3]
    N, C, H, W = input.shape
    out_height = int((H + 2 * padding[0] - kernel_height) / stride[0] + 1)
    out_width = int((W + 2 * padding[1] - kernel_width) / stride[1] + 1)

    # im2col
    x_cols = im2col_indices(input.detach().numpy(), kernel_height, kernel_width, 
                            padding=padding, stride=stride)
    
    # Matrix multiplication
    
    A = weight.reshape((c_out, -1))
    B = torch.tensor(x_cols)
    # print(A.shape, B.shape)
    
    im2col_time = time.time() - start
    
    
    res = fpga_call(A, B)
    
    # res = weight.reshape((c_out, -1)).matmul(torch.tensor(x_cols))    # [C_o, (H_o * W_o * N)]
    start = time.time()

    if bias is not None:
        res += bias.unsqueeze(1).repeat(1, out_height * out_width * N)

    # Reshape & get output
    res = res.reshape(c_out, out_height, out_width, N).permute((3, 0, 1, 2))   # [C_o, H_o, W_o, N]
    
    col2im_time = time.time() - start
    # print(im2col_time, col2im_time, sep=", ")

    return res


# nn._ConvNd
class ConvNd(nn.Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:  # type: ignore[empty-body]
        ...

    in_channels: int
    _reversed_padding_repeated_twice: List[int]
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Union[str, Tuple[int, ...]]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...],
                 padding: Tuple[int, ...],
                 dilation: Tuple[int, ...],
                 transposed: bool,
                 output_padding: Tuple[int, ...],
                 groups: int,
                 bias: bool,
                 padding_mode: str,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    f"Invalid padding string {padding!r}, should be one of {valid_padding_strings}")
            if padding == 'same' and any(s != 1 for s in stride):
                raise ValueError("padding='same' is not supported for strided convolutions")

        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError(f"padding_mode must be one of {valid_padding_modes}, but got padding_mode='{padding_mode}'")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == 'same':
                for d, k, i in zip(dilation, kernel_size,
                                   range(len(kernel_size) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad)
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        if transposed:
            self.weight = Parameter(torch.empty(
                (in_channels, out_channels // groups, *kernel_size), **factory_kwargs))
        else:
            self.weight = Parameter(torch.empty(
                (out_channels, in_channels // groups, *kernel_size), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'


# nn.Conv2d with img2col
# Custom conv2d implementation with im2col and FPGA offloading
class MyConv2d(ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None,
    ) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return conv2d_im2col_fpga(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                        weight, bias, self.stride,
                        _pair(0), self.dilation, self.groups) 

        else:
            return conv2d_im2col_fpga(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> MyConv2d:
    """3x3 convolution with padding"""
    return MyConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> MyConv2d:
    """1x1 convolution"""
    return MyConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# Custom Linear layer implementation with FPGA offloading
class myLinear(nn.Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(myLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        print("nn.Linear:", input.shape, self.weight.shape)
        if self.bias is not None:   
            return fpga_call(input, self.weight.T) + self.bias
        else:
            return fpga_call(input, self.weight.T)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

# torchvision.ResNet
class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[BasicBlock],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:

        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = MyConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = myLinear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, MyConv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck) and m.bn3.weight is not None:
        #             nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
        #         elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
        #             nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[BasicBlock],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    block: Type[BasicBlock],
    layers: List[int],
    weights,
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    if weights is not None:
        num_classes = len(weights.meta["categories"])
        if "num_classes" in kwargs and kwargs["num_classes"] != num_classes:
            raise ValueError(f"The parameter 'num_classes' expected value {num_classes} \
                             but got {kwargs['num_classes']} instead.")
        else:
            kwargs["num_classes"] = num_classes

    model = ResNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


def myresnet18(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>.
    Derived from resnet18 in torchvision <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>.
    * Convolution layers are rewriten with im2col and supports acceleration with systolic array.
    * Model parameters remain unchanged and accept pretrained weights in torchvision.
    """
    weights = ResNet18_Weights.verify(weights)

    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)
