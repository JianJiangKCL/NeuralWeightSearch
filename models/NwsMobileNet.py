import torch
from torch import nn
from torch import Tensor
from .NWS_v2 import *
from typing import Callable, Any, Optional, List
from torchvision.models import mobilenet_v2


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
		n_emb,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if kernel_size == 1:
            Nwsconv = Nws1x1
        elif kernel_size == 3:
            Nwsconv = Nws3x3
        else:
            raise NotImplementedError
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super().__init__(
            Nwsconv(n_emb, in_planes, out_planes, stride=stride, padding=padding, groups=groups),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes


# necessary for backwards compatibility
ConvBNReLU = ConvBNActivation


class InvertedResidual(nn.Module):
    def __init__(
        self,
        n_emb,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(n_emb, inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(n_emb, hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            Nws1x1(n_emb, hidden_dim, oup, stride=1, padding=0),

            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class NwsMobileNetV2(nn.Module):
    def __init__(
        self,
        n_emb: int=512,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(NwsMobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [ConvBNReLU(n_emb, 3, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(n_emb, input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(n_emb, input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        # self.dropout = nn.Dropout(0.2)
        self.last_layer = Nws1x1(n_emb, self.last_channel, num_classes)
        # self.classifier = nn.Sequential(
        #     nn.Dropout(0.2),
        #     # nn.Linear(self.last_channel, num_classes),
        #     Nws1x1(n_emb, self.last_channel, num_classes)
        # )

        # weight initialization
        for m in self.modules():
            if isinstance(m, NwsConv):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                # if m.bias is not None:
                #     nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))

        # x = torch.flatten(x, 1)
        # x = self.dropout(x)
        x = self.last_layer(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def get_summed_diff(self):
        w_diffs = []
        for module in self.modules():
            # this is used to update the model weights
            if isinstance(module, NwsConv):
                w_diffs.append(module.diff)
        summed_diff = sum(w_diffs)
        return summed_diff

    def reset_last_layer(self, num_classes, finetune=True):
        print('reset last layer')
        # tt_layer = copy.deepcopy(self.last_layer)
        # tmp_layer = Nws1x1(self.n_emb, self.outplanes * 8 * self.block.expansion, num_classes, gs=self.last_gs)
        if finetune:

            # this part need to be very careful
            previous_num_class = self.last_layer.weight.data.size()[0]
            # when previous num_class is bigger than current
            if previous_num_class >= num_classes:
                self.last_layer.weight.data = self.last_layer.weight.data[:num_classes, :]

            else:
                weight = Parameter(torch.empty((num_classes, self.last_channel, 1, 1)))
                torch.nn.init.kaiming_normal(weight)
                # when previous num_class is smaller than current
                weight.data[:previous_num_class, :] = self.last_layer.weight.data
                self.last_layer.weight = weight

    def update_qtz(self) -> None:
        for m in self.modules():
            if isinstance(m, Quantize):
                m.update()
                m.zero_buffer()

    def get_num_emb_layers(self):
        num_emb_layers = []
        for m in self.modules():
            if isinstance(m, NwsConv):
                num_emb_layers.append(m.qtz.n_embed)
        return num_emb_layers

    # to save codes in format of numpy or lzma, we have to use the flattened version
    # because the length of codes in different layers are not the same
    def encode_weights(self, flatten=True):
        if flatten:
            embed_indices = torch.LongTensor([])
        else:
            embed_indices = []
        unique_indices_layers = []
        for m in self.modules():
            if isinstance(m, NwsConv):
                embed_ind = m.encode_weights()
                unique_ind = torch.unique(embed_ind, sorted=True)
                unique_indices_layers.append(unique_ind)
                if flatten:
                    embed_indices = torch.cat((embed_indices, embed_ind), dim=0)
                else:
                    embed_indices.append(embed_ind)
        # unique_indices_layers cannot be stacked (Tensored) because it has different lengths
        # unique_indices_layers = torch.stack(unique_indices_layers)
        return embed_indices, unique_indices_layers

    def load_qtz(self, qtzs) -> None:
        cnt = 0
        for m in self.modules():
            if isinstance(m, Quantize):
                m.embed.data = qtzs[cnt].embed.data
                cnt += 1

    def eval_qtz(self):
        for m in self.modules():
            if isinstance(m, Quantize):
                m.eval()

    def zero_buffer(self) -> None:
        for m in self.modules():
            if isinstance(m, Quantize):
                m.zero_buffer()
            if isinstance(m, NwsConv):
                m.diff = 0

    def eval_mode(self, eval_mode):
        for m in self.modules():
            if isinstance(m, Nwsxxx):
                m.set_eval(eval_mode)

    def reset_nembs(self, nembs):
        cnt = 0
        for module in self.modules():
            if isinstance(module, Quantize):
                module.reset_nemb(nembs[cnt])
                cnt += 1

