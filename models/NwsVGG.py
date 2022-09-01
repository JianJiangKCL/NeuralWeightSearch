from typing import Union, List, Dict, Any, cast
from models.NWS_v2 import *

__all__ = [
    "NwsVGG",
    "Nwsvgg16_bn"
]


class NwsVGG(nn.Module):
    def __init__(
        self, n_emb: int, features: nn.Module,  num_classes: int = 1000, dropout: float = 0.2,

    ) -> None:
        super().__init__()
        self.features = features

        self.dropout = nn.Dropout(dropout)
        # we replace last 3 fc layers with 1x1 NWSed conv layer
        self.last_layer = Nws1x1(n_emb, 512, num_classes)

        for m in self.modules():
            if isinstance(m, NwsConv):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                # if m.bias is not None:
                #     nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # x = self.avgpool(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = self.dropout(x)
        x = self.last_layer(x)
        x = x.view(x.size(0), -1)
        return x

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
        if finetune:
            # this part need to be very careful
            previous_num_class = self.last_layer.weight.data.size()[0]
            # when previous num_class is bigger than current
            if previous_num_class >= num_classes:
                self.last_layer.weight.data = self.last_layer.weight.data[:num_classes, :]

            else:
                weight = Parameter(torch.empty((num_classes, 512, 1, 1)))
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


def make_layers(cfg, batch_norm=True, n_emb=256):
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = Nws3x3(n_emb, in_channels, v, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(n_emb, cfg: str, batch_norm: bool,  **kwargs: Any) -> NwsVGG:

    model = NwsVGG(n_emb, make_layers(cfgs[cfg], batch_norm=batch_norm),  **kwargs)
    return model


def Nwsvgg16_bn(n_emb,  **kwargs: Any) -> NwsVGG:

    return _vgg(n_emb, "D", True,  **kwargs)
