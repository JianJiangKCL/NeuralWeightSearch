from .NWS import *


class NwsBasicBlock(nn.Module):
    """Nws Block for resnet 18 and resnet 34

    """
    expansion = 1

    def __init__(self, n_emb, in_channels, out_channels, stride=1, gs=1, sc_gs=1):
        super().__init__()
        # residual function
        self.residual_function = nn.Sequential(
            Nws3x3(n_emb, in_channels, out_channels, stride=stride, padding=1, gs=gs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            Nws3x3(n_emb, out_channels, out_channels * NwsBasicBlock.expansion, padding=1, gs=gs),
            nn.BatchNorm2d(out_channels * NwsBasicBlock.expansion)
        )
        # shortcut is also NWSed
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != NwsBasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                Nws1x1(n_emb, in_channels, out_channels * NwsBasicBlock.expansion, stride=stride, gs=sc_gs),
                nn.BatchNorm2d(out_channels * NwsBasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class NwsResNet(nn.Module):

    def __init__(self, n_emb, block, num_block, num_classes, gs, sc_gs, last_gs, outplanes=64):
        super().__init__()

        self.inplanes = outplanes
        self.n_emb = n_emb
        self.block = block
        self.last_gs = last_gs
        self.outplanes = outplanes
        self.block = block

        self.conv1 = nn.Sequential(
            Nws7x7(n_emb, 3, self.inplanes, stride=2, padding=3, gs=gs),
            # Nws3x3(n_emb, 3, 64, stride=1, padding=1, gs=gs),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True))
        # maxpool?
        self.conv2_x = self._make_layer(block, outplanes, num_block[0], 1, n_emb, gs, sc_gs)
        # after conv_3, the resolution is 16,16
        self.conv3_x = self._make_layer(block, outplanes * 2, num_block[1], 2, n_emb, gs, sc_gs)
        self.conv4_x = self._make_layer(block, outplanes * 4, num_block[2], 2, n_emb, gs, sc_gs)
        self.conv5_x = self._make_layer(block, outplanes * 8, num_block[3], 2, n_emb, gs, sc_gs)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # we replace FC layer with 1x1 NwsConv layer
        self.last_layer = Nws1x1(n_emb, outplanes * 8 * block.expansion, num_classes, gs=last_gs)

    def initial_conv(self):
        for module in self.modules():
            if isinstance(module, NwsConv):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

    def reset_last_layer(self, num_classes, finetune=True):
        print('reset last layer')
        if finetune:
            # this part need to be very careful
            previous_num_class = self.last_layer.weight.data.size()[0]
            # when previous num_class is bigger than current
            if previous_num_class >= num_classes:
                self.last_layer.weight.data = self.last_layer.weight.data[:num_classes, :]
            else:
                weight = Parameter(torch.empty((num_classes, self.outplanes * 8 * self.block.expansion, 1, 1)))
                torch.nn.init.kaiming_normal(weight)
                # when previous num_class is smaller than current
                weight.data[:previous_num_class, :] = self.last_layer.weight.data
                self.last_layer.weight = weight

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
        return embed_indices, unique_indices_layers

    def _make_layer(self, block, out_channels, num_blocks, stride, n_emb, gs, sc_gs):
        strides = [stride] + [1] * (num_blocks - 1)
        block_ret = nn.Sequential()
        cnt = 0
        for stride in strides:
            block_ret.add_module(f'NwsRes{cnt}', block(n_emb, self.inplanes, out_channels, stride, gs, sc_gs))
            self.inplanes = out_channels * block.expansion
            cnt += 1
        return block_ret

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = self.last_layer(output).view(output.size(0), -1)
        return output

    def get_summed_diff(self):
        w_diffs = []
        for module in self.modules():
            # this is used to update the model weights
            if isinstance(module, NwsConv):
                w_diffs.append(module.diff)
        summed_diff = sum(w_diffs)
        return summed_diff

    def update_qtz(self) -> None:
        for m in self.modules():
            if isinstance(m, Quantize):
                m.update()
                m.zero_buffer()

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


def Nws_resnet18(n_emb, num_classes, gs, last_gs=1, **kwargs):
    model = NwsResNet(n_emb, NwsBasicBlock, [2, 2, 2, 2], num_classes, gs, sc_gs=1, last_gs=last_gs, **kwargs)
    return model


def Nws_resnet34(n_emb, num_classes, gs, last_gs=1, **kwargs):
    model = NwsResNet(n_emb, NwsBasicBlock, [3, 4, 6, 3], num_classes, gs, sc_gs=1, last_gs=last_gs, **kwargs)
    return model

