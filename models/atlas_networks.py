import torch
import torch.nn.functional as F
from torch import nn
from torch import optim

from models.networks import ListModule


class HyperNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.z_size = args.zdim
        self.use_bias = True
        self.relu_slope = 0.2
        # target network layers out channels
        target_network_out_ch = [5] + [16, 32, 64, 128] + [3]
        target_network_use_bias = int(True)

        self.model = nn.Sequential(
            nn.Linear(in_features=self.z_size, out_features=64, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=64, out_features=128, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=128, out_features=512, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=512, out_features=1024, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=2048, bias=self.use_bias),
        )

        output = [
            nn.Linear(2048, (target_network_out_ch[x - 1] + target_network_use_bias) * target_network_out_ch[x],
                      bias=True).cuda()
            for x in range(1, len(target_network_out_ch))
        ]

        self.output = ListModule(*output)

    def make_optimizer(self, args):
        def _get_opt_(params):
            if args.optimizer == 'adam':
                optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta1, args.beta2),
                                       weight_decay=args.weight_decay)
            elif args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
            else:
                assert 0, "args.optimizer should be either 'adam' or 'sgd'"
            return optimizer
        opt = _get_opt_(list(self.parameters()))
        return opt

    def forward(self, x):
        output = self.model(x)
        return torch.cat([target_network_layer(output) for target_network_layer in self.output], 1)


class TargetNetwork(nn.Module):
    def __init__(self, zdim, weights):
        super().__init__()

        self.z_size = zdim
        self.use_bias = True
        # target network layers out channels
        out_ch = [16, 32, 64, 128]

        layer_data, split_index = self._get_layer_data(start_index=0, end_index=out_ch[0] * 5,
                                                       shape=(out_ch[0], 5), weights=weights)
        self.layers = {"1": layer_data}

        for x in range(1, len(out_ch)):
            layer_data, split_index = self._get_layer_data(start_index=split_index,
                                                           end_index=split_index + (out_ch[x - 1] * out_ch[x]),
                                                           shape=(out_ch[x], out_ch[x - 1]), weights=weights)
            self.layers[str(x + 1)] = layer_data

        layer_data, split_index = self._get_layer_data(start_index=split_index,
                                                       end_index=split_index + (out_ch[-1] * 3),
                                                       shape=(3, out_ch[-1]), weights=weights)
        self.output = layer_data
        self.activation = torch.nn.ReLU()
        assert split_index == len(weights)

    def forward(self, x):
        for layer_index in self.layers:
            x = torch.mm(x, torch.transpose(self.layers[layer_index]["weight"], 0, 1))
            if self.use_bias:
                assert "bias" in self.layers[layer_index]
                x = x + self.layers[layer_index]["bias"]
            x = self.activation(x)
        return torch.mm(x, torch.transpose(self.output["weight"], 0, 1)) + self.output.get("bias", 0)

    def _get_layer_data(self, start_index, end_index, shape, weights):
        layer_data = {"weight": weights[start_index:end_index].view(shape[0], shape[1])}
        if self.use_bias:
            layer_data["bias"] = weights[end_index:end_index + shape[0]]
            end_index = end_index + shape[0]
        return layer_data, end_index