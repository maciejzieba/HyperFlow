import torch
import numpy as np
import torch.nn.functional as F
from torch import optim
from torch import nn
from models.flow import get_latent_cnf
from models.flow import get_hyper_cnf
from utils import truncated_normal, reduce_tensor, standard_normal_logprob, log_normal_logprob


class Encoder(nn.Module):
    def __init__(self, zdim, input_dim=3, use_deterministic_encoder=False):
        super(Encoder, self).__init__()
        self.use_deterministic_encoder = use_deterministic_encoder
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        if self.use_deterministic_encoder:
            self.fc1 = nn.Linear(512, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc_bn1 = nn.BatchNorm1d(256)
            self.fc_bn2 = nn.BatchNorm1d(128)
            self.fc3 = nn.Linear(128, zdim)
        else:
            # Mapping to [c], cmean
            self.fc1_m = nn.Linear(512, 256)
            self.fc2_m = nn.Linear(256, 128)
            self.fc3_m = nn.Linear(128, zdim)
            self.fc_bn1_m = nn.BatchNorm1d(256)
            self.fc_bn2_m = nn.BatchNorm1d(128)

            # Mapping to [c], cmean
            self.fc1_v = nn.Linear(512, 256)
            self.fc2_v = nn.Linear(256, 128)
            self.fc3_v = nn.Linear(128, zdim)
            self.fc_bn1_v = nn.BatchNorm1d(256)
            self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        if self.use_deterministic_encoder:
            ms = F.relu(self.fc_bn1(self.fc1(x)))
            ms = F.relu(self.fc_bn2(self.fc2(ms)))
            ms = self.fc3(ms)
            m, v = ms, 0
        else:
            m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
            m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
            m = self.fc3_m(m)
            v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
            v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
            v = self.fc3_v(v)

        return m, v


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


########################

# Model
class HyperPointFlow(nn.Module):
    def __init__(self, args):
        super(HyperPointFlow, self).__init__()
        self.input_dim = args.input_dim
        self.zdim = args.zdim
        self.use_latent_flow = args.use_latent_flow
        self.use_deterministic_encoder = args.use_deterministic_encoder
        self.prior_weight = args.prior_weight
        self.recon_weight = args.recon_weight
        self.entropy_weight = args.entropy_weight
        self.distributed = args.distributed
        self.truncate_std = None
        self.encoder = Encoder(
                zdim=args.zdim, input_dim=args.input_dim,
                use_deterministic_encoder=args.use_deterministic_encoder)
        self.latent_cnf = get_latent_cnf(args) if args.use_latent_flow else nn.Sequential()
        self.hyper = HyperFlowNetwork(args)
        self.args = args
        self.point_cnf = get_hyper_cnf(self.args)
        self.gpu = args.gpu
        self.use_sphere_dist = args.use_sphere_dist
        self.mu = 0.0
        self.var = args.start_var

    def update_var(self, step=0.001, min_val=0.001):
            self.var = max(min_val, self.var - step)

    @staticmethod
    def sample_gaussian(size, truncate_std=None, gpu=None):
        y = torch.randn(*size).float()
        y = y if gpu is None else y.cuda(gpu)
        if truncate_std is not None:
            truncated_normal(y, mean=0, std=1, trunc_std=truncate_std)
        return y

    def sample_lognormal(self, size, gpu=None):
        x = torch.randn(*size).float()
        x_norm = torch.norm(x, dim=2).unsqueeze(2).repeat(1, 1, 3)
        r = torch.randn(*(size[0], size[1])).float().unsqueeze(2).repeat(1, 1, 3)
        x = x/x_norm
        y = torch.exp(self.mu + np.sqrt(self.var)*r)*x
        y = y if gpu is None else y.cuda(gpu)
        return y


    @staticmethod
    def reparameterize_gaussian(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.size()).to(mean)
        return mean + std * eps

    @staticmethod
    def gaussian_entropy(logvar):
        const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2))
        ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
        return ent

    def multi_gpu_wrapper(self, f):
        self.encoder = f(self.encoder)
        self.hyper = f(self.hyper)
        self.latent_cnf = f(self.latent_cnf)
        self.point_cnf = f(self.point_cnf)

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
        opt = _get_opt_(list(self.encoder.parameters()) + list(list(self.latent_cnf.parameters())) +
                        list(self.hyper.parameters()) + list(self.point_cnf.parameters()))
        return opt

    def forward(self, x, opt, step, writer=None):
        opt.zero_grad()
        batch_size = x.size(0)
        num_points = x.size(1)
        z_mu, z_sigma = self.encoder(x)
        if self.use_deterministic_encoder:
            z = z_mu + 0 * z_sigma
        else:
            z = self.reparameterize_gaussian(z_mu, z_sigma)

        # Compute H[Q(z|X)]
        if self.use_deterministic_encoder:
            entropy = torch.zeros(batch_size).to(z)
        else:
            entropy = self.gaussian_entropy(z_sigma)

        # Compute the prior probability P(z)
        if self.use_latent_flow:
            w, delta_log_pw = self.latent_cnf(z, None, torch.zeros(batch_size, 1).to(z))
            log_pw = standard_normal_logprob(w).view(batch_size, -1).sum(1, keepdim=True)
            delta_log_pw = delta_log_pw.view(batch_size, 1)
            log_pz = log_pw - delta_log_pw
        else:
            log_pz = torch.zeros(batch_size, 1).to(z)

        # Compute the reconstruction likelihood P(X|z)
        z_new = z.view(*z.size())
        z_new = z_new + (log_pz * 0.).mean()
        target_networks_weights = self.hyper(z_new)

        # Loss
        y, delta_log_py = self.point_cnf(x, target_networks_weights, torch.zeros(batch_size, num_points, 1).to(x))
        if self.use_sphere_dist:
            log_py = log_normal_logprob(y, self.mu, self.var).sum(1)
        else:
            log_py = standard_normal_logprob(y).view(batch_size, -1).sum(1, keepdim=True)
        delta_log_py = delta_log_py.view(batch_size, num_points, 1).sum(1)
        log_px = log_py - delta_log_py

        entropy_loss = -entropy.mean() * self.entropy_weight
        recon_loss = -log_px.mean() * self.recon_weight
        prior_loss = -log_pz.mean() * self.prior_weight
        loss = entropy_loss + prior_loss + recon_loss

        loss.backward()
        opt.step()

        # LOGGING (after the training)
        if self.distributed:
            entropy_log = reduce_tensor(entropy.mean())
            recon = reduce_tensor(-log_px.mean())
            prior = reduce_tensor(-log_pz.mean())
        else:
            entropy_log = entropy.mean()
            recon = -log_px.sum()
            prior = -log_pz.mean()

        recon_nats = recon / float(x.size(1)*x.size(2))
        prior_nats = prior / float(self.zdim)

        if writer is not None:
            writer.add_scalar('train/entropy', entropy_log, step)
            writer.add_scalar('train/prior', prior, step)
            writer.add_scalar('train/prior(nats)', prior_nats, step)
            writer.add_scalar('train/recon', recon, step)
            writer.add_scalar('train/recon(nats)', recon_nats, step)

        return {
            'entropy': entropy_log.cpu().detach().item()
            if not isinstance(entropy_log, float) else entropy_log,
            'prior_nats': prior_nats,
            'recon_nats': recon_nats,
        }

    def generate_points(self, size, low=-1, high=1):
        while True:
            points = torch.zeros([size[0] * 3, *size[1:]]).uniform_(low, high)
            points = points[torch.norm(points, dim=1) < 1]
            if points.shape[0] >= size[0]:
                return points[:size[0]]

    def encode(self, x):
        z_mu, z_sigma = self.encoder(x)
        if self.use_deterministic_encoder:
            return z_mu
        else:
            return self.reparameterize_gaussian(z_mu, z_sigma)

    def decode(self, z, num_points, truncate_std=None, y=None):
        # transform points from the prior to a point cloud, conditioned on a shape code
        target_networks_weights = self.hyper(z)
        if y is None:
            if self.use_sphere_dist:
                y = self.sample_lognormal((z.size(0), num_points, self.input_dim), self.gpu)
            else:
                y = self.sample_gaussian((z.size(0), num_points, self.input_dim), truncate_std, self.gpu)
        x = self.point_cnf(y, target_networks_weights, reverse=True).view(*y.size())
        return y, x

    def sample(self, batch_size, num_points, truncate_std=None, truncate_std_latent=None, gpu=None,
               w=None, y=None):
        assert self.use_latent_flow, "Sampling requires `self.use_latent_flow` to be True."
        # Generate the shape code from the prior
        if w is None:
            w = self.sample_gaussian((batch_size, self.zdim), truncate_std_latent, gpu=gpu).cuda()
        z = self.latent_cnf(w, None, reverse=True).view(*w.size())
        # Sample points conditioned on the shape code
        _, x = self.decode(z, num_points, truncate_std, y=y)
        return z, x

    def reconstruct(self, x, num_points=None, truncate_std=None, y=None):
        num_points = x.size(1) if num_points is None else num_points
        z = self.encode(x)
        _, x = self.decode(z, num_points, truncate_std, y=y)
        return x


class HyperFlowNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.z_size = args.zdim
        self.use_bias = True
        self.relu_slope = 0.2

        self.n_out = 2048

        dims = tuple(map(int, args.hyper_dims.split("-")))
        self.n_out = dims[-1]
        model = []
        for k in range(len(dims)):
            if k == 0:
                model.append(nn.Linear(in_features=self.z_size, out_features=dims[k], bias=self.use_bias))
            else:
                model.append(nn.Linear(in_features=dims[k-1], out_features=dims[k], bias=self.use_bias))
            if k < len(dims) - 1:
                model.append(nn.ReLU(inplace=True))

        self.model = nn.Sequential(*model)

        output = []
        dims = tuple(map(int, args.dims.split("-")))
        for k in range(len(dims)):
            if k == 0:
                output.append(nn.Linear(self.n_out, args.input_dim * dims[k], bias=True))
            else:
                output.append(nn.Linear(self.n_out, dims[k - 1] * dims[k], bias=True))
            #bias
            output.append(nn.Linear(self.n_out, dims[k], bias=True))
            #scaling
            output.append(nn.Linear(self.n_out, dims[k], bias=True))
            output.append(nn.Linear(self.n_out, dims[k], bias=True))
            #shift
            output.append(nn.Linear(self.n_out, dims[k], bias=True))

        output.append(nn.Linear(self.n_out, dims[-1] * args.input_dim, bias=True))
        # bias
        output.append(nn.Linear(self.n_out, args.input_dim, bias=True))
        # scaling
        output.append(nn.Linear(self.n_out, args.input_dim, bias=True))
        output.append(nn.Linear(self.n_out, args.input_dim, bias=True))
        # shift
        output.append(nn.Linear(self.n_out, args.input_dim, bias=True))

        self.output = ListModule(*output)

    def forward(self, x):
        output = self.model(x)
        multi_outputs = []
        for j, target_network_layer in enumerate(self.output):
            multi_outputs.append(target_network_layer(output))
        multi_outputs = torch.cat(multi_outputs, dim=1)
        return multi_outputs
