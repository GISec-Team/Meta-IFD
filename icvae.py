import torch
import torch.nn as nn
import torch.nn.functional as F


class ICVAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, edge_type_size,
                 conditional=True, conditional_size=0):
        super().__init__()

        if conditional:
            assert conditional_size > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list
        assert type(edge_type_size) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, conditional_size, edge_type_size)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, conditional_size, edge_type_size)

    def forward(self, x, c, edge_type):
        means, log_var = self.encoder(x, c, edge_type)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c, edge_type)

        return recon_x, means, log_var, z

    def reparameterize(self, means, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return means + eps * std

    def inference(self, z, c, edge_type):
        recon_x = self.decoder(z, c, edge_type)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, conditional_size, edge_type_size):
        super().__init__()

        self.conditional = conditional

        layer_sizes[0] += conditional_size + len(edge_type_size)

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, 32))
            self.MLP.add_module(name="L2{:d}".format(i), module=nn.Linear(32, 64))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.Tanh())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c, edge_type):
        x = torch.cat((x, c), dim=-1)
        x = torch.cat((x, edge_type), dim=-1)
        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, conditional_size, edge_type_size):
        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        layer_sizes[0] = latent_size + conditional_size + len(edge_type_size)

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, 32))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.Tanh())
            self.MLP.add_module(name="L_out{:d}".format(i), module=nn.Linear(32, out_size))

    def forward(self, z, c, edge_type):
        z = torch.cat((z, c), dim=-1)
        z = torch.cat((z, edge_type), dim=-1)
        x = self.MLP(z)
        rec_x = F.sigmoid(x)

        return rec_x
