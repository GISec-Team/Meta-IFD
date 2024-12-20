import gc
import os.path as osp
import sys
import warnings
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from tqdm import trange
from dataset import Ponzi, Phish
from icvae import ICVAE
from utils import get_parser, one_hot, mkdir

warnings.filterwarnings("ignore")
exc_path = sys.path[0]


def loss_fn(recon_x, x, mean, log_var):
    x = torch.nn.functional.sigmoid(x)
    BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='mean')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return (BCE + KLD) / x.size(0)


def generated_generator_sixedges(args, data, device):
    x_list, c_list, category_list = [], [], []
    e_type_index = {}
    for index, e in enumerate(data.edge_types):
        e_type_index[e] = index
    for i in trange(len(data.edge_types)):
        head_node_type = data.edge_types[i][0]
        tail_node_type = data.edge_types[i][-1]
        head_node_index = data.edge_stores[i]['edge_index'][0]
        tail_node_index = data.edge_stores[i]['edge_index'][-1]
        c_list.append(data[head_node_type].x[head_node_index])
        x_list.append(data[tail_node_type].x[tail_node_index])
        edge_onehot = one_hot(torch.LongTensor([e_type_index[data.edge_types[i]]]), len(e_type_index),
                              dtype=float).repeat(
            len(head_node_index), 1)
        category_list.append(edge_onehot)
    features_x = np.vstack(x_list)
    features_c = np.vstack(c_list)
    features_e_type = np.vstack(category_list)
    del x_list
    del c_list
    del category_list
    gc.collect()
    features_x = torch.tensor(features_x, dtype=torch.float32)
    features_c = torch.tensor(features_c, dtype=torch.float32)
    features_e_type = torch.tensor(features_e_type, dtype=torch.float32)
    cvae_dataset = TensorDataset(features_x, features_c, features_e_type)
    cvae_dataset_sampler = RandomSampler(cvae_dataset)
    cvae_dataset_dataloader = DataLoader(cvae_dataset, sampler=cvae_dataset_sampler, batch_size=args.batch_size)

    # Pretrain
    cvae = ICVAE(encoder_layer_sizes=[data[target_node].x.shape[1], 64],
                 latent_size=50,
                 decoder_layer_sizes=[64, data[target_node].x.shape[1]],
                 edge_type_size=data.edge_types,
                 conditional_size=data[target_node].x.shape[1])
    cvae_optimizer = optim.Adam(cvae.parameters(), lr=0.01)
    cvae.to(device)

    for _ in trange(args.pretrain_epochs, desc='Run CVAE Train'):
        for _, (x, c, e) in enumerate(cvae_dataset_dataloader):
            cvae.train()
            x, c, e = x.to(device), c.to(device), e.to(device)
            recon_x, mean, log_var, _ = cvae(x, c, e)
            cvae_loss = loss_fn(recon_x, x, mean, log_var)
            cvae_optimizer.zero_grad()
            cvae_loss.backward()
            cvae_optimizer.step()
    return cvae


if __name__ == '__main__':
    args = get_parser()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    predict_fn = lambda output: output.max(1, keepdim=True)[1].detach().cpu()
    if 'Ponzi' in args.dataset:
        path = osp.join(osp.dirname(osp.realpath(__file__)),
                        f'./data/Ponzi')

        dataset = Ponzi(path)
        data = dataset[0]
        print(data)
        target_node = 'CA'
    elif 'Phish' in args.dataset:
        path = osp.join(osp.dirname(osp.realpath(__file__)),
                        f'./data/Phish')
        dataset = Phish(path)
        data = dataset[0]
        print(data)
        target_node = 'EOA'
    e_type_index = {}
    for index, e in enumerate(data.edge_types):
        e_type_index[e] = index

    _ = generated_generator_sixedges(args, data, device)
    if 'Ponzi' in args.dataset:
        torch.save(_, mkdir(
            f"./pretrain_model/") + f'icvae_Ponzi.pkl')

    elif 'Phish' in args.dataset:
        torch.save(_, mkdir(
            f"./pretrain_model/") + f'icvae_Phish.pkl')
