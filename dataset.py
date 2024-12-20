#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jcx
@contact: 2112103081@zjut.edu.cn
@file: dataset.py
@time: 2023/4/12 14:27
@desc:
'''
import random
import sys
from typing import Optional, Callable, List

import pandas as pd
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import os.path as osp
import torch
import numpy as np
import torch_geometric.transforms
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch_geometric.io import read_txt_array
from torch_geometric.data import (InMemoryDataset, HeteroData, download_url,
                                  extract_zip)
from torch_geometric.transforms import RemoveIsolatedNodes

from utils import get_parser, seed_everything


class Ponzi(InMemoryDataset):
    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def read_file(self, folder, name):
        try:
            path = osp.join(folder, f'{name}.txt')
            return read_txt_array(path, sep=',', dtype=torch.long)
        except:
            path = osp.join(folder, f'{name}.txt')
            return read_txt_array(path, sep=',', dtype=torch.float)

    def index_to_mask(self, index, size):
        mask = torch.zeros((size,), dtype=torch.bool)
        mask[index] = 1
        return mask

    def process(self):
        args = get_parser()
        seed_everything(args.seed)
        path = f'./dataset/Ponzi_unbalance/'

        CA_x_l = pd.read_csv(path + 'CA_labeled.csv')
        CA_x_l = CA_x_l.drop(
            CA_x_l[
                (CA_x_l['N_inv_c'] == 0) &
                (CA_x_l['N_return_c'] == 0) &
                (CA_x_l['N_inv_t'] == 0) &
                (CA_x_l['N_return_t'] == 0)
                ].index
        )
        CA_x_ul = pd.read_csv(path + 'CA_unlabeled.csv')
        CA = pd.concat([CA_x_l, CA_x_ul])
        CA_x = torch.tensor(CA.iloc[:, 1:].values).to(torch.float)

        EOA = pd.read_csv(path + 'EOA.csv')
        EOA = EOA['name'].tolist()
        EOA_x = pd.read_csv('./dataset/node_hete_EOA.csv')
        EOA = EOA_x[EOA_x['Address'].isin(EOA)]
        EOA_x = torch.tensor(EOA.iloc[:, 1:].values).to(torch.float)

        node_dist_eoa = {name: index for index, name in enumerate(EOA['Address'])}
        node_dist_ca = {name: index for index, name in enumerate(CA['Address'])}

        CA_ls = CA['Address'].tolist()
        EOA_ls = EOA['Address'].tolist()

        ca_call_ca = pd.read_csv(path + 'edge/' + 'call_ca_ca.csv')
        ca_call_ca = ca_call_ca[ca_call_ca[':START_ID'].isin(CA_ls) & ca_call_ca[':END_ID'].isin(CA_ls)]
        ca_call_ca[':START_ID'] = ca_call_ca[':START_ID'].map(node_dist_ca)
        ca_call_ca[':END_ID'] = ca_call_ca[':END_ID'].map(node_dist_ca)
        ca_call_ca_tensor = torch.tensor(ca_call_ca.values)
        ca_call_ca = ca_call_ca_tensor[:, :2].to(torch.long).t()

        ca_call_ca_attr = ca_call_ca_tensor[:, 2:].to(torch.float)
        eoa_call_ca = pd.read_csv(path + 'edge/' + 'call_eoa_ca.csv')
        eoa_call_ca = eoa_call_ca[eoa_call_ca[':START_ID'].isin(EOA_ls) & eoa_call_ca[':END_ID'].isin(CA_ls)]
        eoa_call_ca[':START_ID'] = eoa_call_ca[':START_ID'].map(node_dist_eoa)
        eoa_call_ca[':END_ID'] = eoa_call_ca[':END_ID'].map(node_dist_ca)
        eoa_call_ca_tensor = torch.tensor(eoa_call_ca.values)
        eoa_call_ca = eoa_call_ca_tensor[:, :2].to(torch.long).t()
        eoa_call_ca_attr = eoa_call_ca_tensor[:, 2:].to(torch.float)

        eoa_trans_ca = pd.read_csv(path + 'edge/' + 'trans_eoa_ca.csv')
        eoa_trans_ca = eoa_trans_ca[eoa_trans_ca[':START_ID'].isin(EOA_ls) & eoa_trans_ca[':END_ID'].isin(CA_ls)]
        eoa_trans_ca[':START_ID'] = eoa_trans_ca[':START_ID'].map(node_dist_eoa)
        eoa_trans_ca[':END_ID'] = eoa_trans_ca[':END_ID'].map(node_dist_ca)
        eoa_trans_ca_tensor = torch.tensor(eoa_trans_ca.values)
        eoa_trans_ca = eoa_trans_ca_tensor[:, :2].to(torch.long).t()
        eoa_trans_ca_attr = eoa_trans_ca_tensor[:, 2:].to(torch.float)

        eoa_trans_eoa = pd.read_csv(path + 'edge/' + 'trans_eoa_eoa.csv')
        eoa_trans_eoa = eoa_trans_eoa[
            eoa_trans_eoa[':START_ID'].isin(EOA_ls) & eoa_trans_eoa[':END_ID'].isin(EOA_ls)]
        eoa_trans_eoa[':START_ID'] = eoa_trans_eoa[':START_ID'].map(node_dist_eoa)
        eoa_trans_eoa[':END_ID'] = eoa_trans_eoa[':END_ID'].map(node_dist_eoa)
        eoa_trans_eoa_tensor = torch.tensor(eoa_trans_eoa.values)
        eoa_trans_eoa = eoa_trans_eoa_tensor[:, :2].to(torch.long).t()
        eoa_trans_eoa_attr = eoa_trans_eoa_tensor[:, 2:].to(torch.float)

        ca_trans_eoa = pd.read_csv(path + 'edge/' + 'trans_ca_eoa.csv')
        ca_trans_eoa = ca_trans_eoa[ca_trans_eoa[':START_ID'].isin(CA_ls) & ca_trans_eoa[':END_ID'].isin(EOA_ls)]
        ca_trans_eoa[':START_ID'] = ca_trans_eoa[':START_ID'].map(node_dist_ca)
        ca_trans_eoa[':END_ID'] = ca_trans_eoa[':END_ID'].map(node_dist_eoa)
        ca_trans_eoa_tensor = torch.tensor(ca_trans_eoa.values)
        ca_trans_eoa = ca_trans_eoa_tensor[:, :2].to(torch.long).t()
        ca_trans_eoa_attr = ca_trans_eoa_tensor[:, 2:].to(torch.float)

        ca_trans_ca = pd.read_csv(path + 'edge/' + 'trans_ca_ca.csv')
        ca_trans_ca = ca_trans_ca[ca_trans_ca[':START_ID'].isin(CA_ls) & ca_trans_ca[':END_ID'].isin(CA_ls)]
        ca_trans_ca[':START_ID'] = ca_trans_ca[':START_ID'].map(node_dist_ca)
        ca_trans_ca[':END_ID'] = ca_trans_ca[':END_ID'].map(node_dist_ca)
        ca_trans_ca_tensor = torch.tensor(ca_trans_ca.values)
        ca_trans_ca = ca_trans_ca_tensor[:, :2].to(torch.long).t()
        ca_trans_ca_attr = ca_trans_ca_tensor[:, 2:].to(torch.float)

        CA_y = torch.tensor([1] * 191 + [0] * 1151)
        m = np.array(list(range(0, 1342)))
        train_index, test_index = train_test_split(m, test_size=0.2, random_state=args.seed, stratify=CA_y)
        train_index, val_index = train_test_split(train_index, test_size=0.25, random_state=args.seed,
                                                  stratify=CA_y[train_index])

        data = HeteroData()
        CA_y = F.pad(CA_y, pad=(0, CA_x.shape[0] - CA_y.shape[0]), mode='constant', value=2)
        data['CA'].x = CA_x
        data['EOA'].x = EOA_x
        data['CA'].y = CA_y

        data['CA', 'call', 'CA'].edge_index = ca_call_ca
        data['EOA', 'call', 'CA'].edge_index = eoa_call_ca
        data['EOA', 'trans', 'CA'].edge_index = eoa_trans_ca
        data['CA', 'trans', 'CA'].edge_index = ca_trans_ca
        data['CA', 'trans', 'EOA'].edge_index = ca_trans_eoa
        data['EOA', 'trans', 'EOA'].edge_index = eoa_trans_eoa

        data['CA', 'call', 'CA'].edge_attr = ca_call_ca_attr
        data['EOA', 'call', 'CA'].edge_attr = eoa_call_ca_attr
        data['EOA', 'trans', 'CA'].edge_attr = eoa_trans_ca_attr
        data['CA', 'trans', 'CA'].edge_attr = ca_trans_ca_attr
        data['CA', 'trans', 'EOA'].edge_attr = ca_trans_eoa_attr
        data['EOA', 'trans', 'EOA'].edge_attr = eoa_trans_eoa_attr

        train_mask = self.index_to_mask(train_index, size=data['CA'].y.size(0))
        val_mask = self.index_to_mask(val_index, size=data['CA'].y.size(0))
        test_mask = self.index_to_mask(test_index, size=data['CA'].y.size(0))

        data["CA"].train_mask = train_mask
        data["CA"].val_mask = val_mask
        data["CA"].test_mask = test_mask
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class Phish(InMemoryDataset):
    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def read_file(self, folder, name):
        try:
            path = osp.join(folder, f'{name}.txt')
            return read_txt_array(path, sep=',', dtype=torch.long)
        except:
            path = osp.join(folder, f'{name}.txt')
            return read_txt_array(path, sep=',', dtype=torch.float)

    def index_to_mask(self, index, size):
        mask = torch.zeros((size,), dtype=torch.bool)
        mask[index] = 1
        return mask

    def process(self):
        args = get_parser()
        seed_everything(args.seed)
        path = f'./dataset/Phish_unbalance/'
        CA = pd.read_csv(path + 'CA.csv')
        CA_x = torch.tensor(CA.iloc[:, 1:].values).to(torch.float)
        EOA_labeled = pd.read_csv(path + 'EOA_labeled.csv')
        EOA_unlabeled = pd.read_csv(path + 'EOA_unlabeled.csv')
        EOA_labeled_ls = EOA_labeled['name'].tolist()
        EOA_unlabeled_ls = EOA_unlabeled['name'].tolist()
        EOA_x = pd.read_csv('./dataset/node_hete_EOA.csv')
        EOA_x_phish = EOA_x[EOA_x['Address'].isin(EOA_labeled_ls[:1207])].drop_duplicates()
        EOA_x_unphish = EOA_x[EOA_x['Address'].isin(EOA_labeled_ls[1207:])]
        EOA_x_unlabeled = EOA_x[EOA_x['Address'].isin(EOA_unlabeled_ls)]
        EOA = pd.concat([EOA_x_phish, EOA_x_unphish], axis=0)
        EOA = EOA.drop(
            EOA[
                (EOA['N_inv_c'] == 0) &
                (EOA['N_return_c'] == 0) &
                (EOA['N_inv_t'] == 0) &
                (EOA['N_return_t'] == 0)
                ].index
        )  # delete one phish account without features
        EOA = pd.concat([EOA, EOA_x_unlabeled], axis=0)
        EOA_x = torch.tensor(EOA.iloc[:, 1:].values).to(torch.float)
        CA_ls = CA['Address'].tolist()
        EOA_ls = EOA['Address'].tolist()
        node_dist_eoa = {name: index for index, name in enumerate(EOA['Address'])}
        node_dist_ca = {name: index for index, name in enumerate(CA['Address'])}

        ca_call_ca = pd.read_csv(path + 'edge/' + 'call_ca_ca.csv')
        ca_call_ca = ca_call_ca[ca_call_ca[':START_ID'].isin(CA_ls) & ca_call_ca[':END_ID'].isin(CA_ls)]
        ca_call_ca[':START_ID'] = ca_call_ca[':START_ID'].map(node_dist_ca)
        ca_call_ca[':END_ID'] = ca_call_ca[':END_ID'].map(node_dist_ca)
        ca_call_ca_tensor = torch.tensor(ca_call_ca.values)
        ca_call_ca = ca_call_ca_tensor[:, :2].to(torch.long).t()
        ca_call_ca_attr = ca_call_ca_tensor[:, 2:].to(torch.float)

        eoa_call_ca = pd.read_csv(path + 'edge/' + 'call_eoa_ca.csv')
        eoa_call_ca = eoa_call_ca[eoa_call_ca[':START_ID'].isin(EOA_ls) & eoa_call_ca[':END_ID'].isin(CA_ls)]
        eoa_call_ca[':START_ID'] = eoa_call_ca[':START_ID'].map(node_dist_eoa)
        eoa_call_ca[':END_ID'] = eoa_call_ca[':END_ID'].map(node_dist_ca)
        eoa_call_ca_tensor = torch.tensor(eoa_call_ca.values)
        eoa_call_ca = eoa_call_ca_tensor[:, :2].to(torch.long).t()
        eoa_call_ca_attr = eoa_call_ca_tensor[:, 2:].to(torch.float)

        eoa_trans_ca = pd.read_csv(path + 'edge/' + 'trans_eoa_ca.csv')
        eoa_trans_ca = eoa_trans_ca[eoa_trans_ca[':START_ID'].isin(EOA_ls) & eoa_trans_ca[':END_ID'].isin(CA_ls)]
        eoa_trans_ca[':START_ID'] = eoa_trans_ca[':START_ID'].map(node_dist_eoa)
        eoa_trans_ca[':END_ID'] = eoa_trans_ca[':END_ID'].map(node_dist_ca)
        eoa_trans_ca_tensor = torch.tensor(eoa_trans_ca.values)
        eoa_trans_ca = eoa_trans_ca_tensor[:, :2].to(torch.long).t()
        eoa_trans_ca_attr = eoa_trans_ca_tensor[:, 2:].to(torch.float)

        eoa_trans_eoa = pd.read_csv(path + 'edge/' + 'trans_eoa_eoa.csv')
        eoa_trans_eoa = eoa_trans_eoa[
            eoa_trans_eoa[':START_ID'].isin(EOA_ls) & eoa_trans_eoa[':END_ID'].isin(EOA_ls)]
        eoa_trans_eoa[':START_ID'] = eoa_trans_eoa[':START_ID'].map(node_dist_eoa)
        eoa_trans_eoa[':END_ID'] = eoa_trans_eoa[':END_ID'].map(node_dist_eoa)
        eoa_trans_eoa_tensor = torch.tensor(eoa_trans_eoa.values)
        eoa_trans_eoa = eoa_trans_eoa_tensor[:, :2].to(torch.long).t()
        eoa_trans_eoa_attr = eoa_trans_eoa_tensor[:, 2:].to(torch.float)

        ca_trans_eoa = pd.read_csv(path + 'edge/' + 'trans_ca_eoa.csv')
        ca_trans_eoa = ca_trans_eoa[ca_trans_eoa[':START_ID'].isin(CA_ls) & ca_trans_eoa[':END_ID'].isin(EOA_ls)]
        ca_trans_eoa[':START_ID'] = ca_trans_eoa[':START_ID'].map(node_dist_ca)
        ca_trans_eoa[':END_ID'] = ca_trans_eoa[':END_ID'].map(node_dist_eoa)
        ca_trans_eoa_tensor = torch.tensor(ca_trans_eoa.values)
        ca_trans_eoa = ca_trans_eoa_tensor[:, :2].to(torch.long).t()
        ca_trans_eoa_attr = ca_trans_eoa_tensor[:, 2:].to(torch.float)

        ca_trans_ca = pd.read_csv(path + 'edge/' + 'trans_ca_ca.csv')
        ca_trans_ca = ca_trans_ca[ca_trans_ca[':START_ID'].isin(CA_ls) & ca_trans_ca[':END_ID'].isin(CA_ls)]
        ca_trans_ca[':START_ID'] = ca_trans_ca[':START_ID'].map(node_dist_ca)
        ca_trans_ca[':END_ID'] = ca_trans_ca[':END_ID'].map(node_dist_ca)
        ca_trans_ca_tensor = torch.tensor(ca_trans_ca.values)
        ca_trans_ca = ca_trans_ca_tensor[:, :2].to(torch.long).t()
        ca_trans_ca_attr = ca_trans_ca_tensor[:, 2:].to(torch.float)

        EOA_y = torch.tensor([1] * 1206 + [0] * 1557)
        m = np.array(list(range(0, 1206 + 1557)))
        train_index, test_index = train_test_split(m, test_size=0.2, random_state=args.seed, stratify=EOA_y)
        train_index, val_index = train_test_split(train_index, test_size=0.25, random_state=args.seed,
                                                  stratify=EOA_y[train_index])

        data = HeteroData()
        EOA_y = F.pad(EOA_y, pad=(0, EOA_x.shape[0] - EOA_y.shape[0]), mode='constant', value=2)
        data['EOA'].x = EOA_x
        data['EOA'].y = EOA_y
        data['CA'].x = CA_x
        data['CA', 'call', 'CA'].edge_index = ca_call_ca
        data['EOA', 'call', 'CA'].edge_index = eoa_call_ca
        data['EOA', 'trans', 'CA'].edge_index = eoa_trans_ca
        data['CA', 'trans', 'CA'].edge_index = ca_trans_ca
        data['CA', 'trans', 'EOA'].edge_index = ca_trans_eoa
        data['EOA', 'trans', 'EOA'].edge_index = eoa_trans_eoa

        data['CA', 'call', 'CA'].edge_attr = ca_call_ca_attr
        data['EOA', 'call', 'CA'].edge_attr = eoa_call_ca_attr
        data['EOA', 'trans', 'CA'].edge_attr = eoa_trans_ca_attr
        data['CA', 'trans', 'CA'].edge_attr = ca_trans_ca_attr
        data['CA', 'trans', 'EOA'].edge_attr = ca_trans_eoa_attr
        data['EOA', 'trans', 'EOA'].edge_attr = eoa_trans_eoa_attr

        train_mask = self.index_to_mask(train_index, size=data['EOA'].x.size(0))
        val_mask = self.index_to_mask(val_index, size=data['EOA'].x.size(0))
        test_mask = self.index_to_mask(test_index, size=data['EOA'].x.size(0))

        data["EOA"].train_mask = train_mask
        data["EOA"].val_mask = val_mask
        data["EOA"].test_mask = test_mask
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
