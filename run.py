import math
import os
import copy
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import trange
import os.path as osp
from utils import get_parser, one_hot, mkdir, feature_tensor_normalize
from torch_geometric.loader import NeighborLoader
from dataset import Ponzi, Phish
from model import HMSL
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")


def get_augmented_features(data):
    CA_ls = []
    EOA_ls = []
    cvae_model = torch.load(
        f"./pretrain_model/icvae_{args.dataset}.pkl",
        map_location=device)
    e_type_index = {}
    for index, e in enumerate(data.edge_types):
        e_type_index[e] = index
    for i in range(6):
        src_node = data.edge_types[i][0]
        edge_onehot = one_hot(torch.LongTensor([e_type_index[data.edge_types[i]]]), len(e_type_index)).repeat(
            data[f'{src_node}'].num_nodes, 1).to(device)
        z = torch.randn([data[f'{src_node}'].num_nodes, cvae_model.latent_size]).to(device)
        augmented_features = cvae_model.inference(z, data[f'{src_node}'].x, edge_onehot).detach()
        augmented_features = feature_tensor_normalize(augmented_features).detach()
        if src_node == 'CA':
            CA_ls.append(augmented_features)
        else:
            EOA_ls.append(augmented_features)
    del cvae_model
    return CA_ls, EOA_ls


def train():
    model.train()
    total_examples = total_loss = 0
    for batch in train_data:
        optimizer.zero_grad()
        batch_size = batch[target_node].batch_size
        model_output, loss_co = model(batch.x_dict_new, batch.edge_index_dict)
        loss_train = F.cross_entropy(model_output[:batch_size], batch[target_node].y[:batch_size])
        loss_train = loss_train + args.loss_train * loss_co
        loss_train.backward()
        optimizer.step()
        total_examples += 1
        total_loss += float(loss_train)
    return total_loss / total_examples


@torch.no_grad()
def val():
    model.eval()
    total_examples = total_loss = 0
    all_preds = []
    all_labels = []
    for batch in val_data:
        batch_size = batch[target_node].batch_size
        output, loss_co = model(batch.x_dict_new, batch.edge_index_dict)
        y = batch[target_node].y[:batch_size]
        loss_val = F.cross_entropy(output[:batch_size], y)
        loss_val = loss_val + args.loss_train * loss_co
        total_loss += loss_val
        total_examples += 1
        output = torch.nn.functional.log_softmax(output)
        preds_val_cpu = predict_fn(output[:batch_size]).numpy()
        reals_val_cpu = y.detach().cpu().numpy()
        all_preds.extend(preds_val_cpu)
        all_labels.extend(reals_val_cpu)
    f1_val_micro = f1_score(all_labels, all_preds, average="micro")
    f1_val_macro = f1_score(all_labels, all_preds, average="macro")
    precision_val = precision_score(all_labels, all_preds)
    recall_val = recall_score(all_labels, all_preds)
    f1_val_binary = f1_score(all_labels, all_preds)

    return total_loss / total_examples, f1_val_micro, f1_val_macro, precision_val, recall_val, f1_val_binary


@torch.no_grad()
def test(best_model):
    best_model.eval()
    all_preds = []
    all_labels = []
    for batch in test_data:
        batch_size = batch[target_node].batch_size
        output, loss_co = model(batch.x_dict_new, batch.edge_index_dict)
        output = torch.nn.functional.log_softmax(output)
        preds_test_cpu = predict_fn(output[:batch_size]).numpy()
        reals_test_cpu = batch[target_node].y[:batch_size].detach().cpu().numpy()
        all_preds.extend(preds_test_cpu)
        all_labels.extend(reals_test_cpu)
    f1_test_micro = f1_score(all_labels, all_preds, average="micro")
    f1_test_macro = f1_score(all_labels, all_preds, average="macro")
    precision_test = precision_score(all_labels, all_preds)
    recall_test = recall_score(all_labels, all_preds)
    f1_test_binary = f1_score(all_labels, all_preds)
    return f1_test_micro, f1_test_macro, precision_test, recall_test, f1_test_binary



def get_augmented_data(data):
    data.x_dict_new = data.x_dict
    data.x_dict_new['EOA'] = []
    data.x_dict_new['CA'] = []
    for _ in range(args.concat):
        CA_x, EOA_x = get_augmented_features(data)
        data.x_dict_new['EOA'].append(EOA_x)
        data.x_dict_new['EOA'][_].append(data.x_dict['EOA'])
        data.x_dict_new['CA'].append(CA_x)
        data.x_dict_new['CA'][_].append(data.x_dict['CA'])
        data.x_dict_new['CA'][_] = torch.stack(data.x_dict_new["CA"][_])
        data.x_dict_new['EOA'][_] = torch.stack(data.x_dict_new["EOA"][_])
    return data


if __name__ == '__main__':
    args = get_parser()
    print(args)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    predict_fn = lambda output: output.max(1, keepdim=True)[1].detach().cpu()
    if 'Ponzi' in args.dataset:
        path = osp.join(osp.dirname(osp.realpath(__file__)),
                        f'./data/Ponzi/')
        dataset = Ponzi(path)
        target_node = 'CA'
    elif 'Phish' in args.dataset:
        path = osp.join(osp.dirname(osp.realpath(__file__)),
                        f'./data/Phish/')
        dataset = Phish(path)
        target_node = 'EOA'
    data = dataset[0]
    print(data)
    train_input_nodes = (target_node, data[target_node].train_mask)
    val_input_nodes = (target_node, data[target_node].val_mask)
    test_input_nodes = (target_node, data[target_node].test_mask)
    kwargs = {'batch_size': args.batch_size}
    train_loader = NeighborLoader(data, num_neighbors=[100] * 2, shuffle=True,
                                  input_nodes=train_input_nodes, **kwargs)
    val_loader = NeighborLoader(data, num_neighbors=[100] * 2, shuffle=True,
                                input_nodes=val_input_nodes, **kwargs)
    test_loader = NeighborLoader(data, num_neighbors=[100] * 2, shuffle=False,
                                 input_nodes=test_input_nodes, **kwargs)
    train_data = []
    for batch in train_loader:
        batch = batch.to(device)
        batch = get_augmented_data(batch)
        train_data.append(batch)
    val_data = []
    for batch in val_loader:
        batch = batch.to(device)
        batch = get_augmented_data(batch)
        val_data.append(batch)
    test_data = []
    for batch in test_loader:
        batch = batch.to(device)
        batch = get_augmented_data(batch)
        test_data.append(batch)

    model = HMSL(hidden=args.hidden, out_channels=2, data=data, concat=args.concat, target_node=target_node)
    model = model.to(device)

    test_f1_micro = []
    test_f1_macro = []
    test_precision = []
    test_recall = []
    test_f1 = []

    for i in trange(5, desc='Run Experiments'):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        best_loss = 1e6
        best_f1 = 0
        best_model = None
        patience = 0

        for epoch in range(args.epochs):
            loss = train()
            loss_val, f1_val_micro, f1_val_macro, precision_val, recall_val, f1_val_binary = val()
            if f1_val_macro >= best_f1:
                best_f1 = f1_val_macro
                best_loss = loss_val
                best_model = copy.deepcopy(model)
                patience = 0
            else:
                patience += 1
            if patience > args.patience:
                print('========early_stop========')
                break
        f1_test_micro, f1_test_macro, precision_test, recall_test, f1_test_binary = test(best_model)
        print(
            'f1_test_micro:{:.4f}'.format(f1_test_micro),
            'f1_test_macro:{:.4f}'.format(f1_test_macro),
            'precision_test:{:.4f}'.format(precision_test),
            'recall_test:{:.4f}'.format(recall_test),
            'f1_test_binary:{:.4f}'.format(f1_test_binary))
        test_f1_micro.append(f1_test_micro)
        test_f1_macro.append(f1_test_macro)
        test_precision.append(precision_test)
        test_recall.append(recall_test)
        test_f1.append(f1_test_binary)
    print(
        "final_result:",
        'f1_test_micro:{:.4f}'.format(np.mean(test_f1_micro)),
        'f1_test_macro:{:.4f}'.format(np.mean(test_f1_macro)),
        'precision_test:{:.4f}'.format(np.mean(test_precision)),
        'recall_test:{:.4f}'.format(np.mean(test_recall)),
        'f1_test_binary:{:.4f}'.format(np.mean(test_f1)),
    )

    result_data = [
        [args.lr, args.hidden, args.concat,
         str(round(np.mean(test_f1_micro) * 100, 2)) + '±' + str(round(np.std(test_f1_micro) * 100, 2)),
         str(round(np.mean(test_f1_macro) * 100, 2)) + '±' + str(round(np.std(test_f1_macro) * 100, 2)),
         str(round(np.mean(test_precision) * 100, 2)) + '±' + str(round(np.std(test_precision) * 100, 2)),
         str(round(np.mean(test_recall) * 100, 2)) + '±' + str(round(np.std(test_recall) * 100, 2)),
         str(round(np.mean(test_f1) * 100, 2)) + '±' + str(round(np.std(test_f1) * 100, 2))
         ]]
    result = pd.DataFrame(result_data,
                          columns=['lr', 'hidden1', 'concat', 'loss_train',
                                   'micro_F1', 'macro_F1', 'precision', 'recall', 'f1'])

    save_path = (mkdir(
        f'./result/') +
                 f'Meta-IFD_{args.dataset}.csv')

    if not os.path.exists(save_path):
        result.to_csv(save_path, index=False, mode='a')
    else:
        result.to_csv(save_path, index=False, mode='a', header=False)
