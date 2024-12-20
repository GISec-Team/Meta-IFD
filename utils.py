import argparse
import os
import random
from typing import Optional
import torch
import numpy as np
from torch import Tensor


def get_parser():
    parser = argparse.ArgumentParser(description='Description: Script to run our pretrain_model.')
    # pretrain
    parser.add_argument("--latent_size", type=int, default=10)
    parser.add_argument("--conditional", action='store_true', default=True)
    parser.add_argument('--pretrain_epochs', type=int, default=10, help='Number of epochs to train.')
    # main
    parser.add_argument("--concat", type=int, default=3)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument('--runs', type=int, default=5, help='The number of experiments.')
    parser.add_argument('--dataset', default='Ponzi', help='Dataset string.')
    parser.add_argument('--seed', type=int, default=12, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')
    parser.add_argument('--gpu', type=str, help='gpu id', default='0')
    parser.add_argument('--model', type=str, help='atten', default='easy_model')
    parser.add_argument('--loss_train', type=float, help='parameters of loss_train', default=0.1)
    parser.add_argument('--loss', type=str, default='CE')
    parser.add_argument('--batch_size', type=int, help='batch', default=512)

    return parser.parse_args()


def one_hot(
        index: Tensor,
        num_classes: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Taskes a one-dimensional :obj:`index` tensor and returns a one-hot
    encoded representation of it with shape :obj:`[*, num_classes]` that has
    zeros everywhere except where the index of last dimension matches the
    corresponding value of the input tensor, in which case it will be :obj:`1`.

    .. note::
        This is a more memory-efficient version of
        :meth:`torch.nn.functional.one_hot` as you can customize the output
        :obj:`dtype`.

    Args:
        index (torch.Tensor): The one-dimensional input tensor.
        num_classes (int, optional): The total number of classes. If set to
            :obj:`None`, the number of classes will be inferred as one greater
            than the largest class value in the input tensor.
            (default: :obj:`None`)
        dtype (torch.dtype, optional): The :obj:`dtype` of the output tensor.
    """
    if index.dim() != 1:
        raise ValueError("'index' tensor needs to be one-dimensional")

    if num_classes is None:
        num_classes = int(index.max()) + 1

    out = torch.zeros((index.size(0), num_classes), dtype=dtype,
                      device=index.device)
    return out.scatter_(1, index.unsqueeze(1), 1)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


def mkdir(path):
    """
    :param path:
    :return:
    """
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    return path


def feature_tensor_normalize(feature):
    feature = torch.tensor(feature, dtype=torch.float32)
    rowsum = torch.div(1.0, torch.sum(feature, dim=1))
    rowsum[torch.isinf(rowsum)] = 0.
    feature.mul_(rowsum.unsqueeze(1))

    return feature
