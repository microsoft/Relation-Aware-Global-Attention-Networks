from __future__ import print_function, absolute_import
import json
import os.path as osp
import shutil

import torch
from torch.nn import Parameter

from .osutils import mkdir_if_missing


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def save_checkpoint(state, epoch, is_best, save_interval=1, fpath='checkpoint.pth.tar'):
    docpath = None
    dirpath = osp.dirname(fpath)
    fname = osp.basename(fpath)
    mkdir_if_missing(dirpath)
    docpath = osp.join(dirpath, fname.split('.')[0] + '_{}.pth.tar'.format(epoch))
    torch.save(state, docpath)
    if is_best:
        shutil.copy(fpath, osp.join(dirpath, 'model_best.pth.tar'))


def load_checkpoint(fpath):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath)
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))


def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model


def unfreeze_all_params(model):
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)


def freeze_specific_params(module):
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
