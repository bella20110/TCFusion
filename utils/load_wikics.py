import torch
import os.path as osp


def get_raw_text_wikics(use_text=False, seed=0):
    data = torch.load('dataset/wikics/wikics_fixed_sbert.pt')
    data.train_mask = data.train_mask[:,seed]
    data.val_mask = data.val_mask[:,seed]
    data.test_mask = data.test_masks[seed]
    data.train_id = torch.arange(data.num_nodes)[data.train_mask].tolist()
    data.val_id = torch.arange(data.num_nodes)[data.val_mask].tolist()
    data.test_id = torch.arange(data.num_nodes)[data.test_mask].tolist()
    return data, data.raw_texts