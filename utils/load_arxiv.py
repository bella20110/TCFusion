"""
Reference: https://github.com/XiaoxinHe/TAPE/blob/main/core/data_utils/load_arxiv.py
"""

from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
import pandas as pd
import os
import numpy as np
import random
from torch_geometric.data import Data

def get_raw_text_arxiv(use_text=False, seed=0):
    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    data = dataset[0]

    idx_splits = dataset.get_idx_split()
    data.train_id = idx_splits['train']
    data.val_id = idx_splits['valid']
    data.test_id = idx_splits['test']
    train_mask = torch.zeros(data.num_nodes).bool()
    val_mask = torch.zeros(data.num_nodes).bool()
    test_mask = torch.zeros(data.num_nodes).bool()
    train_mask[idx_splits['train']] = True
    val_mask[idx_splits['valid']] = True
    test_mask[idx_splits['test']] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # data.edge_index = data.adj_t.to_symmetric()
    data.edge_index
    if not use_text:
        return data, None

    nodeidx2paperid = pd.read_csv(
        './dataset/ogbn_arxiv/nodeidx2paperid.csv')

    raw_text = pd.read_csv('./dataset/ogbn_arxiv/titleabs.tsv', sep='\t')
    raw_text.columns = ['paper id', 'title', 'abs']

    df = pd.merge(nodeidx2paperid, raw_text, on='paper id')

    text = {'title': [], 'abs': [], 'label': []}

    for ti, ab in zip(df['title'], df['abs']):
        text['title'].append(ti)
        text['abs'].append(ab)

    # Load the label index to arXiv category mapping data
    label_mapping_data = pd.read_csv('./dataset/ogbn_arxiv/labelidx2arxivcategeory.csv')
    label_mapping_data.columns = ['label idx', 'arxiv category', 'specific']

    for i in range(len(data.y)):
        row = label_mapping_data.loc[label_mapping_data['label idx'].isin(data.y[i].numpy())]
        # If the row doesn't exist, return a message indicating this
        if len(row) == 0:
            raise 'No matching arXiv category found for this label index.'

        # Parse the arXiv category string to be in the desired format 'cs.XX'
        arxiv_category = 'cs.' + row['arxiv category'].values[0].split()[-1].upper()
        text['label'].append(arxiv_category)

    return data, text


def get_raw_text_arxiv_2023_33868(use_text=True, base_path="dataset/arxiv_2023"):
    # Load processed data
    edge_index = torch.load(os.path.join(base_path, "processed", "edge_index.pt"))

    # Load raw data
    # edge_df = pd.read_csv(os.path.join(base_path, "raw", "edge.csv.gz"), compression='gzip')
    titles_df = pd.read_csv(os.path.join(base_path, "raw", "titles.csv.gz"), compression='gzip')
    abstracts_df = pd.read_csv(os.path.join(base_path, "raw", "abstracts.csv.gz"), compression='gzip')
    ids_df = pd.read_csv(os.path.join(base_path, "raw", "ids.csv.gz"), compression='gzip')
    labels_df = pd.read_csv(os.path.join(base_path, "raw", "labels.csv.gz"), compression='gzip')

    # Load split data
    train_id_df = pd.read_csv(os.path.join(base_path, "split", "train.csv.gz"), compression='gzip')
    val_id_df = pd.read_csv(os.path.join(base_path, "split", "valid.csv.gz"), compression='gzip')
    test_id_df = pd.read_csv(os.path.join(base_path, "split", "test.csv.gz"), compression='gzip')

    num_nodes = len(ids_df)
    titles = titles_df['titles'].tolist()
    abstracts = abstracts_df['abstracts'].tolist()
    ids = ids_df['ids'].tolist()
    labels = labels_df['labels'].tolist()
    train_id = train_id_df['train_id'].tolist()
    val_id = val_id_df['val_id'].tolist()
    test_id = test_id_df['test_id'].tolist()

    features = torch.load(os.path.join(base_path, "processed", "features.pt"))

    y = torch.load(os.path.join(base_path, "processed", "labels.pt"))

    train_mask = torch.tensor([x in train_id for x in range(num_nodes)])
    val_mask = torch.tensor([x in val_id for x in range(num_nodes)])
    test_mask = torch.tensor([x in test_id for x in range(num_nodes)])

    print("to Data")
    data = Data(
        x=features,
        y=y,
        paper_id=ids,
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=num_nodes,
    )

    data.train_id = train_id
    data.val_id = val_id
    data.test_id = test_id

    if not use_text:
        return data, None

    text = {'title': titles, 'abs': abstracts, 'label': labels, 'id': ids}

    return data, text

def get_raw_text_arxiv_2023(use_text=False, seed=0):

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    data = torch.load('dataset/arxiv_2023/graph.pt')

    # split data
    data.num_nodes = len(data.y)
    num_nodes = data.num_nodes
    node_id = np.arange(num_nodes)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[:int(num_nodes * 0.6)])
    data.val_id = np.sort(
        node_id[int(num_nodes * 0.6):int(num_nodes * 0.8)])
    data.test_id = np.sort(node_id[int(num_nodes * 0.8):])

    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(num_nodes)])
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(num_nodes)])

    # data.edge_index = data.adj_t.to_symmetric()
    if not use_text:
        return data, None

    df = pd.read_csv('dataset/arxiv_2023/paper_info.csv')
    # Load the label index to arXiv category mapping data
    label_mapping_data = pd.read_csv('./dataset/ogbn_arxiv/labelidx2arxivcategeory.csv')
    label_mapping_data.columns = ['label idx', 'arxiv category', 'specific']
    text = {'title': [], 'abs': [], 'label': []}
    for ti, ab, label in zip(df['title'], df['abstract'], df['label']):
        text['title'].append(ti)
        text['abs'].append(ab)
        row = label_mapping_data.loc[label_mapping_data['label idx'].isin([label])]
        # If the row doesn't exist, return a message indicating this
        if len(row) == 0:
            raise 'No matching arXiv category found for this label index.'
        # Parse the arXiv category string to be in the desired format 'cs.XX'
        arxiv_category = 'cs.' + row['arxiv category'].values[0].split()[-1].upper()
        text['label'].append(arxiv_category)

    return data, text


def generate_arxiv_keys_list():
    label_mapping_data = pd.read_csv('./dataset/ogbn_arxiv/labelidx2arxivcategeory.csv')
    label_mapping_data.columns = ['label idx', 'arxiv category', 'specific']
    label_mapping_data['sim_categories'] = label_mapping_data['arxiv category'].apply(lambda category: 'cs.' + category.split()[-1].upper())
    id_sim_dict = label_mapping_data.set_index('label idx')['sim_categories'].to_dict()
    return id_sim_dict

