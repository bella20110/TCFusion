import json
import pickle
import re

import numpy as np
import os
import torch
import time
import math
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from utils.utils import get_subgraph, load_data, sample_test_nodes, map_arxiv_labels, get_combine_text, get_semantic_and_structure_neighbors, get_combine_text, get_qwen_instruct, get_e5_Instruct
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer
import faiss
from pyserini.search.lucene import LuceneSearcher
from collections import Counter
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file
from datasketch import MinHash
from heapq import nlargest

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
        }


def normalize_softmax(scores):
    values = np.array(list(scores.values()))
    exp_values = np.exp(values)
    sum_exp = np.sum(exp_values)
    normalized = exp_values / sum_exp
    return dict(zip(scores.keys(), normalized))


def normalize_l1(scores):
    values = np.array(list(scores.values()))
    norm = np.sum(np.abs(values)) + 1e-8
    normalized = values / norm
    return dict(zip(scores.keys(), normalized))



def get_signature_vec(neighbors, num_perm=128):
    m = MinHash(num_perm=num_perm)
    if neighbors is None or len(neighbors) == 0:
        return np.zeros(num_perm, dtype=np.uint64)
    for n in neighbors:
        m.update(str(n).encode('utf8'))
    return np.array(m.digest(), dtype=np.uint64)


def count_isolated_nodes(node_list, edge_matrix):
    """
    统计无连接边的结点数量
    :param node_list: 所有结点的列表（如 [1,2,3,4] 或 ['A','B','C']）
    :param edge_matrix: 引文矩阵（两行，列数为边数，如 [[起点1,起点2],[终点1,终点2]]）
    :return: 无边结点数量、无边结点列表
    """
    # 步骤1：提取所有参与边的结点（起点+终点）
    start_nodes = edge_matrix[0].numpy().tolist()  # 第一行：所有边的起点
    end_nodes = edge_matrix[1].numpy().tolist()  # 第二行：所有边的终点
    has_edge_nodes = set(start_nodes + end_nodes)  # 去重，得到有边的结点集合

    # 步骤2：找出无边的结点
    isolated_nodes = [node for node in node_list if node not in has_edge_nodes]

    # 步骤3：返回数量和具体结点
    return len(isolated_nodes), isolated_nodes

def main():
    dataset_name = "arxiv_2023"
    dataset_name = "cora"
    # dataset_name = "arxiv"
    # dataset_name = "wikics"
    # dataset_name = "citeseer"
    seeds = [0, 1, 2, 3, 4]
    test_log = np.zeros(len(seeds))
    test_predict_log = np.zeros(len(seeds))
    test_sim_log = np.zeros(len(seeds))
    test_cite_log = np.zeros(len(seeds))
    model_name = "all-MiniLM-L6-v2"
    model_name = "intfloat/e5-large"

    model_name = "e5-large_lora"
    # model_name = "e5-large"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    normalize_function = normalize_l1
    # 节点相似性
    all_hops = [1, 2]

    scoreType = 1
    now_hop = 2
    data, text = load_data(dataset_name, use_text=False, seed=0)
    unique_classes = np.unique(data.y)
    nodes_sum = data.num_nodes
    unique_classes_arr = np.array(list(unique_classes))
    n_classes = len(unique_classes_arr)
    cls2idx = {cls: idx for idx, cls in enumerate(unique_classes_arr)}
    structure_neighbor_dict = {}
    count_neighbor = [0, 0, 0]
    cache_path = "./neighbor_dict/{}_{}hop_minhash_neighbors_type2.pkl".format(dataset_name, len(all_hops))
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            structure_neighbor_dict = pickle.load(f)
    else:
        node_neighbor_structure = {}
        hop_records = {}
        minhash_signatures = {}
        MAX_NEIGHBOR_PER_HOP = 1000
        for node in tqdm(range(nodes_sum), desc="get node neighbors and minhash"):
            if isinstance(node, torch.Tensor):
                node = node.item()
            current_nodes = torch.tensor([node])
            visited_nodes = set(current_nodes.tolist())
            neighbor_hops = {}

            for hop in all_hops:
                mask = torch.isin(data.edge_index[0], current_nodes) | torch.isin(data.edge_index[1], current_nodes)
                neighbor_nodes = torch.unique(torch.cat((data.edge_index[0][mask], data.edge_index[1][mask])))
                neighbor_nodes = neighbor_nodes[~torch.isin(neighbor_nodes, torch.tensor(list(visited_nodes)))]
                if len(neighbor_nodes) > MAX_NEIGHBOR_PER_HOP:
                    neighbor_nodes = neighbor_nodes[torch.randperm(len(neighbor_nodes))[:MAX_NEIGHBOR_PER_HOP]]
                visited_nodes.update(neighbor_nodes.tolist())
                count_neighbor[hop - 1] += len(visited_nodes)
                for neighbor in neighbor_nodes.tolist():
                    neighbor_hops[neighbor] = hop
                current_nodes = neighbor_nodes

            # visited_nodes.remove(node)
            node_neighbor_structure[node] = visited_nodes
            hop_records[node] = neighbor_hops
            minhash_signatures[node] = get_signature_vec(visited_nodes, 128)

        for node in tqdm(range(nodes_sum), desc="calculate Jaccard similarity"):
            if isinstance(node, torch.Tensor):
                node = node.item()
            neighbors = node_neighbor_structure[node]
            if len(neighbors) > 0:
                sig_u = minhash_signatures[node]
                sig_neighbors = [minhash_signatures[idx] for idx in neighbors]
                similarities = np.mean(sig_neighbors == sig_u, axis=1)  # 计算相等比例
                # similarities = np.zeros(len(neighbors), dtype=int)
                hop_dict = hop_records[node]
                top_k = sorted(
                    ((n, s, hop_dict.get(n, 0)) for n, s in zip(neighbors, similarities)),
                    key=lambda x: x[1],
                    reverse=True
                )
                structure_neighbor_dict[node] = top_k
            else:
                structure_neighbor_dict[node] = []

        del node_neighbor_structure, hop_records, minhash_signatures
        with open(cache_path, "wb") as f:
            pickle.dump(structure_neighbor_dict, f)

    for seed in seeds:
        start_time = time.perf_counter()
        data, text = load_data(dataset_name, use_text=True, seed=seed)
        # sumnode = list(data.val_id) + list(data.test_id)
        # count, nodes = count_isolated_nodes(sumnode, data.edge_index)
        # print("孤立节点数量:" + str(count))
        # print("孤立节点占比:" + str(count / len(sumnode)))
        # continue
        weights = None
        best_acc = 0
        best_k = 0
        best_sim = 0
        best_pre = 0
        best_cite = 0
        if isinstance(data.train_id[0], torch.Tensor):
            data.train_id = data.train_id.numpy()
            data.val_id = data.val_id.numpy()
            data.test_id = data.test_id.numpy()

        train_id_set = set(data.train_id)
        model_path = './model/' + model_name
        predict_file = "./predict/{}_{}_predict_{}.npy".format(dataset_name, model_name.split("/")[-1], seed)
        predict = np.load(predict_file)

        node_embedding_file = "./emb/{}_{}_embeddings_{}.npy".format(dataset_name, model_name.split("/")[-1], seed)
        # node_embedding_file = None
        if node_embedding_file is not None:
            node_emb = torch.tensor(np.load(node_embedding_file))
        else:
            def last_token_pool(last_hidden_states: Tensor,
                                attention_mask: Tensor) -> Tensor:
                left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
                if left_padding:
                    return last_hidden_states[:, -1]
                else:
                    sequence_lengths = attention_mask.sum(dim=1) - 1
                    batch_size = last_hidden_states.shape[0]
                    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

            # model = SentenceTransformer(model_name, cache_folder=model_path)
            if model_name == 'all-MiniLM-L6-v2' or model_name == 'intfloat/e5-large':
                tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                model = AutoModel.from_pretrained(model_name, add_pooling_layer=False).to(device)
                combine_text = get_combine_text(text, dataset_name, tokenizer)
            max_length = 512

            tokenized_inputs = tokenizer(
                combine_text,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs['input_ids'][0])

            dataset = TextDataset(tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'])
            dataloader = DataLoader(dataset, batch_size=12, shuffle=False,
                                    collate_fn=lambda x: tokenizer.pad(x, return_tensors="pt"))
            all_outputs = []
            tqdm_bar = tqdm(dataloader, desc="embedding")
            for batch in tqdm_bar:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)
                if model_name == 'scibert' or model_name == 'all-MiniLM-L6-v2' or model_name == 'intfloat/e5-large':
                    node_emb = outputs.last_hidden_state[:, 0].detach().cpu()
                else:
                    node_emb = last_token_pool(outputs.last_hidden_state, batch['attention_mask'])
                all_outputs.append(node_emb) 

            node_emb = torch.cat(all_outputs, dim=0)
            node_emb = F.normalize(node_emb, p=2, dim=-1).cpu().numpy()

            # node_emb = model.encode(combine_text, convert_to_tensor=True, show_progress_bar=True).cuda()
            np.save("./emb/{}_{}_embeddings.npy".format(dataset_name, model_name.split("/")[-1]), node_emb)

        dimension = node_emb.shape[1]  # Dimensionality of embeddings
        base_index = faiss.index_factory(dimension, 'Flat', faiss.METRIC_INNER_PRODUCT)
        index = faiss.IndexIDMap(base_index)
        index.add_with_ids(node_emb, list(range(len(node_emb))))

        k_ = [1, 5, 10, 15, 20, 25, 30]
        all_distances = {}
        all_indices = {}

        if dataset_name == "arxiv":
            distance_path = "./similarity_search/{}_{}hop_distances.npy".format(dataset_name, len(all_hops))
            indices_path = "./similarity_search/{}_{}hop_indices.npy".format(dataset_name, len(all_hops))
        else:
            distance_path = "./similarity_search/{}_{}hop_distances_{}.npy".format(dataset_name, len(all_hops), seed)
            indices_path = "./similarity_search/{}_{}hop_indices_{}.npy".format(dataset_name, len(all_hops), seed)
        if os.path.exists(distance_path) and os.path.exists(indices_path):
            all_distances = np.load(distance_path, allow_pickle=True).item()
            all_indices = np.load(indices_path, allow_pickle=True).item()

        else:
            for id in tqdm(list(data.val_id) + list(data.test_id)):
                cur_node_emb = node_emb[id].reshape(1, -1)
                distances, indices = index.search(cur_node_emb, 50)
                all_distances[id] = distances[0]
                all_indices[id] = indices[0]
            np.save(distance_path, all_distances)
            np.save(indices_path, all_indices)


        predict_score = {}
        predict_label = {}
        for id in tqdm(range(nodes_sum)):
            if id in train_id_set:
                continue
            predict_score1 = {}
            for cls in unique_classes:
                predict_score1[cls] = predict[id][cls]
            predict_score[id] = normalize_softmax(predict_score1)
            predict_label[id] = max(predict_score1, key=predict_score1.get)

        if dataset_name == "arxiv":
            structure_score_path = "./similarity_search/{}_{}hop_structure_score.npy".format(dataset_name, len(all_hops))
        else:
            structure_score_path = "./similarity_search/{}_{}hop_structure_score_{}.npy".format(dataset_name,
                                                                                             len(all_hops), seed)
        if os.path.exists(structure_score_path):
            structure_class_score = np.load(structure_score_path, allow_pickle=True).item()
        else:
            structure_class_score = {}
            for id in tqdm(list(data.val_id) + list(data.test_id)):
                important_neighbors = structure_neighbor_dict[id]

                if not important_neighbors:  # 无邻居时直接赋值0
                    structure_class_score[id] = normalize_function({cls: 0 for cls in unique_classes})
                    continue

                stu_class_count = {h: np.zeros(n_classes, dtype=np.float32) for h in all_hops}
                count_h = Counter()
                for i, t in enumerate(important_neighbors):
                    t_id = int(t[0])
                    if t_id == id:
                        continue
                    weight_t1 = t[1]
                    h = t[2]
                    weight = weight_t1 / h
                    count_h[h] += 1
                    if t_id in train_id_set:
                        cls_idx = cls2idx[data.y[t_id].item()]
                        stu_class_count[h][cls_idx] += weight
                    else:
                        node_score = predict_score[t_id]
                        score_arr = np.array([node_score[cls] for cls in unique_classes_arr], dtype=np.float32)
                        stu_class_count[h] += score_arr * weight

                final_score_arr = np.zeros(n_classes, dtype=np.float32)
                for h in all_hops:
                    cnt = count_h.get(h, 0)
                    denom = np.sqrt(cnt) if cnt > 0 else 1.0
                    final_score_arr += stu_class_count[h] / denom
                final_score = {cls: final_score_arr[cls2idx[cls]] for cls in unique_classes}
                structure_class_score[id] = normalize_function(final_score)

            np.save(structure_score_path, structure_class_score)


        for now_k in k_:
            semantic_score = {}

            for id in tqdm(list(data.val_id) + list(data.test_id)):
                distances = all_distances.get(id)[0:now_k+1]
                indices = all_indices.get(id)[0:now_k+1]

                # 2. 数组替代字典，初始化类别总和/计数（float32节省内存+提速）
                class_sum_arr = np.zeros(n_classes, dtype=np.float32)
                class_count_arr = np.zeros(n_classes, dtype=np.int32)
                total_count = 0

                # 3. 遍历邻居，批量计算（zip同时遍历距离和索引）
                for dist, idx in zip(distances, indices):
                    idx = int(idx)  # 确保是整数ID
                    if idx == id:
                        continue
                    total_count += 1
                    # 3.1 训练集节点：直接累加距离和计数
                    if idx in train_id_set:
                        cls = data.y[idx].item()
                        cls_idx = cls2idx[cls]
                        class_sum_arr[cls_idx] += dist
                        class_count_arr[cls_idx] += 1
                    # 3.2 非训练集节点：用预测得分加权
                    else:
                        # 从批量缓存中取预测结果（避免重复查找）
                        node_score = predict_score[idx]
                        node_label = predict_label[idx]

                        # 向量化累加class_sum：替代for cls in unique_classes
                        score_arr = np.array([node_score[cls] for cls in unique_classes_arr], dtype=np.float32)
                        class_sum_arr += score_arr * dist  # dist即weight

                        # 累加预测类别的计数
                        label_idx = cls2idx[node_label]
                        class_count_arr[label_idx] += 1

                # 4. 计算最终得分（scoreType=1的逻辑）
                class_score1_arr = np.zeros(n_classes, dtype=np.float32)
                if total_count > 0:
                    class_score1_arr = class_sum_arr  # 原逻辑：avg_distance直接取class_sum（若要平均则除以class_count_arr）
                # 5. 数组转回字典并归一化
                class_score1 = {cls: class_score1_arr[cls2idx[cls]] for cls in unique_classes}
                semantic_score[id] = normalize_function(class_score1)

            print("---------------------")
            models = [predict_score, semantic_score, structure_class_score]
            X, y = [], []
            for id in data.val_id:
                for cls in unique_classes_arr:
                    X.append([m[id][cls] for m in models])  # M=3
                    y.append(1 if data.y[id] == cls else 0)

            reg = Ridge(alpha=1.0, fit_intercept=False)
            reg.fit(X, y)


            def reg_predict(id_):
                scores = {}
                for c in unique_classes_arr:
                    x = np.array([[m[id_][c] for m in models]])
                    scores[c] = reg.predict(x)[0]
                return scores


            cor_sum = 0
            cor_predict_sum = 0
            cor_sim_sum = 0
            cor_cite_sum = 0
            sum = 0
            for id in data.test_id:
                combined_scores = {}
                node_predict = predict_score[id]
                similarity_scores = semantic_score[id]
                citing_scores = structure_class_score[id]
                for key in similarity_scores.keys():
                    combined_score = weights[0] * node_predict[key] + weights[1] * similarity_scores[key] + weights[2] * \
                                 citing_scores[key]
                    combined_scores[key] = combined_score
                combined_scores = reg_predict(id)
                best_key = max(combined_scores, key=combined_scores.get)
                best_predict_key = max(node_predict, key=node_predict.get)
                best_sim_key = max(similarity_scores, key=similarity_scores.get)
                best_cite_key = max(citing_scores, key=citing_scores.get)
                if best_key == data.y[id]:
                    cor_sum += 1
                if best_predict_key == data.y[id]:
                    cor_predict_sum += 1
                if best_sim_key == data.y[id]:
                    cor_sim_sum += 1
                if best_cite_key == data.y[id]:
                    cor_cite_sum += 1
                sum += 1
            if cor_sum / sum > best_acc:
                best_acc = cor_sum / sum
                best_pre = cor_predict_sum / sum
                best_sim = cor_sim_sum / sum
                best_cite = cor_cite_sum / sum
                best_k = now_k
                # weights = reg.coef_

        print("****************")
        print("best best_k=", best_k)
        print("weights=", weights)
        print("best predict acc=", best_pre)
        print("best sim acc=", best_sim)
        print("best cite acc=", best_cite)
        print("best acc=", best_acc)
        print("****************")
        test_predict_log[seed] = best_pre
        test_sim_log[seed] = best_sim
        test_cite_log[seed] = best_cite
        test_log[seed] = best_acc
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"seed:{seed}, 运行时间: {elapsed_time:.4f} 秒")


    print("-------------------")
    print("----------")
    print("predict:", test_predict_log)
    print(f"test: {test_predict_log.mean() * 100:.2f} ± {test_predict_log.std() * 100:.2f}")
    print("----------")
    print("sim:", test_sim_log)
    print(f"test: {test_sim_log.mean() * 100:.2f} ± {test_sim_log.std() * 100:.2f}")
    print("----------")
    print("cite:", test_cite_log)
    print(f"test: {test_cite_log.mean() * 100:.2f} ± {test_cite_log.std() * 100:.2f}")
    print("----------")
    print("test:", test_log)
    print(f"test: {test_log.mean() * 100:.2f} ± {test_log.std() * 100:.2f}")
    print("-------------------")




if __name__ == '__main__':
    main()