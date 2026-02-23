from collections import OrderedDict

import torch
import numpy as np
import json
import os
import sys
from tqdm import tqdm
from openai import OpenAI
from scipy.spatial.distance import cosine
from utils.load_arxiv import get_raw_text_arxiv, get_raw_text_arxiv_2023
from utils.load_cora import get_raw_text_cora
from utils.load_wikics import get_raw_text_wikics
from utils.load_citeseer import get_raw_text_citeseer
from utils.load_cora_few_shot import get_cora_few_shot, get_cora500_few_shot
from utils.mysimcse import simcse
# from utils.load_cora import get_raw_text_cora
from utils.load_pubmed import get_raw_text_pubmed
# from utils.load_arxiv_2023 import get_raw_text_arxiv_2023
# from utils.load_products import get_raw_text_products
from time import sleep
from joblib import Parallel, delayed
from time import sleep
from utils.prompts import generate_system_prompt, arxiv_natural_lang_mapping

from time import sleep
from random import randint
import threading
from joblib import Parallel, delayed
from datasketch import MinHash
from heapq import nlargest

# openai.api_key  = os.environ['OPENAI_API_KEY']
client = OpenAI(api_key="sk-bd56cdaee8d14b65b0d409ca107de68c",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

def load_data_few_shot(dataset, use_text=False, seed=0, tr_num=1, val_num=20):
    if dataset == "cora":
        data, text = get_cora_few_shot(use_text, seed, tr_num, val_num)
    elif dataset == "cora500":
        data, text = get_cora500_few_shot(use_text, seed, tr_num, val_num)
    elif dataset == "pubmed":
        data, text = get_cora_few_shot(use_text, seed, tr_num, val_num)
    elif dataset == "citeseer":
        data, text = get_cora_few_shot(use_text, seed, tr_num, val_num)
    return data, text
def load_data(dataset, use_text=False, seed=0):
    """
    Load data based on the dataset name.

    Parameters:
        dataset (str): Name of the dataset to be loaded. Options are "cora", "pubmed", "arxiv", "arxiv_2023", and "product".
        use_text (bool, optional): Whether to use text data. Default is False.
        seed (int, optional): Random seed for data loading. Default is 0.

    Returns:
        Tuple: Loaded data and text information.

    Raises:
        ValueError: If the dataset name is not recognized.
    """
    if dataset == "arxiv":
        data, text = get_raw_text_arxiv(use_text, seed)
    elif dataset == "arxiv_2023":
        data, text = get_raw_text_arxiv_2023(use_text, seed)
    elif dataset == "cora":
        data, text = get_raw_text_cora(use_text, seed)
    elif dataset == "pubmed":
        data, text = get_raw_text_pubmed(use_text, seed)
    elif dataset == "arxiv":
        data, text = get_raw_text_arxiv(use_text)
    elif dataset == "wikics":
        data, text = get_raw_text_wikics(use_text, seed)
    elif dataset == "citeseer":
        data, text = get_raw_text_citeseer(use_text, seed)
    # elif dataset == "arxiv_2023":
    #     data, text = get_raw_text_arxiv_2023(use_text)
    # elif dataset == "product":
    #     data, text = get_raw_text_products(use_text)
    # else:
    #     raise ValueError("Dataset must be one of: cora, pubmed, arxiv")
    return data, text


def get_subgraph(node_idx, edge_index, hop=1):
    """
    Get subgraph around a specific node up to a certain hop.

    Parameters:
        node_idx (int): Index of the node.
        edge_index (torch.Tensor): Edge index tensor.
        hop (int, optional): Number of hops around the node to consider. Default is 1.

    Returns:
        list: Lists of nodes for each hop distance.
    """

    current_nodes = torch.tensor([node_idx])
    all_hops = []

    for _ in range(hop):
        # mask = torch.isin(edge_index[0], current_nodes)
        mask = torch.isin(edge_index[0], current_nodes) | torch.isin(edge_index[1], current_nodes)

        # Add both the source and target nodes involved in the edges 
        # new_nodes = torch.unique(edge_index[1][mask])
        new_nodes = torch.unique(torch.cat((edge_index[0][mask], edge_index[1][mask])))

        # Remove the current nodes to get only the new nodes added in this hop
        diff_nodes_set = set(new_nodes.numpy()) - set(current_nodes.numpy())
        diff_nodes = torch.tensor(list(diff_nodes_set))

        all_hops.append(diff_nodes.tolist())

        # Update current nodes for the next iteration
        current_nodes = torch.unique(torch.cat((current_nodes, new_nodes)))

    return all_hops


def get_sub(node_idx, edge_index):
    current_nodes = torch.tensor([node_idx])
    all_hops = []
    for _ in range(3):
        mask = torch.isin(edge_index[0], current_nodes)

        # Add both the source and target nodes involved in the edges
        new_nodes = torch.unique(edge_index[1][mask])

        # Remove the current nodes to get only the new nodes added in this hop
        diff_nodes_set = set(new_nodes.numpy()) - set(current_nodes.numpy())
        diff_nodes = torch.tensor(list(diff_nodes_set))

        all_hops.append(diff_nodes.tolist())

        # Update current nodes for the next iteration
        current_nodes = torch.unique(torch.cat((current_nodes, new_nodes)))

    if len(current_nodes) < 100:
        current_cited_nodes = torch.tensor([node_idx])
        for _ in range(2):
            mask = torch.isin(edge_index[1], current_cited_nodes)
            new_nodes = torch.unique(edge_index[0][mask])
            diff_nodes_set = set(new_nodes.numpy()) - set(current_cited_nodes.numpy())
            diff_nodes = torch.tensor(list(diff_nodes_set))
            all_hops.append(diff_nodes.tolist())
            current_cited_nodes = torch.unique(torch.cat((current_cited_nodes, new_nodes)))
            if len(current_nodes) + len(current_cited_nodes) > 50:
                break
    return all_hops


def sample_test_nodes(data, text, sample_size, dataset):
    """
    Randomly sample test nodes for evaluation.

    Parameters:
        data: Graph data object.
        text: Textual information associated with nodes.
        sample_size (int): Number of test nodes to sample.
        dataset (str): Name of the dataset being used.

    Returns:
        list: Indices of sampled test nodes.
    """

    np.random.seed(42)
    test_indices = np.where(data.test_mask.numpy())[0]

    if dataset != "product":
        sampled_indices = np.random.choice(test_indices, size=sample_size, replace=False)
        sampled_indices = sampled_indices.tolist()

    else:
        # Sample 2 times the sample size
        # node_indices = sample_test_nodes(data, 2 * sample_size)
        sampled_indices_double = np.random.choice(test_indices, size=2 * sample_size, replace=False)

        # Filter out the indices of nodes with title "NA\n"
        sampled_indices = [node_idx for i, node_idx in enumerate(sampled_indices_double)
                           if text['title'][node_idx] != "NA\n"]
        sampled_indices = sampled_indices[:sample_size]

        # sanity check
        count = 0
        for node_idx in sampled_indices:
            if text['title'][node_idx] == "NA\n":
                count += 1
        assert count == 0
        assert len(sampled_indices) == sample_size

    return sampled_indices


def get_completion_from_messages(messages,
                                 model="qwen-turbo",
                                 temperature=0, max_tokens=500):
    """
    Get completion from the OpenAI API based on the given messages.

    Parameters:
        messages (list): Messages to be sent to the OpenAI API.
        model (str, optional): The name of the model to be used. Default is "gpt-3.5-turbo".
        temperature (float, optional): Sampling temperature. Default is 0.
        max_tokens (int, optional): Maximum number of tokens for the response. Default is 500.

    Returns:
        str: The content of the completion message.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False
    )

    return response.choices[0].message.content


def map_arxiv_labels(data, text, source, arxiv_style):
    """
    Map arXiv labels based on the given source and mapping style.

    Parameters:
        data: Graph data object.
        text: Textual information associated with nodes.
        source (str): Data source, e.g., "arxiv".
        arxiv_style (str): Style of arXiv label mapping, either "identifier" or "natural language".

    Returns:
        Updated text information with new labels.
    """

    if source == "arxiv":
        if arxiv_style == "identifier":
            for i in range(len(data.y)):
                text['label'][i] = "arxiv " + text['label'][i].lower()
        elif arxiv_style == "natural language":
            for i in range(len(data.y)):
                text['label'][i] = arxiv_natural_lang_mapping[text['label'][i]]
    return text


def get_important_neighbors(node_index, neighbors, text, dataset, max_papers_1=5, k=5):
    """
    Get indices of important neighboring nodes.

    Parameters:
        node_index (int): Index of the target node.
        neighbors (list): List of neighboring node indices.
        text: Textual information associated with nodes.
        dataset (str): The name of the dataset being used.
        max_papers_1 (int, optional): Maximum number of neighbor papers for the first hop. Default is 5.
        k (int, optional): Number of most important neighbors to return. Default is 5.

    Returns:
        list: Indices of important neighbors.
    """

    # Get the title of the target node
    target_title = text['title'][node_index]

    Target_word = "Product" if dataset == "product" else "Paper"

    # Create a message to ask the model for the most important papers
    message = {'role': 'system',
               'content': f'The {Target_word.lower()} of interest is "{target_title}". Please return a Python list of at most {k} indices of the most related {Target_word.lower()}s among the following neighbors, ordered from most related to least related. If there are fewer than {k} neighbors, just rank the neighbors by relevance. The list should look like this: [1, 2, 3, ...]'}

    # Limit the number of neighbors based on max_papers_1
    limited_neighbors = neighbors[:max_papers_1]

    # Add the titles of each neighbor to the message
    idx_to_neighbor = {}
    for i, neighbor_idx in enumerate(limited_neighbors, start=1):
        neighbor_title = text['title'][neighbor_idx]
        message['content'] += f"\n{i}: {neighbor_title}"
        idx_to_neighbor[i] = neighbor_idx

    message['content'] += "\n\nAnswer:\n\n"

    print(f"Message: {message['content']}")

    response = get_completion_from_messages([message], model="qwen-turbo")

    print(f"Response: {response}")

    # Assume the model's response is a Python list of indices of the most important neighbors
    # Extract these indices from the response
    try:
        important_neighbors_indices = [idx_to_neighbor[idx] for idx in eval(response) if idx in idx_to_neighbor]
    except:
        print("Unable to parse the response as a Python list.")
        return []

    print(f"Important neighbors indices: {important_neighbors_indices}")

    return important_neighbors_indices


def handle_important_neighbors(node_index, text, dataset, all_hops, data, abstract_len, include_label, max_papers_1):
    """
    Handle important neighbors when attention is used.

    Parameters:
        node_index (int): Index of the target node.
        text: Textual information of the node.
        dataset (str): The name of the dataset.
        all_hops (list): List of all-MiniLM-L6-v2 neighbor nodes up to a certain hop.
        data: Graph data object.
        abstract_len (int): Length of the abstract to consider.
        include_label (bool): Whether to include labels.
        max_papers_1 (int): Maximum number of papers for the first hop.

    Returns:
        str: String containing information about important neighbors.
    """

    prompt_str = ""
    Target_word = "Product" if dataset == "product" else "Paper"
    k = 5
    attention_dir = f"attention/{dataset}/attention_{k}"
    filename = f"{attention_dir}/{node_index}.json"

    if os.path.exists(filename):
        with open(filename, "r") as f:
            important_neighbors = json.load(f)
    else:
        neighbors = list(set(all_hops[0]))
        important_neighbors = get_important_neighbors(node_index, neighbors, text, max_papers_1, k)
        important_neighbors = [int(x) for x in important_neighbors]

        if not os.path.exists(attention_dir):
            os.makedirs(attention_dir)
        with open(filename, "w") as f:
            json.dump(important_neighbors, f)

    if len(important_neighbors) > 0:
        prompt_str += f"It has following important neighbors, from most related to least related:\n"
        for i, neighbor_idx in enumerate(important_neighbors):
            neighbor_title = text['title'][neighbor_idx]
            prompt_str += f"{Target_word} {i + 1} title: {neighbor_title}\n"
            if abstract_len > 0:
                neighbor_abstract = text['abs'][neighbor_idx]
                prompt_str += f"{Target_word} {i + 1} abstract: {neighbor_abstract[:abstract_len]}\n"
            if include_label and (data.train_mask[neighbor_idx] or data.val_mask[neighbor_idx]):
                label = text['label'][neighbor_idx]
                prompt_str += f"Label: {label}\n"
    return prompt_str


def handle_standard_neighbors(node_index, text, all_hops, data, hop, max_papers_1,
                              max_papers_2, abstract_len, include_label, dataset):
    """
    Handle neighbors when attention is not used.

    Parameters:
        node_index (int): Index of the target node.
        text: Textual information of the node.
        all_hops (list): List of all-MiniLM-L6-v2 neighbor nodes up to a certain hop.
        data: Graph data object.
        hop (int): Number of hops to consider.
        max_papers_1 (int): Maximum number of papers for the first hop.
        max_papers_2 (int): Maximum number of papers for the second hop.
        abstract_len (int): Length of the abstract to consider.
        include_label (bool): Whether to include labels.
        dataset (str): Name of the dataset being used.

    Returns:
        str: String containing information about standard neighbors.
    """

    prompt_str = ""
    Target_word = "Product" if dataset == "product" else "Paper"

    for h in range(0, hop):
        neighbors_at_hop = all_hops[h]
        neighbors_at_hop = np.array(neighbors_at_hop)
        neighbors_at_hop = np.unique(neighbors_at_hop)
        if h == 0:
            neighbors_at_hop = neighbors_at_hop[:max_papers_1]
        else:
            neighbors_at_hop = neighbors_at_hop[:max_papers_2]

        if len(neighbors_at_hop) > 0:
            if dataset != 'product':
                prompt_str += f"The following papers were cited or quoted in this paper:\n"
                # prompt_str += f"It has following neighbor papers at hop {h+1}:\n"
            else:
                prompt_str += f"It has following neighbor products purchased toghther at hop {h + 1}:\n"

            for i, neighbor_idx in enumerate(neighbors_at_hop):
                neighbor_title = text['title'][neighbor_idx]
                prompt_str += f"{Target_word} {i + 1} title: {neighbor_title}\n"

                if abstract_len > 0:
                    neighbor_abstract = text['abs'][neighbor_idx]
                    prompt_str += f"{Target_word} {i + 1} abstract: {neighbor_abstract[:abstract_len]}\n"

                if include_label and (data.train_mask[neighbor_idx] or data.val_mask[neighbor_idx]):
                    label = text['label'][neighbor_idx]
                    prompt_str += f"Label: {label}\n"
    return prompt_str


def handle_semantic_and_structure_neighbors(text, all_hops, data, abstract_len, include_label, dataset):
    prompt_str = ""
    Target_word = "Product" if dataset == "product" else "Paper"
    if len(all_hops) > 0:
        if dataset != 'product':
            prompt_str += f"Reference papers:\n"
            # prompt_str += f"It has following neighbor papers at hop {h+1}:\n"
        else:
            prompt_str += f"It has following neighbor products purchased toghther :\n"

        for i, neighbor_idx in enumerate(all_hops):
            neighbor_title = text['title'][neighbor_idx]
            if dataset == 'cora':
                prompt_str += f"{Target_word} {i + 1}: {neighbor_title} "
            else:
                prompt_str += f"{Target_word} {i + 1} title: {neighbor_title}"

            if abstract_len > 0:
                neighbor_abstract = text['abs'][neighbor_idx]
                prompt_str += f"{Target_word} {i + 1} abstract: {neighbor_abstract[:abstract_len]}\n"
            if include_label and data.train_mask[neighbor_idx]:
            # if include_label:
                label = text['label'][neighbor_idx]
                prompt_str += f", Label: {label}\n"
            else:
                prompt_str += "\n"
    return prompt_str


def get_node_info(node_indices, data, text, mode, dataset, source, hop=1, max_papers_1=20, max_papers_2=10,
                  abstract_len=0, print_prompt=True, include_label=False, return_message=False,
                  arxiv_style=False, include_options=False, include_abs=False, zero_shot_CoT=False,
                  few_shot=False, use_attention=False, neighbours=None):
    """
    Main function to get node information based on various modes and options.

    Parameters:
        node_indices (list): List of node indices to consider.
        data: Graph data object.
        text: Textual information associated with nodes.
        mode (str): Mode of operation, either 'neighbors' or 'ego'.
        dataset (str): Name of the dataset being used.
        source (str): Source of the data.
        hop (int, optional): Number of hops to consider. Default is 1.
        max_papers_1 (int, optional): Maximum number of papers for the first hop. Default is 20.
        max_papers_2 (int, optional): Maximum number of papers for the second hop. Default is 10.
        abstract_len (int, optional): Length of the abstract to consider. Default is 0.
        print_prompt (bool, optional): Whether to print the prompt. Default is True.
        include_label (bool, optional): Whether to include labels. Default is False.
        return_message (bool, optional): Whether to return the message. Default is False.
        arxiv_style (bool, optional): Whether to use arXiv style for labels. Default is False.
        include_options (bool, optional): Whether to include options in the system prompt. Default is False.
        include_abs (bool, optional): Whether to include abstracts. Default is False.
        zero_shot_CoT (bool, optional): Whether to use zero-shot CoT. Default is False.
        few_shot (bool, optional): Whether to use few-shot learning. Default is False.
        use_attention (bool, optional): Whether to use attention. Default is False.

    Returns:
        Depending on the 'return_message' flag, either prints the prompt and ideal answer or returns a list of messages.
    """

    for node_index in node_indices:
        if mode == 'neighbors':
            # Initial setup for neighbors mode
            title = text['title'][node_index]
            prompt_str = f"Title: {title}\n"

            # Include abstract if required
            if include_abs:
                if source == 'product':
                    content = text['content'][node_index]
                    prompt_str = f"Content: {content}\n" + prompt_str
                else:
                    abstract = text['abs'][node_index]
                    prompt_str = f"Abstract: {abstract}\n" + prompt_str

            sys_prompt_str = generate_system_prompt(source, arxiv_style=arxiv_style, cora_style='score',
                                                    include_options=include_options)
            # all_hops = get_subgraph(node_index, data.edge_index, hop)

            # Check for test nodes
            if data.train_mask[node_index] or data.val_mask[node_index]:
                print('node indices should only contain test nodes!!')

            # Handle neighbors based on attention
            # if use_attention:
            #     prompt_str += handle_important_neighbors(node_index, text, dataset, all_hops, data, abstract_len, include_label, max_papers_1)
            # else:
            #     prompt_str += handle_standard_neighbors(node_index, text, all_hops, data, hop, max_papers_1, max_papers_2,
            #                                             abstract_len, include_label, dataset)
            prompt_str += handle_semantic_and_structure_neighbors(text, neighbours, data, abstract_len, include_label,
                                                                  dataset)
            # Finalize prompt for neighbors mode
            prompt_str += "Do not give any reasoning or logic for your answer.\nAnswer: \n\n"

            # Return the message
            if return_message:
                return [{'role': 'system', 'content': sys_prompt_str}, {'role': 'user', 'content': f"{prompt_str}"}]

        elif mode == 'ego':
            # Formulate the prompt
            sys_prompt_str_abs = generate_system_prompt(source, arxiv_style, include_options=include_options)

            title = text['title'][node_index]

            few_shot_examples = ""
            if few_shot:
                with open(f"few_shot_examples/{dataset}.txt", 'r') as f:
                    few_shot_examples = f.read()

            # Check if the source is a product
            if source == 'product':
                content = text['content'][node_index]
                if include_abs:
                    prompt_str = f"{few_shot_examples}\nContent: {content}\nTitle: {title}\n"
                else:
                    prompt_str = f"{few_shot_examples}\nTitle: {title}\n"
            else:
                abstract = text['abs'][node_index]
                if include_abs:
                    prompt_str = f"{few_shot_examples}\nAbstract: {abstract}\nTitle: {title}\n"
                else:
                    prompt_str = f"{few_shot_examples}\nTitle: {title}\n"

            if zero_shot_CoT:
                prompt_str += "Answer: \n\n Let's think step by step.\n"
            else:
                prompt_str += "Do not provide your reasoning.\nAnswer: \n\n"

            if return_message:
                return [{'role': 'system',
                         'content': sys_prompt_str_abs},
                        {'role': 'user',
                         'content': f"{prompt_str}"}]

        else:
            print('Invalid mode! Please use either "neighbors" or "abstract"')


def get_matched_option(prediction, valid_options):
    """
    Extracts options from the prediction string and returns the last matched option.

    Parameters:
    - prediction (str): The prediction string containing potential options.
    - valid_options (set): The set of valid options to match against.

    Returns:
    - str: The last matched option or an empty string if no matches are found.
    """
    matched_options = []

    # Iteratively check each substring of the prediction
    for option in valid_options:
        if option in prediction:
            matched_options.append(option)

    # Return the last matched option if available, else return an empty string
    return matched_options[-1] if matched_options else ""


def print_node_info_and_compare_prediction(node_index, data, text, include_label, dataset, source,
                                           abstract_len=0, hop=1, mode="neighbors", max_papers_1=15,
                                           max_papers_2=5, print_out=False, print_prompt=False, arxiv_style=False,
                                           include_options=False, include_abs=False, zero_shot_CoT=False,
                                           few_shot=False, use_attention=False, options=None, neighbours=None):
    """
    Print node information, generate a message, and compare the generated message with the ideal answer.

    Parameters:
        node_index (int): Index of the node.
        data: Graph data object.
        text: Textual information associated with nodes.
        include_label (bool): Whether to include labels.
        dataset (str): Name of the dataset being used.
        source (str): Source of the data.
        abstract_len (int, optional): Length of the abstract. Default is 0.
        hop (int, optional): Number of hops to consider. Default is 1.
        mode (str, optional): Mode of operation, either 'neighbors' or 'ego'. Default is 'neighbors'.
        max_papers_1 (int, optional): Maximum number of papers for the first hop. Default is 15.
        max_papers_2 (int, optional): Maximum number of papers for the second hop. Default is 5.
        print_out (bool, optional): Whether to print the output. Default is False.
        print_prompt (bool, optional): Whether to print the prompt. Default is False.
        arxiv_style (bool, optional): Whether to use arXiv style for labels. Default is False.
        include_options (bool, optional): Whether to include options in the system prompt. Default is False.
        include_abs (bool, optional): Whether to include abstracts. Default is False.
        zero_shot_CoT (bool, optional): Whether to use zero-shot CoT. Default is False.
        few_shot (bool, optional): Whether to use few-shot learning. Default is False.
        use_attention (bool, optional): Whether to use attention. Default is False.
        options (set, optional): Set of valid options. Required if zero_shot_CoT is True.

    Returns:
        int: Returns 1 if the prediction is correct, otherwise 0.
    """

    message = get_node_info([node_index], data, text, hop=hop, dataset=dataset, source=source,
                            mode=mode, max_papers_1=max_papers_1, max_papers_2=max_papers_2, return_message=True,
                            include_label=include_label, abstract_len=abstract_len, print_prompt=print_prompt,
                            arxiv_style=arxiv_style, include_options=include_options,
                            zero_shot_CoT=zero_shot_CoT, few_shot=few_shot, include_abs=include_abs,
                            use_attention=use_attention, neighbours=neighbours)

    if print_out:
        print(message[0]['content'], end="\n\n")
        print(message[1]['content'], end="\n\n")

    ideal_answer = text['label'][node_index]

    print("Ideal_answer:", ideal_answer, end="\n\n")

    # Get completion message and print
    response = get_completion_from_messages(message, model="qwen-max")
    if print_out:
        print(response)

    if source == "arxiv" and arxiv_style == "identifier":
        response = response.lower()

    prediction = response if response is not None else ""
    if zero_shot_CoT:
        # Use the helper function to get the last matched option
        if options == None:
            raise "options is not define!"
        prediction = get_matched_option(prediction, options)

    if prediction is not None:
        print("Prediction: ", prediction)

        # Compare the prediction with ideal_answer
        print("Is prediction correct? ", prediction == ideal_answer, end="\n\n")

        return int(prediction == ideal_answer)
    else:
        print("No valid prediction could be made.")


def get_combine_text(text, dataset_name, tokenizer):
    if dataset_name == "cora" or dataset_name == "pubmed" or dataset_name == "arxiv" or dataset_name == "arxiv_2023":
        title = text['title']
        abstract = text['abs']
        combine_text = [str(title[i]) + tokenizer.pad_token + str(abstract[i]) for i in range(len(title))]
        return combine_text

    elif dataset_name == "wikics" or dataset_name == 'citeseer':
        return text
    elif dataset_name == "product":
        title = text['title']
        abstract = text['content']
        combine_text = ["Title: " + title[i] + "Content: " + abstract[i] for i in range(len(title))]
        return combine_text
    else:
        raise NotImplementedError


def get_e5_Instruct(text, dataset_name, data):
    if dataset_name == "cora" or dataset_name == "pubmed" or dataset_name == "arxiv" or dataset_name == "arxiv_2023":
        title = text['title']
        abstract = text['abs']
        combine_text = []
        for i in range(len(title)):
            if i in data.train_id:
                combine_text.append('passage:' + str(title[i]) + str(abstract[i]))
            else:
                combine_text.append('query:Search for related paper based on relevant information:' + str(title[i]) + str(abstract[i]))
        # combine_text = ['passage:' + str(title[i]) + str(abstract[i]) for i in range(len(title))]
        return combine_text
    elif dataset_name == "citeseer":
        return text['text']
    elif dataset_name == "product":
        title = text['title']
        abstract = text['content']
        combine_text = ["Title: " + title[i] + "Content: " + abstract[i] for i in range(len(title))]
        return combine_text
    else:
        raise NotImplementedError


def get_qwen_instruct(text, dataset_name):
    instruct = "Instruct: Given the title and abstract of a paper, retrieve relevant papers.\nQuery: "
    if dataset_name == "cora" or dataset_name == "pubmed" or dataset_name == "arxiv" or dataset_name == "arxiv_2023":
        title = text['title']
        abstract = text['abs']
        combine_text = [instruct + 'title: ' + str(title[i]) + ", abstract:" + str(abstract[i]) for i in range(len(title))]
        return combine_text
    elif dataset_name == "citeseer":
        return text['text']
    elif dataset_name == "product":
        title = text['title']
        abstract = text['content']
        combine_text = ["Title: " + title[i] + "Content: " + abstract[i] for i in range(len(title))]
        return combine_text
    else:
        raise NotImplementedError


def get_rag_prompt(data, text, dataset_name, ids, titles, abstracts, labels, source, use_label_guidance=False, predict_model=None):
    combine_text = {'prompt': [], 'label': []}
    if use_label_guidance:
        predict_dict = load_neighbor("./label_predict/{}_{}_predict.json".format(dataset_name, predict_model))
    for id in ids:
        if isinstance(id, torch.Tensor):
            id = id.item()
        title = titles[id]
        abstract = abstracts[id]
        if use_label_guidance:
            predict = predict_dict.get(id)[0:3]
            sys_prompt_str = generate_system_prompt(source=source, predict=predict, arxiv_style="part")
        else:
            sys_prompt_str = generate_system_prompt(include_options=True, source=source, predict=[],
                                                    arxiv_style="subcategory")
        # if data.y[id] not in predict:
        #     predict.append(data.y[id].item())
        prompt_str = sys_prompt_str + f"\nAbstract: {abstract}\nTitle: {title}\n"
        prompt_str += "Reference paper:\n"

        combine_text['prompt'].append(prompt_str)
        combine_text['label'].append(labels[id])

    return combine_text

def get_prompt(data, text, dataset_name, ids, titles, abstracts, labels, structure_neighbor_dict,
               semantic_neighbor_dict, BM25_neighbor_dict, source, use_neighbor=False, use_label_guidance=False, predict_model=None):
    combine_text = {'prompt': [], 'label': []}
    if use_label_guidance:
        predict_dict = load_neighbor("./label_predict/{}_{}_predict.json".format(dataset_name, predict_model))
    k = 5
    for id in ids:
        if isinstance(id, torch.Tensor):
            id = id.item()
        title = titles[id]
        abstract = abstracts[id]
        if use_label_guidance:
            predict = predict_dict.get(id)[0:3]
            sys_prompt_str = generate_system_prompt(source=source, predict=predict, arxiv_style="part")
        else:
            sys_prompt_str = generate_system_prompt(include_options=True, source=source, predict=[],
                                                    arxiv_style="subcategory")
        # if data.y[id] not in predict:
        #     predict.append(data.y[id].item())
        prompt_str = sys_prompt_str + f"\nAbstract: {abstract}\nTitle: {title}\n"
        if use_neighbor:
            structure_node_neighbours = structure_neighbor_dict.get(id)
            semantic_node_neighbours = semantic_neighbor_dict.get(id)
            bm25_node_neighbours = BM25_neighbor_dict.get(id)
            structure_node_neighbours = [item[0] for item in structure_node_neighbours[:k]]
            semantic_node_neighbours = [item[0] for item in semantic_node_neighbours[:k]]
            bm25_node_neighbours = [item[0] for item in bm25_node_neighbours[:k]]
            # structure_node_neighbours = [item[0] for item in structure_node_neighbours[:k] if item[1] > 0]
            # semantic_node_neighbours = [item[0] for item in semantic_node_neighbours[:k] if item[1] > 0.9]

            node_neighbours = list(OrderedDict.fromkeys(semantic_node_neighbours + bm25_node_neighbours + structure_node_neighbours))
            # node_neighbours = semantic_node_neighbours
            prompt_str += handle_semantic_and_structure_neighbors(text, node_neighbours, data, 0, True, dataset_name)

        combine_text['prompt'].append(prompt_str)
        combine_text['label'].append(labels[id])

    return combine_text


def get_prompt_text(data, text, dataset_name, source, use_neighbor, use_label_guidance, model_name):
    train_ids = list(data.train_id)
    valid_ids = list(data.val_id)
    test_ids = list(data.test_id)
    # structure_neighbor_dict = None
    # semantic_neighbor_dict = None
    if use_neighbor:
        structure_neighbor_dict, semantic_neighbor_dict, BM25_neighbor_dict = get_semantic_and_structure_neighbors(
                                                                                               hop=3,
                                                                                               neighbor_file=True,
                                                                                               dataset_name=dataset_name,
                                                                                               model_name=model_name)
    else:
        structure_neighbor_dict = None
        semantic_neighbor_dict = None
        BM25_neighbor_dict = None
    if dataset_name == "cora" or dataset_name == "pubmed" or dataset_name == "arxiv" or dataset_name == "arxiv_2023":
        titles = text['title']
        abstracts = text['abs']
        labels = text['label']
        train_prompt_text = get_prompt(data, text, dataset_name, train_ids, titles, abstracts, labels,
                                       structure_neighbor_dict,
                                       semantic_neighbor_dict,
                                       BM25_neighbor_dict, source, use_neighbor, use_label_guidance, model_name)
        valid_prompt_text = get_prompt(data, text, dataset_name, valid_ids, titles, abstracts, labels,
                                       structure_neighbor_dict,
                                       semantic_neighbor_dict,
                                       BM25_neighbor_dict, source, use_neighbor, use_label_guidance, model_name)
        test_prompt_text = get_prompt(data, text, dataset_name, test_ids, titles, abstracts, labels,
                                      structure_neighbor_dict,
                                      semantic_neighbor_dict,
                                      BM25_neighbor_dict, source, use_neighbor, use_label_guidance, model_name)
        return train_prompt_text, valid_prompt_text, test_prompt_text
    elif dataset_name == "citeseer":
        return text['text']
    elif dataset_name == "product":
        title = text['title']
        abstract = text['content']
        combine_text = ["Title: " + title[i] + "Content: " + abstract[i] for i in range(len(title))]
        return combine_text
    else:
        raise NotImplementedError


def get_rag_prompt_text(data, text, dataset_name, source, use_label_guidance, model_name):
    train_ids = list(data.train_id)
    valid_ids = list(data.val_id)
    test_ids = list(data.test_id)
    if dataset_name == "cora" or dataset_name == "pubmed" or dataset_name == "arxiv" or dataset_name == "arxiv_2023":
        titles = text['title']
        abstracts = text['abs']
        labels = text['label']
        train_prompt_text = get_rag_prompt(data, text, dataset_name, train_ids, titles, abstracts, labels,
                                            source, use_label_guidance, model_name)
        valid_prompt_text = get_rag_prompt(data, text, dataset_name, valid_ids, titles, abstracts, labels,
                                           source, use_label_guidance, model_name)
        test_prompt_text = get_rag_prompt(data, text, dataset_name, test_ids, titles, abstracts, labels,
                                          source, use_label_guidance, model_name)
        return train_prompt_text, valid_prompt_text, test_prompt_text
    elif dataset_name == "citeseer":
        return text['text']
    elif dataset_name == "product":
        title = text['title']
        abstract = text['content']
        combine_text = ["Title: " + title[i] + "Content: " + abstract[i] for i in range(len(title))]
        return combine_text
    else:
        raise NotImplementedError

def save_neighbor(neighbor, path):
    neighbor_dict_str_keys = {str(key): value for key, value in neighbor.items()}
    with open(path, "w") as file:
        json.dump(neighbor_dict_str_keys, file)


def load_neighbor(filename):
    with open(filename) as file:
        neighbor_dict = json.load(file)
    neighbor_dict = {int(key): value for key, value in neighbor_dict.items()}
    return neighbor_dict


def get_top_k_neighbor_with_label_BM25(node_id, search, text, k):
    title = text['title'][node_id]
    abstract = text['abs'][node_id]
    contents = f"{title}. {abstract}"
    hits = search.search(contents, k=k+1)

    cur_out = [[int(hits[i].docid), float(hits[i].score)] for i in range(len(hits)) if int(hits[i].docid) != node_id]
    return cur_out


def get_top_k_neighbor_with_label_simcse(node_id, index, node_emb, k):
    # hop = 2
    # while len(neighbors) < 20:
    #     all_hops = get_subgraph(node_id, data.edge_index, hop)
    #     neighbors = [element for sublist in all_hops for element in sublist]
    #     if len(all_hops[hop-1]) == 0:
    #         break
    #     hop += 1
    cur_node_emb = node_emb[node_id].reshape(1, -1)
    distances, indices = index.search(cur_node_emb, k + 1)

    cur_out = [[int(indices[0][i]), float(distances[0][i])] for i in range(k + 1) if int(indices[0][i]) != node_id]

    return cur_out


def get_category_simcse(node_id, index, node_emb, k):
    cur_node_emb = node_emb[node_id].reshape(1, -1)
    distances, indices = index.search(cur_node_emb, k + 1)

    cur_out = [[int(indices[0][i]), float(distances[0][i])] for i in range(k + 1) if int(indices[0][i]) != node_id]

    return cur_out

def get_top_k_neighbor_with_structure(neighbors, node_neighbors, k):
    # neighbors = []
    # hop = 2
    # while len(neighbors) < 20:
    #     all_hops = get_subgraph(node_id, data.edge_index, hop)
    #     neighbors = [element for sublist in all_hops for element in sublist]
    #     if len(all_hops[hop-1]) == 0:
    #         break
    #     hop += 1
    results = []
    neighbors = set(neighbors)
    for neighbor in neighbors:
        neighbor_neighbors = node_neighbors.get(str(neighbor))
        if neighbor_neighbors is None:
            raise ValueError("get neighbor error")

        intersection = len(neighbors & neighbor_neighbors)
        union = len(neighbors | neighbor_neighbors)
        jaccard = intersection / union if union != 0 else 0
        results.append((neighbor, jaccard))
    sorted_score = sorted(results, key=lambda item: item[1], reverse=True)
    if len(neighbors) > k:
        cur_out = [sorted_score[i][0] for i in range(k)]
    else:
        cur_out = [sorted_score[i][0] for i in range(len(neighbors))]
    return cur_out


def jaccard_for_node(node, neighbor_dict):
    result = []
    if isinstance(node, torch.Tensor):
        node = node.item()
    set1 = neighbor_dict.get(str(node), None)
    for neighbor in set1:
        set2 = neighbor_dict.get(str(neighbor), None)
        inter = len(set1 & set2)
        union = len(set1 | set2)
        score = inter / union if union else 0.0
        result.append((neighbor, score))
    sorted_score = sorted(result, key=lambda item: item[1], reverse=True)
    if len(set1) > 10:
        cur_out = [sorted_score[i][0] for i in range(10)]
    else:
        cur_out = [sorted_score[i][0] for i in range(len(set1))]
    return node, cur_out


def compute_signature_for_node(neighbors, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for n in neighbors:
        m.update(str(n).encode('utf8'))
    return m


def get_signature_vec(neighbors, num_perm=128):
    m = MinHash(num_perm=num_perm)
    if neighbors is None:
        return np.zeros(num_perm, dtype=np.uint64)
    for n in neighbors:
        m.update(str(n).encode('utf8'))
    return np.array(m.digest(), dtype=np.uint64)


def get_minhash_signatures(node_id, node_neighbor, num_perm=128, load_file=False, file_name=None):
    minhash_signatures = []
    structure_neighbor_dict = {}
    if load_file:
        minhash_signatures_array = np.load(file_name)
    else:
        for node in tqdm(range(len(node_id)), desc="calculate minhash_signatures"):
            neighbors = node_neighbor.get(str(node))
            minhash_signatures.append(get_signature_vec(neighbors, num_perm))
        minhash_signatures_array = np.array(minhash_signatures)
        np.save(file_name, minhash_signatures_array)

    for node in tqdm(node_id, desc="calculate Jaccard"):
        if isinstance(node, torch.Tensor):
            node = node.item()
        neighbors = node_neighbor.get(str(node))

        sig_u = minhash_signatures_array[node]
        sig_neighbors = minhash_signatures_array[neighbors]
        similarities = np.mean(sig_neighbors == sig_u, axis=1)  # 计算相等比例

        top_k = nlargest(10, zip(neighbors, similarities), key=lambda x: x[1])
        structure_neighbor_dict[node] = top_k
        # sims = []
        # for v in neighbors:
        #     nei_neighbors = node_neighbor.get(str(v))
        #     sig_v = get_signature(v, nei_neighbors)
        #     sim = np.mean(sig_u == sig_v)
        #     sims.append((v, sim))
        # top_k = nlargest(10, sims, key=lambda x: x[1])
        # structure_neighbor_dict[node] = top_k

    return structure_neighbor_dict


def get_semantic_and_structure_neighbors(text=None, node_id=None, node_emb=None, index=None, node_neighbor=None, hop=None,
                                         neighbor_file=None, dataset_name=None,
                                         model_name=None, BM25_searcher=None):
    emb_name = 'finetune_' + model_name
    num_perm = 128
    minihash_file = './emb/' + dataset_name + '_minhash_' + str(num_perm) + '_signatures.npy'

    if neighbor_file == None:
        semantic_neighbor_dict = {}
        BM25_dict = {}
        # structure_neighbor_dict = get_minhash_signatures(node_id, node_neighbor, num_perm=num_perm, load_file=False, file_name=minihash_file)
        for id in tqdm(node_id):
            if isinstance(id, torch.Tensor):
                id = id.item()
            BM25_neighbors = get_top_k_neighbor_with_label_BM25(id, BM25_searcher, text, k=10)
            # neighbors = node_neighbor.get(str(id))
            # if neighbors is None:
            #     raise "get neighbor error"
            # semantic_neighbors = get_top_k_neighbor_with_label_simcse(id, index, node_emb, k=10)
            # structure_neighbors = get_top_k_neighbor_with_structure(neighbors, node_neighbor, k=10)
            # structure_neighbor_dict[id] = structure_neighbors
            # semantic_neighbor_dict[id] = semantic_neighbors
            BM25_dict[id] = BM25_neighbors
        # save_neighbor(structure_neighbor_dict,
        #               'neighbor_dict/{}_{}hop_top10_minhash_structure_neighbor.json'.format(dataset_name, hop))
        # save_neighbor(semantic_neighbor_dict, 'neighbor_dict/{}_{}_top10_semantic_train_neighbor.json'.format(dataset_name, emb_name))
        save_neighbor(BM25_dict,
                      'neighbor_dict/{}_top10_BM25_neighbor.json'.format(dataset_name))
        BM25_neighbor_dict = load_neighbor(
            f"neighbor_dict/{dataset_name}_top10_BM25_neighbor.json")
        structure_neighbor_dict = load_neighbor(
            f"neighbor_dict/{dataset_name}_{hop}hop_top10_minhash_structure_neighbor.json")
        semantic_neighbor_dict = load_neighbor(
            f"neighbor_dict/{dataset_name}_{emb_name}_top10_semantic_train_neighbor.json")
    else:
        BM25_neighbor_dict = load_neighbor(
            f"neighbor_dict/{dataset_name}_top10_BM25_neighbor.json")
        structure_neighbor_dict = load_neighbor(
            f"neighbor_dict/{dataset_name}_{hop}hop_top10_minhash_structure_neighbor.json")
        semantic_neighbor_dict = load_neighbor(f"neighbor_dict/{dataset_name}_{emb_name}_top10_semantic_train_neighbor.json")

    return structure_neighbor_dict, semantic_neighbor_dict, BM25_neighbor_dict


def process_and_compare_predictions(node_index_list, data, text, dataset_name, source, hop=2,
                                    max_papers_1=20, max_papers_2=10, mode="title",
                                    include_label=True, abstract_len=0, arxiv_style=False,
                                    include_options=False, include_abs=False, zero_shot_CoT=False,
                                    few_shot=False, use_attention=False, options=None, timeout=60, seed=42):
    """
    Process and compare predictions for a list of node indices.

    Parameters:
        node_index_list (list): List of node indices to process.
        data: Graph data object.
        text: Textual information associated with nodes.
        dataset (str): Name of the dataset being used.
        source (str): Source of the data.
        hop (int, optional): Number of hops to consider. Default is 2.
        max_papers_1 (int, optional): Maximum number of papers for the first hop. Default is 20.
        max_papers_2 (int, optional): Maximum number of papers for the second hop. Default is 10.
        mode (str, optional): Mode of operation, either 'title' or other modes. Default is 'title'.
        include_label (bool, optional): Whether to include labels. Default is True.
        abstract_len (int, optional): Length of the abstract to consider. Default is 0.
        arxiv_style (bool, optional): Whether to use arXiv style for labels. Default is False.
        include_options (bool, optional): Whether to include options in the system prompt. Default is False.
        include_abs (bool, optional): Whether to include abstracts. Default is False.
        zero_shot_CoT (bool, optional): Whether to use zero-shot CoT. Default is False.
        few_shot (bool, optional): Whether to use few-shot learning. Default is False.
        use_attention (bool, optional): Whether to use attention. Default is False.
        options (set, optional): Set of valid options. Required if zero_shot_CoT is True.
        timeout (int, optional): Maximum time to wait for a function to complete. Default is 60.

    Returns:
        tuple: The first element is the accuracy of the predictions, and the second is a list of wrong indexes.
    """

    i = 0
    count = 0
    wrong_indexes = []
    base_sleep_time = 0.5  # Starting sleep time
    max_sleep_time = 60  # Maximum sleep time

    # node_emb = np.load("paper_embeddings.npy")
    # structure_neighbor_dict, semantic_neighbor_dict = get_semantic_and_structure_neighbors(data, node_index_list, node_emb, neighbor_file=True, dataset_name=dataset_name, seed=seed)

    while i < len(node_index_list):
        retries = 0
        while True:  # Infinite loop for retries
            result_container = [None]  # List to store the result of the threaded function
            exception_container = [None]  # List to store exceptions if any

            # Function to run in the thread
            def thread_target():
                try:
                    print(f"Processing index {i}...")
                    node_index = node_index_list[i]
                    # structure_node_neighbours = structure_neighbor_dict.get(node_index)
                    # semantic_node_neighbours = semantic_neighbor_dict.get(node_index)
                    # node_neighbours = list(set(structure_node_neighbours+semantic_node_neighbours))
                    result = print_node_info_and_compare_prediction(node_index, data, text, dataset=dataset_name,
                                                                    source=source,
                                                                    hop=hop, max_papers_1=max_papers_1,
                                                                    max_papers_2=max_papers_2, mode=mode,
                                                                    include_label=include_label, print_out=False,
                                                                    arxiv_style=arxiv_style,
                                                                    include_options=include_options,
                                                                    zero_shot_CoT=zero_shot_CoT, few_shot=few_shot,
                                                                    include_abs=include_abs,
                                                                    use_attention=use_attention, options=options,
                                                                    neighbours=[])
                    result_container[0] = result
                except Exception as e:
                    exception_container[0] = e

            # Start the function in a separate thread
            thread = threading.Thread(target=thread_target)
            thread.start()
            thread.join(timeout=timeout)

            if result_container[0] is not None:
                count += result_container[0]
                print(f"Prediction: {result_container[0]}")
                if result_container[0] == 0:  # If the prediction is wrong, save the index
                    wrong_indexes.append(i)
                i += 1
                break  # Exit the retry loop once the processing is successful

            # If there was an exception or timeout
            else:
                if exception_container[0]:  # If there was an exception
                    print(f"An error occurred at index {i}: {exception_container[0]}")
                else:  # If there was a timeout
                    print(f"Function timed out at index {i}")
                retries += 1
                sleep_time = min(base_sleep_time * (2 ** retries) + randint(0, 1000) / 1000, max_sleep_time)
                print(f"Retrying in {sleep_time} seconds...")
                sleep(sleep_time)

    print("Accuracy:", count / len(node_index_list))
    print("Wrong indexes:", wrong_indexes)
    print("Wrong indexes length:", len(wrong_indexes))
    assert len(wrong_indexes) == len(node_index_list) - count

    return count / len(node_index_list), wrong_indexes
