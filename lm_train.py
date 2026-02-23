import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from utils import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import logging
from torch.utils.data import Dataset, DataLoader,  WeightedRandomSampler
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import BertModel, AlbertModel, BertConfig, BertTokenizer, RobertaModel, RobertaTokenizer, \
    RobertaConfig, AutoModel, AutoTokenizer, AutoConfig

from utils.utils import save_neighbor

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = "cuda" if torch.cuda.is_available() else "cpu"

class ArxivDataset(Dataset):
    def __init__(self, data, text, split="train"):
        """
        :param data: Data 对象（包含所有数据）
        :param split: "train", "val", or "test"（指定数据集划分）
        """
        self.data = data

        # 根据 split 选择不同的数据
        if split == "train":
            mask = data.train_mask
        elif split == "val":
            mask = data.val_mask
        elif split == "test":
            mask = data.test_mask
        else:
            raise ValueError("Invalid split! Choose from ['train', 'val', 'test'].")

        # 只选择 mask 里的数据
        self.indices = torch.where(mask)[0].long()  # 获取索引
        self.x = data.x[self.indices]
        self.y = data.y[self.indices]
        self.text = [text[i] for i in self.indices.tolist()]
        # self.text = text['prompt']

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.text[idx]

class BatchTextCall(object):
    """call function for tokenizing and getting batch text
    """

    def __init__(self, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def text2id(self, batch_text):
        return self.tokenizer(batch_text, max_length=self.max_len, add_special_tokens=True,
                              truncation=True, padding=True, return_tensors='pt')
    def title2id(self, batch_text):
        return self.tokenizer(batch_text, max_length=50, add_special_tokens=False,
                              truncation=True, padding=True, return_tensors='pt')
    def __call__(self, batch):
        batch_feature = [item[0] for item in batch]
        batch_label = [item[1] for item in batch]
        batch_text = [item[2] for item in batch]

        source = self.text2id(batch_text)
        token = source.get('input_ids').squeeze(1)
        mask = source.get('attention_mask').squeeze(1)
        label = torch.tensor(batch_label)

        return token, mask, label

def choose_bert_type(path, bert_type="tiny_albert"):
    """
    choose bert type for chinese, tiny_albert or macbert（bert）
    return: tokenizer, model
    """
    if bert_type == "albert":
        model_config = BertConfig.from_pretrained(path)
        model = AlbertModel.from_pretrained(path, config=model_config)
    elif bert_type == "bert":
        model_config = BertConfig.from_pretrained(path)
        model = BertModel.from_pretrained(path, config=model_config)
    elif bert_type == "roberta":
        model_config = RobertaConfig.from_pretrained(path)
        model = RobertaModel.from_pretrained(path, config=model_config)
    else:
        model_config, model = None, None
        print("ERROR, not choose model!")

    return model_config, model

class SentenceClsHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # classifier_dropout = (
        #     config.header_dropout_prob if config.header_dropout_prob is not None else config.hidden_dropout_prob
        # )
        self.dropout = nn.Dropout(config.header_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels, bias=False)

    def forward(self, feature):
        x = self.dropout(feature)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Sentence_Transformer(nn.Module):
    def __init__(self, args):
        super(Sentence_Transformer, self).__init__()
        pretrained_repo = args.pretrained_repo
        config = AutoConfig.from_pretrained(pretrained_repo)
        config.num_labels = args.num_labels
        config.header_dropout_prob = args.header_dropout_prob
        config.save_pretrained(save_directory=args.output_dir)
        self.without_finetune = args.without_finetune
        self.bert_model = AutoModel.from_pretrained(pretrained_repo, config=config, add_pooling_layer=False)
        self.head = SentenceClsHead(config)
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels, bias=True)
        if args.use_peft:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=args.peft_r,
                lora_alpha=args.peft_lora_alpha,
                lora_dropout=args.peft_lora_dropout,
            )
            self.bert_model = PeftModel(self.bert_model, lora_config)
            self.bert_model.print_trainable_parameters()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all-MiniLM-L6-v2 token embeddings
        data_type = token_embeddings.dtype
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(data_type)
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def average_pool(self, last_hidden_states, attention_mask):  # for E5_model
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def forward(self, input_ids, att_mask, return_hidden=False):
        if self.without_finetune:
            with torch.no_grad():
                bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
        else:
            bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
        # sentence_embeddings = self.mean_pooling(bert_out, att_mask)
        # outputs[0]=last hidden state
        sentence_embeddings = self.average_pool(bert_out.last_hidden_state, att_mask)

        out = self.head(sentence_embeddings)

        # out = self.head(sentence_embeddings)

        if return_hidden:
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            return out, sentence_embeddings
        else:
            return out

class MultiClass(nn.Module):
    """ text processed by bert model encode and get cls vector for multi classification
    """

    def __init__(self, bert_encode_model, model_config, num_classes=10, pooling_type='first-last-avg'):
        super(MultiClass, self).__init__()
        self.bert = bert_encode_model
        self.num_classes = num_classes
        self.embedding_dim = model_config.hidden_size
        self.fc = nn.Linear(model_config.hidden_size, num_classes)
        self.pooling = pooling_type


    def forward(self, batch_token, batch_attention_mask):
        out = self.bert(batch_token,
                        attention_mask=batch_attention_mask,
                        output_hidden_states=True)
        if self.pooling == 'cls':
            out = out.last_hidden_state[:, 0, :]  # [batch, 768]
        elif self.pooling == 'pooler':
            out = out.pooler_output  # [batch, 768]
        elif self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            out = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        elif self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            out = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]
        else:
            raise "should define pooling type first!"

        out_fc = self.fc(out)
        return out_fc


def evaluation(model, dataloader, loss_func):
    # model.load_state_dict(torch.load(save_path))
    model.eval()
    total_loss = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for ind, (token, mask, label) in enumerate(dataloader):
            token = token.to(device)
            mask = mask.to(device)
            label = label.to(device)

            out = model(token, mask)
            loss = loss_func(out, label)
            total_loss += loss.detach().item()

            label = label.data.cpu().numpy()
            predic = torch.max(out.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, label)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    return acc, total_loss / len(dataloader)



def train():
    label2ind_dict = pd.read_csv('./dataset/ogbn_arxiv/labelidx2arxivcategeory.csv')
    label2ind_dict = label2ind_dict.set_index('label idx')['arxiv category'].to_dict()
    torch.backends.cudnn.benchmark = True

    dataset_name = "arxiv_2023"
    dataset_name = "cora"
    # dataset_name = "arxiv"
    dataset_name = "wikics"
    # dataset_name = "citeseer"
    # dataset_name = "product"

    seeds = [0, 1, 2, 3, 4]
    batch_size = 5
    eval_batch_size = 10
    if dataset_name == 'arxiv_2023' or dataset_name == 'arxiv':
        num_classes = 40
    elif dataset_name == 'cora':
        num_classes = 7
    elif dataset_name == 'wikics':
        num_classes = 10
    elif dataset_name == 'citeseer':
        num_classes = 6
    epochs = 20
    lr = 1e-5
    only_test = True
    model_name = "all-MiniLM-L6-v2-MiniLM-L6-v2"
    model_name = "e5-large"
    # model_name = "bert"
    if model_name == 'all-MiniLM-L6-v2-MiniLM-L6-v2':
        model_path = r'./model/all-MiniLM-L6-v2-MiniLM-L6-v2'
        model_fea = 384
    elif model_name == 'bert':
        model_path = r'./model/bert-base-uncased'
        model_fea = 768
    elif model_name == 'e5-large':
        model_path = r'intfloat/e5-large'
        model_fea = 1024
    parser = argparse.ArgumentParser("des")
    parser.add_argument('--pretrained_repo', type=str, default=model_path)
    parser.add_argument('--num_labels', type=int, default=num_classes)
    parser.add_argument('--output_dir', type=str, default='./log/{}_train'.format(dataset_name))
    parser.add_argument('--hidden_size', type=int, default=model_fea)
    parser.add_argument('--use_peft', type=bool, default=False)
    parser.add_argument('--without_finetune', type=bool, default=False)
    parser.add_argument('--header_dropout_prob', type=float, default=0.2)
    parser.add_argument("--peft_r", type=int, default=8)
    parser.add_argument("--peft_lora_alpha", type=float, default=32)
    parser.add_argument("--peft_lora_dropout", type=float, default=0.1)
    args = parser.parse_args()
    logging.basicConfig(
        filename='./log/{}_{}_train.log'.format(model_name, dataset_name),  # 日志文件路径
        filemode='a',  # 模式 'a' 表示追加日志到文件，'w' 表示覆盖写入
        format='%(asctime)s - %(levelname)s - %(message)s',  # 日志输出格式
        level=logging.INFO
    )

    valid_log = np.zeros(len(seeds))
    test_log = np.zeros(len(seeds))

    if not args.without_finetune:
        if args.use_peft:
            model_name = model_name + "_lora"
        else:
            model_name = model_name + "_finetune"

    for seed in seeds:
        data, text = utils.load_data(dataset_name, use_text=True, seed=seed)
        logging.info(
            "\n {} train lr:{} seed:{}".format(model_name, lr, seed))
        print("{} train lr:{} seed:{}".format(model_name, lr, seed))

        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_repo, local_files_only=True)
        multi_classification_model = Sentence_Transformer(args)
        multi_classification_model.to(device)

        combine_text = utils.get_combine_text(text, dataset_name, tokenizer)

        # 创建 DataLoader
        train_dataset = ArxivDataset(data, combine_text, split="train")
        train_dataset_call = BatchTextCall(tokenizer, max_len=512)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=train_dataset_call)

        valid_dataset = ArxivDataset(data, combine_text, split="val")
        valid_loader = DataLoader(valid_dataset, batch_size=eval_batch_size, shuffle=True, num_workers=2, collate_fn=train_dataset_call)

        test_dataset = ArxivDataset(data, combine_text, split="test")
        test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=True, num_workers=2, collate_fn=train_dataset_call)

        if only_test:
            if args.without_finetune:
                head_state_dict = {k: v for k, v in multi_classification_model.state_dict().items() if
                                   k.startswith('head.')}
                torch.save(head_state_dict, 'head_weights.pth')
            elif args.without_finetune and args.usee_peft:
                torch.save(multi_classification_model.state_dict(),
                           './pth/{}_saved_{}_{}_model.pth'.format(dataset_name, model_name, seed))
            elif not args.without_finetune:
                state_dict = torch.load('./pth/{}_saved_{}_{}_model.pth'.format(dataset_name, model_name, seed), map_location=device)
            multi_classification_model.load_state_dict(state_dict)
            embeddings = []
            res = []
            for i in tqdm(range(0, len(combine_text), eval_batch_size), desc="processing"):
                batch = combine_text[i:i + eval_batch_size]
                # 预处理输入
                inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
                token = inputs.get('input_ids').squeeze(1)
                mask = inputs.get('attention_mask').squeeze(1)
                with torch.no_grad():  # 不计算梯度，节省内存
                    logits, sentence_emb = multi_classification_model(token, mask, return_hidden=True)

                result = torch.softmax(logits, dim=-1)
                res.append(result.cpu().numpy())
                embeddings.append(sentence_emb.cpu().numpy())
            all_predict = np.concatenate(res, axis=0)
            all_embeddings = np.concatenate(embeddings, axis=0)
            save_neighbor(predict_label, './label_predict/{}_{}_predict.json'.format(dataset_name, model_name))
            np.save("./predict/{}_{}_predict_{}.npy".format(dataset_name, model_name, seed), all_predict)
            np.save("./emb/{}_{}_embeddings_{}.npy".format(dataset_name, model_name, seed), all_embeddings)  # 仅保存 embedding
            continue

        loss_func = F.cross_entropy
        param_optimizer = list(multi_classification_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        num_train_optimization_steps = len(train_loader) * epochs
        optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=lr)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 int(len(train_loader) * 0.6),
                                                                 num_train_optimization_steps)

        #
        loss_total, top_acc, patience, no_improvement_count = [], 0, 4, 0
        eval_step = (len(train_loader) // 6) - 1
        stop_flag = False
        for epoch in range(epochs):
            multi_classification_model.train()
            tqdm_bar = tqdm(train_loader, desc="Training epoch{epoch}".format(epoch=epoch))
            for i, (token, mask, label) in enumerate(tqdm_bar):
                token = token.to(device)
                mask = mask.to(device)
                label = label.to(device)

                out = multi_classification_model(token, mask)
                loss = loss_func(out, label)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                loss_total.append(loss.detach().item())
                if (i+1) % eval_step == 0:
                    time_valid_begin = time.time()
                    valid_accuracy, valid_loss = evaluation(multi_classification_model, valid_loader, loss_func)
                    time_valid_end = time.time()
                    print("valid cost time %.4f" % (time_valid_end - time_valid_begin))
                    print("Accuracy: %.4f in valid" % valid_accuracy)
                    logging.info("epoch {} valid accuracy:{} mean loss:{}".format(epoch, valid_accuracy, valid_loss))

                    if top_acc < valid_accuracy:
                        no_improvement_count = 0
                        top_acc = valid_accuracy
                        print("begin test")
                        time_test_begin = time.time()
                        test_accuracy, test_loss = evaluation(multi_classification_model, test_loader, loss_func)
                        time_test_end = time.time()
                        print("test cost time %.4f" % (time_test_end - time_test_begin))
                        print("Accuracy: %.4f in test" % test_accuracy)
                        logging.info("epoch {} test accuracy:{} mean loss:{}".format(epoch, test_accuracy, test_loss))
                        valid_log[seed] = valid_accuracy
                        test_log[seed] = test_accuracy
                        if args.without_finetune:
                            head_state_dict = {k: v for k, v in multi_classification_model.state_dict().items() if k.startswith('head.')}
                            torch.save(head_state_dict, './pth/{}_saved_{}_{}_model.pth'.format(dataset_name, model_name, seed))
                        elif args.without_finetune and args.usee_peft:
                            torch.save(multi_classification_model.state_dict(),
                                       './pth/{}_saved_{}_{}_model.pth'.format(dataset_name, model_name, seed))
                        elif not args.without_finetune:
                            torch.save(multi_classification_model.state_dict(), './pth/{}_saved_{}_{}_model.pth'.format(dataset_name, model_name, seed))

                    else:
                        no_improvement_count += 1
                        if no_improvement_count >= patience:
                            print(f"验证准确率连续 {patience} 次未提升，停止训练。")
                            stop_flag = True
                            break
            if stop_flag:
                break
    print("valid:", valid_log)
    print(f"valid: {valid_log.mean():.4f} ± {valid_log.std():.4f}")
    print("test:", test_log)
    print(f"test: {test_log.mean():.4f} ± {test_log.std():.4f}")
    #


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='bert classification')
    # parser.add_argument("-c", "--config", type=str, default="./config.yaml")
    # args = parser.parse_args()
    # config = load_config(args.config)
    #
    # print(type(config.lr), type(config.batch_size))
    # config.lr = float(config.lr)
    train()
