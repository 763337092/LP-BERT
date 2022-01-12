import random
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

from util_fn import save_pickle, load_pickle

DATA_PATH = '../data'

def text2tokens(text, tokenizer, max_len=None):
    # if self.tokenizer_type == "bert":
    text = str(text).lower()
    text = tokenizer.cls_token + " " + text
    wps = tokenizer.tokenize(text)
    if max_len is not None:
        wps = tokenizer.tokenize(text)[:max_len]
    wps.append(tokenizer.sep_token)
    # print(wps)
    return wps

def tokens2bert_ids(ent_ids, rel_ids, tokenizer, max_len, type):
    max_ent_len = max_len - 3 - len(rel_ids)
    ent_ids = ent_ids[:max_ent_len]
    if type == 'src':
        input_ids = [tokenizer.cls_token] + ent_ids + [tokenizer.sep_token] + rel_ids + [tokenizer.sep_token]
        mask_ids = [1] * len(input_ids)
        segment_ids = [0] * (len(ent_ids) + 2) + [1] * (len(rel_ids) + 1)
    elif type == 'tgt':
        input_ids = [tokenizer.cls_token] + ent_ids + [tokenizer.sep_token]
        mask_ids = [1] * len(input_ids)
        segment_ids = [0] * (len(ent_ids) + 2)
    else:
        print('*****tokens2bert_ids type error')

    if len(input_ids) < max_len:
        padding_num = max_len - len(input_ids)
        input_ids += [tokenizer.pad_token] * padding_num
        mask_ids += [0] * padding_num
        segment_ids += [0] * padding_num
    input_ids = tokenizer.convert_tokens_to_ids(input_ids)

    return input_ids, mask_ids, segment_ids

class LoadDataset:
    def __init__(self, dataset, cache_path, neg_times=0, sample_seed=2021, load_cache=False, debug=False):
        self.dataset = dataset
        print('*' * 50)
        print('dataset: ', self.dataset)
        print('*' * 50)
        if debug:
            _num = 100 if self.dataset != 'FB15k-237' else 20
            self.train = pd.read_csv(f'{DATA_PATH}/{dataset}/concat_train.csv', sep='\t', nrows=_num)
            self.dev = pd.read_csv(f'{DATA_PATH}/{dataset}/concat_dev.csv', sep='\t', nrows=_num)
            self.test = pd.read_csv(f'{DATA_PATH}/{dataset}/concat_test.csv', sep='\t', nrows=_num)
        else:
            self.train = pd.read_csv(f'{DATA_PATH}/{dataset}/concat_train.csv', sep='\t')
            self.dev = pd.read_csv(f'{DATA_PATH}/{dataset}/concat_dev.csv', sep='\t')
            self.test = pd.read_csv(f'{DATA_PATH}/{dataset}/concat_test.csv', sep='\t')
        if self.dataset is not 'WN18RR':
            self.train = self.make_reverse_data(self.train)
            self.dev = self.make_reverse_data(self.dev)
            self.test = self.make_reverse_data(self.test)
        print(f'Train|Dev|Test: {len(self.train)}|{len(self.dev)}|{len(self.test)}')

        if load_cache:
            self.ent_list = load_pickle(f'{cache_path}/ent_list.pkl')
            self.ent2text = load_pickle(f'{cache_path}/ent2text.pkl')
            self.ent2id = load_pickle(f'{cache_path}/ent2id.pkl')
            self.ent2short_text = load_pickle(f'{cache_path}/ent2short_text.pkl')
            self.rel_list = load_pickle(f'{cache_path}/rel_list.pkl')
            self.rel2text = load_pickle(f'{cache_path}/rel2text.pkl')
            self.rel2id = load_pickle(f'{cache_path}/rel2id.pkl')

            self.subj2objs_tr = load_pickle(f'{cache_path}/subj2objs_tr.pkl')
            self.obj2subjs_tr = load_pickle(f'{cache_path}/obj2subjs_tr.pkl')
            self.subj2objs = load_pickle(f'{cache_path}/subj2objs.pkl')
            self.obj2subjs = load_pickle(f'{cache_path}/obj2subjs.pkl')
        else:
            self.ent_list, self.ent2text, self.ent2id, self.ent2short_text = self.get_entity_info()
            self.rel_list, self.rel2text, self.rel2id = self.get_relation_info()

            self.subj2objs_tr, self.obj2subjs_tr = self.build_graph([self.train])
            self.subj2objs, self.obj2subjs = self.build_graph([self.train, self.dev, self.test])

            save_pickle(self.ent_list, f'{cache_path}/ent_list.pkl')
            save_pickle(self.ent2text, f'{cache_path}/ent2text.pkl')
            save_pickle(self.ent2id, f'{cache_path}/ent2id.pkl')
            save_pickle(self.ent2short_text, f'{cache_path}/ent2short_text.pkl')
            save_pickle(self.rel_list, f'{cache_path}/rel_list.pkl')
            save_pickle(self.rel2text, f'{cache_path}/rel2text.pkl')
            save_pickle(self.rel2id, f'{cache_path}/rel2id.pkl')
            save_pickle(self.subj2objs_tr, f'{cache_path}/subj2objs_tr.pkl')
            save_pickle(self.obj2subjs_tr, f'{cache_path}/obj2subjs_tr.pkl')
            save_pickle(self.subj2objs, f'{cache_path}/subj2objs.pkl')
            save_pickle(self.obj2subjs, f'{cache_path}/obj2subjs.pkl')

    def make_reverse_data(self, df):
        reversed_df = pd.DataFrame()
        reversed_df['entity1'] = df.entity2.values
        reversed_df['text1'] = df.text2.values
        reversed_df['entity2'] = df.entity1.values
        reversed_df['text2'] = df.text1.values
        if self.dataset == 'FB15k-237':
            reversed_df['relation'] = df.relation.apply(lambda x: '/reversed'+str(x))
        else:
            reversed_df['relation'] = df.relation.apply(lambda x: 'reversed_' + str(x))
        reversed_df['text'] = df.text.apply(lambda x: 'reversed ' + str(x))

        return pd.concat([df, reversed_df]).reset_index(drop=True)

    def get_entity_info(self):
        ent_list, ent2text, ent2id = [], {}, {}
        ent2short_text = {}
        ent_text_df = pd.concat([
            self.train[['entity1', 'text1']].rename(columns={'entity1': 'entity', 'text1': 'text'}),
            self.train[['entity2', 'text2']].rename(columns={'entity2': 'entity', 'text2': 'text'}),
            self.dev[['entity1', 'text1']].rename(columns={'entity1': 'entity', 'text1': 'text'}),
            self.dev[['entity2', 'text2']].rename(columns={'entity2': 'entity', 'text2': 'text'}),
            self.test[['entity1', 'text1']].rename(columns={'entity1': 'entity', 'text1': 'text'}),
            self.test[['entity2', 'text2']].rename(columns={'entity2': 'entity', 'text2': 'text'}),
        ]).drop_duplicates().reset_index(drop=True)
        for idx, (ent, text) in enumerate(tqdm(ent_text_df.values, desc='Making ent_list, ent2text, ent2id')):
            ent_list.append(ent)
            ent2text[ent] = text
            ent2id[ent] = idx
            if self.dataset == "WN18RR":
                ent2short_text[ent] = str(text).split(',')[0]
            elif self.dataset == "umls":
                ent2short_text[ent] = str(text).split(' -- ')[0]
        if self.dataset == "FB15k-237":
            ent2short_text_df = pd.read_csv(f'{DATA_PATH}/{self.dataset}/fb-entity2text-new.txt', sep='\t', header=None)
            print(ent2short_text_df.head())
            ent2short_text_df.columns = ['entity', 'short_text']
            ent2short_text_df['short_text'] = ent2short_text_df['short_text'].apply(lambda x: str(x).split(' -- ')[0])
            for ent, text in ent2short_text_df.values:
                ent2short_text[ent] = text
                # print('*****', ent, text)

        print(f'Entity num={len(ent_list)}')
        return ent_list, ent2text, ent2id, ent2short_text

    def get_relation_info(self):
        rel_list, rel2text, rel2id = [], {}, {}
        rel_text_df = pd.concat([
            self.train[['relation', 'text']], self.dev[['relation', 'text']], self.test[['relation', 'text']]
        ]).drop_duplicates().reset_index(drop=True)
        for idx, (rel, text) in enumerate(tqdm(rel_text_df.values, desc='Making rel_list, rel2text, rel2id')):
            rel_list.append(rel)
            rel2text[rel] = text
            rel2id[rel] = idx
        print(f'Relation num={len(rel_list)}')
        return rel_list, rel2text, rel2id

    def build_graph(self, df_list):
        # build positive graph from triplets
        subj2objs, obj2subjs = {}, {}

        for _df in df_list:
            for _head, _rel, _tail in tqdm(_df[['entity1', 'relation', 'entity2']].values, desc='build_graph'):
                if _head not in subj2objs:
                    subj2objs[_head] = dict()
                if _tail not in obj2subjs:
                    obj2subjs[_tail] = dict()
                if _rel not in subj2objs[_head]:
                    subj2objs[_head][_rel] = set()
                subj2objs[_head][_rel].add(_tail)
                if _rel not in obj2subjs[_tail]:
                    obj2subjs[_tail][_rel] = set()
                obj2subjs[_tail][_rel].add(_head)
        return subj2objs, obj2subjs

class LinkPredictionPretrainDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, df, tokenizer, max_seq_length, mlm_prob, vocab_size):
        self.dataset = dataset
        self.df = df

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # fn
        self.text2tokens = text2tokens
        self.tokens2bert_ids = tokens2bert_ids

        self.mlm_prob = mlm_prob
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        hrt_list = self.df['entity1'][item], self.df['relation'][item], self.df['entity2'][item]

        El_idxs = self.tokenizer.encode_plus(str(self.dataset.ent2text[hrt_list[0]]).lower(), add_special_tokens=False)['input_ids']
        R_idxs = self.tokenizer.encode_plus(str(self.dataset.rel2text[hrt_list[1]]).lower(), add_special_tokens=False)['input_ids']
        Er_idxs = self.tokenizer.encode_plus(str(self.dataset.ent2text[hrt_list[2]]).lower(), add_special_tokens=False)['input_ids']

        if self.dataset == 'umls':
            TargetEl_idxs = self.tokenizer.encode_plus(self.dataset.ent2text[hrt_list[0]].lower(), add_special_tokens=False)['input_ids']
            TargetEr_idxs = self.tokenizer.encode_plus(self.dataset.ent2text[hrt_list[2]].lower(), add_special_tokens=False)['input_ids']
        else:
            TargetEl_idxs = self.tokenizer.encode_plus(str(self.dataset.ent2short_text[hrt_list[0]]).lower(), add_special_tokens=False)['input_ids']
            TargetEr_idxs = self.tokenizer.encode_plus(str(self.dataset.ent2short_text[hrt_list[2]]).lower(), add_special_tokens=False)['input_ids']

        l_mask_idxs = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token) for _ in range(len(TargetEl_idxs))]
        r_mask_idxs = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token) for _ in range(len(TargetEr_idxs))]
        rel_mask_idxs = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token) for _ in range(len(R_idxs))]

        #### predict r entity
        El_max_length = self.max_seq_length - len(R_idxs) - len(TargetEr_idxs) - 2
        if len(El_idxs) >= El_max_length:
            El_idxs_filted = El_idxs[:El_max_length]
        else:
            El_idxs_filted = El_idxs

        r_inputs_ids = [self.tokenizer.cls_token_id] + El_idxs_filted + R_idxs + r_mask_idxs + [self.tokenizer.sep_token_id]
        r_padding_length = max(0, self.max_seq_length - len(r_inputs_ids))
        r_inputs_ids = r_inputs_ids + ([self.tokenizer.pad_token_id] * r_padding_length)
        r_inputs_ids = r_inputs_ids[:self.max_seq_length]

        r_inputs_ids, r_mlm_labels = self.mlm_process(r_inputs_ids, mlm_range_list=[[1, 1 + len(El_idxs_filted)]])

        r_emlm_labels = np.zeros(self.max_seq_length, dtype=np.int64) - 100
        r_emlm_start_idx = 1 + len(El_idxs_filted) + len(R_idxs)
        r_emlm_end_idx = min(r_emlm_start_idx+len(TargetEr_idxs), self.max_seq_length)
        for idx in range(r_emlm_start_idx, r_emlm_end_idx):
            r_emlm_labels[idx] = TargetEr_idxs[int(idx-r_emlm_start_idx)]

        #### predict l entity
        Er_max_length = self.max_seq_length - len(R_idxs) - len(TargetEl_idxs) - 2
        if len(Er_idxs) >= Er_max_length:
            Er_idxs_filted = Er_idxs[:Er_max_length]
        else:
            Er_idxs_filted = Er_idxs

        l_inputs_ids = [self.tokenizer.cls_token_id] + l_mask_idxs + R_idxs + Er_idxs_filted + [self.tokenizer.sep_token_id]
        l_padding_length = max(0, self.max_seq_length - len(l_inputs_ids))
        l_inputs_ids = l_inputs_ids + ([self.tokenizer.pad_token_id] * l_padding_length)
        l_inputs_ids = l_inputs_ids[:self.max_seq_length]

        l_inputs_ids, l_mlm_labels = self.mlm_process(
            l_inputs_ids, mlm_range_list=[[
                1 + len(l_mask_idxs) + len(R_idxs),
                min(self.max_seq_length-1, 1 + len(l_mask_idxs) + len(R_idxs) + len(Er_idxs_filted))
            ]])

        l_emlm_labels = np.zeros(self.max_seq_length, dtype=np.int64) - 100
        l_emlm_start_idx = 1
        l_emlm_end_idx = 1 + len(TargetEl_idxs)
        for idx in range(l_emlm_start_idx, l_emlm_end_idx):
            l_emlm_labels[idx] = TargetEl_idxs[int(idx - l_emlm_start_idx)]

        #### predict relation
        entities_max_length = self.max_seq_length - len(R_idxs) - 2
        if len(El_idxs) + len(Er_idxs) > entities_max_length:
            if len(El_idxs) < entities_max_length // 2:
                rel_El_idxs = El_idxs
                rel_Er_idxs = Er_idxs[:entities_max_length-len(El_idxs)]
            elif len(Er_idxs) < entities_max_length // 2:
                rel_El_idxs = El_idxs[:entities_max_length-len(Er_idxs)]
                rel_Er_idxs = Er_idxs
            else:
                rel_El_idxs = El_idxs[:entities_max_length // 2]
                rel_Er_idxs = Er_idxs[:entities_max_length - entities_max_length // 2]
        else:
            rel_El_idxs = El_idxs
            rel_Er_idxs = Er_idxs
        rel_inputs_ids = [self.tokenizer.cls_token_id] + rel_El_idxs + rel_mask_idxs + rel_Er_idxs + [self.tokenizer.sep_token_id]
        rel_padding_length = max(0, self.max_seq_length - len(rel_inputs_ids))
        rel_inputs_ids = rel_inputs_ids + ([self.tokenizer.pad_token_id] * rel_padding_length)
        rel_inputs_ids = rel_inputs_ids[:self.max_seq_length]

        rel_inputs_ids, rel_mlm_labels = self.mlm_process(
            rel_inputs_ids, mlm_range_list=[
                [1, 1 + len(rel_El_idxs)],
                [1 + len(rel_El_idxs) + len(R_idxs), min(self.max_seq_length-1, 1 + len(rel_El_idxs) + len(R_idxs) + len(rel_Er_idxs))]
            ])

        rel_emlm_labels = np.zeros(self.max_seq_length, dtype=np.int64) - 100
        rel_emlm_start_idx = 1 + len(rel_El_idxs)
        rel_emlm_end_idx = 1 + len(rel_El_idxs) + len(R_idxs)
        for idx in range(rel_emlm_start_idx, rel_emlm_end_idx):
            rel_emlm_labels[idx] = R_idxs[int(idx - rel_emlm_start_idx)]

        #### sample
        threshold_list = [0.4, 0.8, 1.]
        random_state = random.random()
        if random_state < threshold_list[0]:
            inputs_ids = l_inputs_ids
            mlm_labels = l_mlm_labels
            emlm_labels = l_emlm_labels
        elif random_state < threshold_list[1]:
            inputs_ids = r_inputs_ids
            mlm_labels = r_mlm_labels
            emlm_labels = r_emlm_labels
        else:
            inputs_ids = rel_inputs_ids
            mlm_labels = rel_mlm_labels
            emlm_labels = rel_emlm_labels

        # return
        return {
            'src_entities': hrt_list[0],
            'rels': hrt_list[1],
            'tgt_entities': hrt_list[2],

            'inputs_ids': torch.tensor(inputs_ids, dtype=torch.long),
            'mlm_labels': torch.tensor(mlm_labels, dtype=torch.long),
            'emlm_labels': torch.tensor(emlm_labels, dtype=torch.long),
        }

    def mlm_process(self, input_ids, mlm_range_list):
        labels = np.zeros(self.max_seq_length, dtype=np.int64) - 100

        for mlm_range in mlm_range_list:
            for idx in range(mlm_range[0], mlm_range[1]):
                if idx >= self.max_seq_length:
                    break
                rand_num = random.random()
                if rand_num < self.mlm_prob:
                    if input_ids[idx] < self.vocab_size:
                        labels[idx] = input_ids[idx]
                    else:
                        labels[idx] = 0
                    if random.random() < 0.8:
                        input_ids[idx] = 0
                    elif random.random() < 0.5:
                        try:
                            input_ids[idx] = input_ids[idx]
                        except:
                            input_ids[idx] = 0
                    else:
                        while True:
                            random_token = random.sample([_id for _id in range(1, self.vocab_size)], 1)[0]
                            if random_token != input_ids[idx]:
                                break
                        try:
                            input_ids[idx] = random_token
                        except:
                            input_ids[idx] = 0
                else:
                    try:
                        input_ids[idx] = input_ids[idx]
                    except:
                        input_ids[idx] = 0
        return input_ids, labels

class LinkPredictionPairDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, df, tokenizer, max_seq_length):
        self.dataset = dataset
        self.df = df

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # fn
        self.text2tokens = text2tokens
        self.tokens2bert_ids = tokens2bert_ids

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        hrt_list = self.df['entity1'][item], self.df['relation'][item], self.df['entity2'][item]

        head_ids = self.text2tokens(self.dataset.ent2text[hrt_list[0]], self.tokenizer)
        rel_ids = self.text2tokens(self.dataset.rel2text[hrt_list[1]], self.tokenizer)
        tail_ids = self.text2tokens(self.dataset.ent2text[hrt_list[2]], self.tokenizer)
        head_ids, rel_ids, tail_ids = head_ids[1:-1], rel_ids[1:-1], tail_ids[1:-1]
        # print('0hrt_list: ', self.dataset.ent2text[hrt_list[0]])
        # print('1hrt_list: ', self.dataset.rel2text[hrt_list[1]])
        # print('2hrt_list: ', self.dataset.ent2text[hrt_list[2]])
        # print('head_ids: ', head_ids)
        # print('rel_ids: ', rel_ids)
        # print('tail_ids: ', tail_ids)

        src_input_ids, src_mask_ids, src_segment_ids = self.tokens2bert_ids(head_ids, rel_ids, self.tokenizer, self.max_seq_length, 'src')
        tgt_input_ids, tgt_mask_ids, tgt_segment_ids = self.tokens2bert_ids(tail_ids, rel_ids, self.tokenizer, self.max_seq_length, 'tgt')

        # return
        return {
            'src_entities': hrt_list[0],
            'rels': hrt_list[1],
            'tgt_entities': hrt_list[2],
            'src_input_ids': torch.tensor(src_input_ids, dtype=torch.long),
            'src_attention_mask': torch.tensor(src_mask_ids, dtype=torch.long),
            'src_token_type_ids': torch.tensor(src_segment_ids, dtype=torch.long),
            'tgt_input_ids': torch.tensor(tgt_input_ids, dtype=torch.long),
            'tgt_attention_mask': torch.tensor(tgt_mask_ids, dtype=torch.long),
            'tgt_token_type_ids': torch.tensor(tgt_segment_ids, dtype=torch.long),
            # 'labels': torch.tensor(self.df['label'][item], dtype=torch.long),
        }