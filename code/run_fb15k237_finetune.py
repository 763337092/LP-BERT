import os
import random
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import BertModel, BertPreTrainedModel, BertConfig, BertTokenizer, AdamW, get_linear_schedule_with_warmup,\
    RobertaConfig, RobertaTokenizer

from util_fn import save_pickle, load_pickle, EarlyStopping, AverageMeter
from data_fn import LoadDataset, LinkPredictionPairDataset
from models_fn import LPBERT
from train_fn import emb_infer_fn, infer_fn, finetuning_train_fn, finetuning_eval_fn
from eval_fn import get_triplet_candidate, safe_ranking, get_pred_topk
from loss_fn import FocalBCELoss, FocalBCEWithLogitsLoss

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

DEBUG = False

DATA_PATH = '../data'
DATASET = 'FB15k-237'
ROOT_PRETRAINED_PATH = '../data'
PRETRAINED_MODEL_PATH = f'{ROOT_PRETRAINED_PATH}/spanbert-base-cased/'
# PRETRAINED_MODEL_PATH = f'{ROOT_PRETRAINED_PATH}/bert-base-chinese/'
# PRETRAINED_MODEL_PATH2 = f'{DATA_PATH}/{DATASET}/FB15k-237_pretrain_3tasks1211/'
PRETRAINED_MODEL_PATH2 = f'{DATA_PATH}/{DATASET}/FB15k-237_pretrain/'
CACHE_PATH = f'{DATA_PATH}/{DATASET}/FB15k-237_finetune/'
if not os.path.exists(CACHE_PATH):
    os.mkdir(CACHE_PATH)
if DEBUG:
    CACHE_PATH = f'{CACHE_PATH}/debug'
    PRETRAINED_MODEL_PATH2 = f'{PRETRAINED_MODEL_PATH2}/debug'
    if not os.path.exists(CACHE_PATH):
        os.mkdir(CACHE_PATH)

NUM_WORKERS = 20
MAX_SEQ_LENGTH = 100
GPU_NUM = 1
BATCH_SIZE = 120 * GPU_NUM
ACCUMULATION_STEPS = 1
WEIGHT_DECAY = 0.01
LR = 1e-3
BERT_LR = 3e-5
ADAM_EPSILON = 1e-6
WARMUP = 0.05
EPOCHS = 7
EARLYSTOP_NUM = 1
RECALL_NUM = 1000
# NEG_SAMPLE_RATE = 10
if DEBUG:
    EPOCHS = 1

tokenizer = BertTokenizer.from_pretrained(f'{PRETRAINED_MODEL_PATH}/vocab.txt', do_lower_case=True)
# tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name_or_path=PRETRAINED_MODEL_PATH,
#                                              vocab_file=f'{PRETRAINED_MODEL_PATH}/vocab.json',
#                                              merges_file=f'{PRETRAINED_MODEL_PATH}/merges.txt')

dataset = LoadDataset(dataset=DATASET, cache_path=CACHE_PATH, neg_times=5, sample_seed=2021, load_cache=False, debug=DEBUG)

torch.cuda.empty_cache()
device = torch.device("cuda")
bert_config = BertConfig.from_pretrained(f'{PRETRAINED_MODEL_PATH}/config.json', num_labels=2)
bert_config.output_hidden_states = True
model = LPBERT(pretrained_path=PRETRAINED_MODEL_PATH, config=bert_config, dataset_name=DATASET)
model.to(device)
model = nn.DataParallel(model)

pretrained_dict = torch.load(f'{PRETRAINED_MODEL_PATH2}/pytorch_model.bin')
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

def loss_fn(y_pred, y_true, weights):
    # loss_fct = nn.CrossEntropyLoss()
    loss_fct = FocalBCELoss(alpha=0.5)
    return loss_fct(y_pred, y_true)

# no_decay = ['bias', 'LayerNorm.weight']
# optimizer_grouped_parameters = [
#     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': WEIGHT_DECAY},
#     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
# ]

MODEL_WEIGHT = f"{CACHE_PATH}/model.bin"
#"""
no_decay = ['bias', 'LayerNorm.weight']

bert_param_optimizer = list(model.module.bert.named_parameters())
linear_param_optimizer = list(model.module.proj.named_parameters())# + list(model.module.linear11.named_parameters()) + list(model.module.linear12.named_parameters())+ list(model.module.linear21.named_parameters()) + list(model.module.linear22.named_parameters())

optimizer_grouped_parameters = [
    {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': WEIGHT_DECAY, 'lr': BERT_LR},
    {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': BERT_LR},

    {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': WEIGHT_DECAY, 'lr': LR},
    {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': LR}
]

adam_betas = (0.9, 0.98)

optimizer = AdamW(optimizer_grouped_parameters, lr=LR,
                  betas=adam_betas, eps=ADAM_EPSILON)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP, num_training_steps=EPOCHS*len(dataset.train)//BATCH_SIZE)

es = EarlyStopping(patience=EARLYSTOP_NUM, mode="min")
#"""
for epoch in range(EPOCHS):
    train_set = LinkPredictionPairDataset(dataset=dataset, df=dataset.train, tokenizer=tokenizer,
                                          max_seq_length=MAX_SEQ_LENGTH)
    dev_set = LinkPredictionPairDataset(dataset=dataset, df=dataset.dev, tokenizer=tokenizer,
                                        max_seq_length=MAX_SEQ_LENGTH)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    dev_loader = DataLoader(dev_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    # for batch in train_loader:
    #     print(batch)
    #     break

    finetuning_train_fn(dataset, train_loader, model, optimizer, loss_fn, device, scheduler=scheduler)
    eval_loss, eval_pred = finetuning_eval_fn(dataset, dev_loader, model, loss_fn, device)
    # eval_auc = roc_auc_score(valid_data.label.values, eval_pred[:, 1])
    print(f'Epoch {epoch+1}/{EPOCHS} eval_loss={eval_loss:.4f}')

    es(eval_loss, model, model_path=MODEL_WEIGHT)
    if es.early_stop:
        print("Early stopping")
        break
#"""

model.load_state_dict(torch.load(MODEL_WEIGHT))

# all_entity_df = pd.DataFrame(columns=['entity1', 'relation', 'entity2'])
_entity1_list, _relation_list, _entity2_list = [], [], []
for ent in tqdm(dataset.ent_list, desc='Making all entity df'):
    for rel in dataset.rel_list:
        _entity1_list.append(ent)
        _relation_list.append(rel)
        _entity2_list.append(ent)
        # tmp = {
        #     'entity1': ent,
        #     'relation': rel,
        #     'entity2': ent,
        # }
        # all_entity_df = all_entity_df.append(tmp, ignore_index=True)
all_entity_df = pd.DataFrame()
all_entity_df['entity1'] = _entity1_list
all_entity_df['relation'] = _relation_list
all_entity_df['entity2'] = _entity2_list
# all_entity_df['label'] = 0
ent_set = LinkPredictionPairDataset(dataset=dataset, df=all_entity_df, tokenizer=tokenizer, max_seq_length=MAX_SEQ_LENGTH)
ent_loader = DataLoader(ent_set, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS)
print('ent_set: ', len(ent_set))
print('ent_loader: ', len(ent_loader))
ent_rel_emb, ent_emb = emb_infer_fn(ent_loader, model, device)
# torch.save(ent_rel_emb, f'{CACHE_PATH}/ent_rel_emb.pt')
# torch.save(ent_emb, f'{CACHE_PATH}/ent_emb.pt')

ent_rel2emb, ent2emb = collections.defaultdict(dict), dict()
for idx, (ent1, rel, ent2) in enumerate(all_entity_df[['entity1', 'relation', 'entity2']].values):
    ent_rel2emb[ent1][rel] = ent_rel_emb[idx]
    if ent2 not in ent2emb:
        ent2emb[ent2] = ent_emb[idx]
print('ent_rel2emb: ', len(ent_rel2emb))
print('ent2emb: ', len(ent2emb))
# save_pickle(ent_rel2emb, f'{CACHE_PATH}/ent_rel2emb.pkl')
# save_pickle(ent2emb, f'{CACHE_PATH}/ent2emb.pkl')

##### Get Test Candidates
test_score_dict, test_candicate_dict = infer_fn(model, device, BATCH_SIZE*2, dataset.test,
                                           dataset.subj2objs, dataset.ent_list, ent_rel2emb, ent2emb, topk=RECALL_NUM)
print(f'Test Score: '
      f'Hits@1={test_score_dict["hits@1"]:.4f} Hits@3={test_score_dict["hits@3"]:.4f} Hits@10={test_score_dict["hits@10"]:.4f} '
      f'Hits@30={test_score_dict["hits@30"]:.4f} Hits@50={test_score_dict["hits@50"]:.4f} Hits@100={test_score_dict["hits@100"]:.4f} '
      f'Hits@300={test_score_dict["hits@300"]:.4f} Hits@500={test_score_dict["hits@500"]:.4f} Hits@1000={test_score_dict["hits@1000"]:.4f} '
      f'MR={test_score_dict["MR"]:.4f} MRR={test_score_dict["MRR"]:.4f} Q95R={test_score_dict["Q95R"]:.4f}')
save_pickle(test_candicate_dict, f'{CACHE_PATH}/test_candicate_dict.pkl')

# ##### Get Train Candidates
# train_score_dict, train_candicate_dict = infer_fn(model, device, BATCH_SIZE*2, dataset.train,
#                                            dataset.subj2objs, dataset.ent_list, ent_rel2emb, ent2emb, topk=RECALL_NUM)
# print(f'train Score: '
#       f'Hits@1={train_score_dict["hits@1"]:.4f} Hits@3={train_score_dict["hits@3"]:.4f} Hits@10={train_score_dict["hits@10"]:.4f} '
#       f'Hits@30={train_score_dict["hits@30"]:.4f} Hits@50={train_score_dict["hits@50"]:.4f} Hits@100={train_score_dict["hits@100"]:.4f} '
#       f'Hits@300={train_score_dict["hits@300"]:.4f} Hits@500={train_score_dict["hits@500"]:.4f} Hits@1000={train_score_dict["hits@1000"]:.4f} '
#       f'MR={train_score_dict["MR"]:.4f} MRR={train_score_dict["MRR"]:.4f} Q95R={train_score_dict["Q95R"]:.4f}')
# save_pickle(train_candicate_dict, f'{CACHE_PATH}/train_candicate_dict.pkl')
#
# ##### Get Valid Candidates
# valid_score_dict, valid_candicate_dict = infer_fn(model, device, BATCH_SIZE*2, dataset.dev,
#                                            dataset.subj2objs, dataset.ent_list, ent_rel2emb, ent2emb, topk=RECALL_NUM)
# print(f'Valid Score: '
#       f'Hits@1={valid_score_dict["hits@1"]:.4f} Hits@3={valid_score_dict["hits@3"]:.4f} Hits@10={valid_score_dict["hits@10"]:.4f} '
#       f'Hits@30={valid_score_dict["hits@30"]:.4f} Hits@50={valid_score_dict["hits@50"]:.4f} Hits@100={valid_score_dict["hits@100"]:.4f} '
#       f'Hits@300={valid_score_dict["hits@300"]:.4f} Hits@500={valid_score_dict["hits@500"]:.4f} Hits@1000={valid_score_dict["hits@1000"]:.4f} '
#       f'MR={valid_score_dict["MR"]:.4f} MRR={valid_score_dict["MRR"]:.4f} Q95R={valid_score_dict["Q95R"]:.4f}')
# save_pickle(valid_candicate_dict, f'{CACHE_PATH}/valid_candicate_dict.pkl')
