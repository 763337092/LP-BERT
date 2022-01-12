import os
import math
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
# from transformers.modeling_bert import BertLayerNorm

from util_fn import save_pickle, load_pickle, EarlyStopping, AverageMeter
from data_fn import LoadDataset, LinkPredictionPretrainDataset
from models_fn import MultiTaskBert
from train_fn import pretraining_loss_fn, pretraining_train_fn, pretraining_eval_fn
from eval_fn import get_triplet_candidate, safe_ranking, get_pred_topk

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

DEBUG = False

DATA_PATH = '../data'
DATASET = 'FB15k-237'
ROOT_PRETRAINED_PATH = '../data'
PRETRAINED_MODEL_PATH = f'{ROOT_PRETRAINED_PATH}/spanbert-base-cased/'
# PRETRAINED_MODEL_PATH = f'{ROOT_PRETRAINED_PATH}/bert-base-chinese/'
CACHE_PATH = f'{DATA_PATH}/{DATASET}/FB15k-237_pretrain/'
if not os.path.exists(CACHE_PATH):
    os.mkdir(CACHE_PATH)
if DEBUG:
    CACHE_PATH = f'{CACHE_PATH}/debug'
    if not os.path.exists(CACHE_PATH):
        os.mkdir(CACHE_PATH)

NUM_WORKERS = 20
MAX_SEQ_LENGTH = 152# 128
# MAX_TARGET_SEQ_LENGTH = 24
GPU_NUM = 3
BATCH_SIZE = 32 * GPU_NUM
ACCUMULATION_STEPS = 1
WEIGHT_DECAY = 0.01
LR = 1e-4
BERT_LR = 5e-5
ADAM_EPSILON = 1e-6
WARMUP = 0.05
EPOCHS = 50
EARLYSTOP_NUM = 3
SCHEDULE_DECAY = 1
MLM_PROB = 0.15
# NEG_SAMPLE_RATE = 10
if DEBUG:
    EPOCHS = 1

tokenizer = BertTokenizer.from_pretrained(f'{PRETRAINED_MODEL_PATH}/vocab.txt', do_lower_case=True)

dataset = LoadDataset(dataset=DATASET, cache_path=CACHE_PATH, neg_times=5, sample_seed=2021, load_cache=False, debug=DEBUG)

torch.cuda.empty_cache()
device = torch.device("cuda:0")
bert_config = BertConfig.from_pretrained(f'{PRETRAINED_MODEL_PATH}/config.json', num_labels=2)
bert_config.output_hidden_states = True
vocab_size = bert_config.vocab_size
model = MultiTaskBert(pretrained_path=PRETRAINED_MODEL_PATH, config=bert_config)
model.to(device)
model = nn.DataParallel(model)

train_set = LinkPredictionPretrainDataset(dataset, df=dataset.train, tokenizer=tokenizer, max_seq_length=MAX_SEQ_LENGTH, mlm_prob=MLM_PROB, vocab_size=vocab_size)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
valid_set = LinkPredictionPretrainDataset(dataset, df=dataset.dev, tokenizer=tokenizer, max_seq_length=MAX_SEQ_LENGTH, mlm_prob=MLM_PROB, vocab_size=vocab_size)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
# for batch in train_loader:
#     print(batch)
#     break

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=SCHEDULE_DECAY-1, verbose=True)
best_val_loss = 10000.
es = EarlyStopping(patience=EARLYSTOP_NUM, mode="min")
MODEL_WEIGHT = f"{CACHE_PATH}/pytorch_model.bin"
for epoch in range(EPOCHS):
    pretraining_train_fn(train_loader, model, optimizer, pretraining_loss_fn, device, scheduler=None)
    valid_loss, valid_mlm_loss, valid_emlm_loss = pretraining_eval_fn(valid_loader, model, pretraining_loss_fn, device)
    print(f'Epoch {epoch + 1}/{EPOCHS} valid_loss={valid_loss:.4f} valid_mlm_loss={valid_mlm_loss:.4f} valid_emlm_loss={valid_emlm_loss:.4f}')
    if valid_loss <= best_val_loss:
        best_val_loss = valid_loss
    scheduler.step(valid_loss)
    es(valid_loss, model, model_path=MODEL_WEIGHT)
    if es.early_stop:
        print("Early stopping")
        break
