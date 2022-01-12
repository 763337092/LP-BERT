import os
import random
import logging
import collections
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, SequentialSampler

def get_triplet_candidate(_scores, topk=50):
    pos_score = _scores[0]  # []
    same_score_loc = np.where(_scores == pos_score)[0]
    assert same_score_loc.size > 0 and same_score_loc[0] == 0
    rdm_pos_loc = same_score_loc[random.randint(0, same_score_loc.shape[0] - 1)]
    _sort_idxs = np.argsort(-_scores)
    _rank = np.where(_sort_idxs == rdm_pos_loc)[0][0] + 1
    _sort_idxs = list(_sort_idxs)
    if 0 in _sort_idxs[:topk]:
        _sort_idxs.remove(0)
    return _sort_idxs[:topk]

def get_pred_topk(_scores, topk=50):
    _sort_idxs = np.argsort(-_scores)
    return _sort_idxs[:topk]

def safe_ranking(_scores, verbose=False):
    pos_score = _scores[0]  # []
    same_score_loc = np.where(_scores == pos_score)[0]
    assert same_score_loc.size > 0 and same_score_loc[0] == 0
    rdm_pos_loc = same_score_loc[random.randint(0, same_score_loc.shape[0]-1)]
    _sort_idxs = np.argsort(-_scores)
    _rank = np.where(_sort_idxs == rdm_pos_loc)[0][0] + 1

    if verbose:
        _default_rank = np.where(_sort_idxs == 0)[0][0] + 1
        print("From safe_ranking: default rank is {}, after safe {}".format(_default_rank, _rank))
    return _rank
