import time
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from util_fn import AverageMeter
from eval_fn import get_triplet_candidate, safe_ranking, get_pred_topk

def pretraining_loss_fn(mlm_output, emlm_output, mlm_label, emlm_label):
    # loss_fct = nn.MSELoss()
    # loss_fct = nn.BCEWithLogitsLoss()
    # loss_fct = nn.BCELoss()
    # loss_fct1 = FocalBCELoss(gamma=2, alpha=0.99)
    loss_fct1 = nn.BCELoss()
    loss_fct2 = nn.CrossEntropyLoss()
    loss_fct3 = nn.MSELoss()

    mlm_loss = loss_fct2(mlm_output[mlm_label.ne(-100)], mlm_label[mlm_label.ne(-100)])
    emlm_loss = loss_fct2(emlm_output[emlm_label.ne(-100.)], emlm_label[emlm_label.ne(-100.)])
    return mlm_loss, emlm_loss

def emb_infer_fn(data_loader, model, device):
    ent_rel_emb_list, ent_emb_list = [], []
    emb_size = None
    model.eval()
    tk0 = tqdm(data_loader, total=len(data_loader), desc="Emb Infer")
    with torch.no_grad():
        for bi, d in enumerate(tk0):
            torch.cuda.empty_cache()
            src_input_ids = d["src_input_ids"].to(device, dtype=torch.long)
            src_attention_mask = d["src_attention_mask"].to(device, dtype=torch.long)
            src_token_type_ids = d["src_token_type_ids"].to(device, dtype=torch.long)
            tgt_input_ids = d["tgt_input_ids"].to(device, dtype=torch.long)
            tgt_attention_mask = d["tgt_attention_mask"].to(device, dtype=torch.long)
            tgt_token_type_ids = d["tgt_token_type_ids"].to(device, dtype=torch.long)

            ent_rel_emb = model.module.encoder(
                src_input_ids,
                attention_mask=src_attention_mask,
                token_type_ids=src_token_type_ids
            ).detach().cpu()
            ent_emb = model.module.encoder(
                tgt_input_ids,
                attention_mask=tgt_attention_mask,
                token_type_ids=tgt_token_type_ids
            ).detach().cpu()

            ent_rel_emb_list.append(ent_rel_emb)
            ent_emb_list.append(ent_emb)
            if emb_size is None:
                emb_size = ent_rel_emb.shape[1]
    # return np.concatenate(ent_rel_emb_list).reshape(-1, emb_size), np.concatenate(ent_emb_list).reshape(-1, emb_size)
    return torch.cat(ent_rel_emb_list, dim=0).contiguous(), torch.cat(ent_emb_list, dim=0).contiguous()

def infer_fn(model, device, batch_size,
             dev, subj2objs, ent_list, ent_rel2emb, ent2emb, topk):

    ranks_left, ranks_right, ranks = [], [], []
    hits_left, hits_right, hits = [], [], []
    top_ten_hit_count = 0
    for i in range(1000):
        hits_left.append([])
        hits_right.append([])
        hits.append([])

    new_dev_dict = collections.defaultdict(dict)
    pred_head_list, pred_rel_list, pred_tail_list, pred_topk_list = [], [], [], []
    for _idx_ex, (_head, _rel, _tail) in enumerate(tqdm(dev[['entity1', 'relation', 'entity2']].values, desc="get candidates")):
        torch.cuda.empty_cache()
        head_ent_list = []
        tail_ent_list = []

        # tail corrupt
        _pos_tail_ents = subj2objs[_head][_rel]
        _neg_tail_ents = set(ent_list) - _pos_tail_ents
        head_ent_list.extend([_head] * (1 + len(_neg_tail_ents)))
        tail_ent_list.append(_tail)  # positive example
        tail_ent_list.extend(_neg_tail_ents)  # negative examples

        triplet_list = list([_h, _rel, _t] for _h, _t in zip(head_ent_list, tail_ent_list))

        # build dataset
        rep_src_list = [ent_rel2emb[_h][_rel] for _h, _rel, _ in triplet_list]
        rep_tgt_list = [ent2emb[_t] for _, _, _t in triplet_list]
        all_rep_src = torch.stack(rep_src_list, dim=0).to(device)
        all_rep_tgt = torch.stack(rep_tgt_list, dim=0).to(device)

        local_scores_list = []
        sim_batch_size = batch_size
        for _idx_r in range(0, all_rep_src.shape[0], sim_batch_size):
            _rep_src, _rep_tgt = all_rep_src[_idx_r: _idx_r + sim_batch_size], all_rep_tgt[
                                                                               _idx_r: _idx_r + sim_batch_size]
            with torch.no_grad():
                logits = model.module.classifier(_rep_src, _rep_tgt)
                # logits = torch.softmax(logits, dim=-1)
                local_scores = logits.detach().cpu().numpy()# [:, 1]
            local_scores_list.append(local_scores.reshape(-1))
        scores = np.concatenate(local_scores_list, axis=0)

        right_rank = safe_ranking(scores)
        tails_corrupt_idx = get_pred_topk(scores, topk)
        tails_corrupt = [tail_ent_list[i] for i in tails_corrupt_idx]

        ranks.append(right_rank)

        new_dev_dict[(_head, _rel, _tail)]["tails_corrupt"] = tails_corrupt

        # pred_head_list.append(_head)
        # pred_rel_list.append(_rel)
        # pred_tail_list.append(_tail)
        # pred_topk_list.append('\3'.join(tails_corrupt))

        top_ten_hit_count += int(right_rank <= 10)
        # hits
        for hits_level in range(1000):
            if right_rank <= hits_level + 1:
                hits[hits_level].append(1.0)
                hits_right[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)
                hits_right[hits_level].append(0.0)
    score_dict = {
        'hits@1': np.mean(hits[0]),
        'hits@3': np.mean(hits[2]),
        'hits@10': np.mean(hits[9]),
        'hits@30': np.mean(hits[29]),
        'hits@50': np.mean(hits[49]),
        'hits@100': np.mean(hits[99]),
        'hits@300': np.mean(hits[299]),
        'hits@500': np.mean(hits[499]),
        'hits@1000': np.mean(hits[999]),
        'MR': np.mean(ranks),
        'MRR': np.mean(1. / np.array(ranks)),
        'Q95R': np.quantile(ranks, q=0.95),
    }

    # pred_df = pd.DataFrame()
    # pred_df['head'] = pred_head_list
    # pred_df['relation'] = pred_rel_list
    # pred_df['tail'] = pred_tail_list
    # pred_df['pred'] = pred_topk_list

    return score_dict, new_dev_dict#, pred_df

def wn18rr_infer_fn(model, device, batch_size,
             dev, subj2objs, obj2subjs, ent_list, ent_rel2emb, ent2emb, topk):

    ranks_left, ranks_right, ranks = [], [], []
    hits_left, hits_right, hits = [], [], []
    top_ten_hit_count = 0
    for i in range(1000):
        hits_left.append([])
        hits_right.append([])
        hits.append([])

    new_dev_dict = collections.defaultdict(dict)
    pred_head_list, pred_rel_list, pred_tail_list, pred_topk_list = [], [], [], []
    for _idx_ex, (_head, _rel, _tail) in enumerate(tqdm(dev[['entity1', 'relation', 'entity2']].values, desc="get candidates")):
        torch.cuda.empty_cache()
        head_ent_list = []
        tail_ent_list = []

        # head corrupt
        _pos_head_ents = obj2subjs[_tail][_rel]
        _neg_head_ents = set(ent_list) - _pos_head_ents
        head_ent_list.append(_head)  # positive example
        head_ent_list.extend(_neg_head_ents)  # negative examples
        tail_ent_list.extend([_tail] * (1 + len(_neg_head_ents)))
        split_idx = len(head_ent_list)

        # tail corrupt
        _pos_tail_ents = subj2objs[_head][_rel]
        _neg_tail_ents = set(ent_list) - _pos_tail_ents
        head_ent_list.extend([_head] * (1 + len(_neg_tail_ents)))
        tail_ent_list.append(_tail)  # positive example
        tail_ent_list.extend(_neg_tail_ents)  # negative examples

        triplet_list = list([_h, _rel, _t] for _h, _t in zip(head_ent_list, tail_ent_list))

        # build dataset
        rep_src_list = [ent_rel2emb[_h][_rel] for _h, _rel, _ in triplet_list]
        rep_tgt_list = [ent2emb[_t] for _, _, _t in triplet_list]
        all_rep_src = torch.stack(rep_src_list, dim=0).to(device)
        all_rep_tgt = torch.stack(rep_tgt_list, dim=0).to(device)

        local_scores_list = []
        sim_batch_size = batch_size
        for _idx_r in range(0, all_rep_src.shape[0], sim_batch_size):
            _rep_src, _rep_tgt = all_rep_src[_idx_r: _idx_r + sim_batch_size], all_rep_tgt[
                                                                               _idx_r: _idx_r + sim_batch_size]
            with torch.no_grad():
                logits = model.module.classifier(_rep_src, _rep_tgt)
                # logits = torch.softmax(logits, dim=-1)
                local_scores = logits.detach().cpu().numpy()# [:, 1]
            local_scores_list.append(local_scores)
        scores = np.concatenate(local_scores_list, axis=0)

        left_scores = scores[:split_idx]
        left_rank = safe_ranking(left_scores)
        heads_corrupt_idx = get_pred_topk(left_scores, topk)
        heads_corrupt = [head_ent_list[i] for i in heads_corrupt_idx]

        right_scores = scores[split_idx:]
        right_rank = safe_ranking(right_scores)
        tails_corrupt_idx = get_pred_topk(right_scores, topk)
        tails_corrupt = [tail_ent_list[i + split_idx] for i in tails_corrupt_idx]

        ranks_left.append(left_rank)
        ranks_right.append(right_rank)
        ranks.extend([left_rank, right_rank])

        new_dev_dict[(_head, _rel, _tail)]["heads_corrupt"] = heads_corrupt
        new_dev_dict[(_head, _rel, _tail)]["tails_corrupt"] = tails_corrupt

        # pred_head_list.append(_head)
        # pred_rel_list.append(_rel)
        # pred_tail_list.append(_tail)
        # pred_topk_list.append('\3'.join(tails_corrupt))

        top_ten_hit_count += (int(left_rank <= 10) + int(right_rank <= 10))
        # hits
        for hits_level in range(1000):
            if left_rank <= hits_level + 1:

                hits[hits_level].append(1.0)
                hits_left[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)
                hits_left[hits_level].append(0.0)

            if right_rank <= hits_level + 1:
                hits[hits_level].append(1.0)
                hits_right[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)
                hits_right[hits_level].append(0.0)
    score_dict = {
        'hits@1': np.mean(hits[0]),
        'hits@3': np.mean(hits[2]),
        'hits@10': np.mean(hits[9]),
        'hits@30': np.mean(hits[29]),
        'hits@50': np.mean(hits[49]),
        'hits@100': np.mean(hits[99]),
        'hits@300': np.mean(hits[299]),
        'hits@500': np.mean(hits[499]),
        'hits@1000': np.mean(hits[999]),
        'MR': np.mean(ranks),
        'MRR': np.mean(1. / np.array(ranks)),
        'Q95R': np.quantile(ranks, q=0.95),
    }

    # pred_df = pd.DataFrame()
    # pred_df['head'] = pred_head_list
    # pred_df['relation'] = pred_rel_list
    # pred_df['tail'] = pred_tail_list
    # pred_df['pred'] = pred_topk_list

    return score_dict, new_dev_dict#, pred_df

def pretraining_train_fn(data_loader, model, optimizer, loss_fn, device, scheduler=None):
    model.train()
    losses = AverageMeter()
    mlm_losses = AverageMeter()
    emlm_losses = AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader), desc="Train")
    for bi, d in enumerate(tk0):
        torch.cuda.empty_cache()
        inputs_ids = d["inputs_ids"].to(device, dtype=torch.long)
        mlm_label = d["mlm_labels"].to(device, dtype=torch.long)
        emlm_label = d["emlm_labels"].to(device, dtype=torch.long)

        model.zero_grad()
        mlm_outputs, emlm_outputs = model(inputs_ids)
        # print('outputs: ', outputs.shape)
        # print('labels: ', outputs.shape)
        mlm_loss, emlm_loss = loss_fn(mlm_outputs, emlm_outputs, mlm_label, emlm_label)# * 100
        loss = mlm_loss + emlm_loss
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        torch.cuda.empty_cache()

        losses.update(loss.item(), inputs_ids.size(0))
        mlm_losses.update(mlm_loss.item(), inputs_ids.size(0))
        emlm_losses.update(emlm_loss.item(), inputs_ids.size(0))
        tk0.set_postfix(loss=losses.avg, mlm_loss=mlm_losses.avg, emlm_loss=emlm_losses.avg)

def pretraining_eval_fn(data_loader, model, loss_fn, device):
    model.train()
    losses = AverageMeter()
    mlm_losses = AverageMeter()
    emlm_losses = AverageMeter()

    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader), desc="Eval")
        for bi, d in enumerate(tk0):
            torch.cuda.empty_cache()
            inputs_ids = d["inputs_ids"].to(device, dtype=torch.long)
            mlm_label = d["mlm_labels"].to(device, dtype=torch.long)
            emlm_label = d["emlm_labels"].to(device, dtype=torch.long)

            model.zero_grad()
            mlm_outputs, emlm_outputs = model(inputs_ids)
            # print('outputs: ', outputs.shape)
            # print('labels: ', outputs.shape)
            mlm_loss, emlm_loss = loss_fn(mlm_outputs, emlm_outputs, mlm_label, emlm_label)# * 100
            loss = mlm_loss + emlm_loss
            # loss.backward()
            # optimizer.step()
            # if scheduler:
            #     scheduler.step()
            torch.cuda.empty_cache()

            losses.update(loss.item(), inputs_ids.size(0))
            mlm_losses.update(mlm_loss.item(), inputs_ids.size(0))
            emlm_losses.update(emlm_loss.item(), inputs_ids.size(0))
            tk0.set_postfix(loss=losses.avg, mlm_loss=mlm_losses.avg, emlm_loss=emlm_losses.avg)
    return losses.avg, mlm_losses.avg, emlm_losses.avg

def finetuning_train_fn(dataset, data_loader, model, optimizer, loss_fn, device, scheduler=None):
    model.train()
    losses = AverageMeter()
    clf_losses = AverageMeter()
    dis_losses = AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader), desc="Train")
    for bi, d in enumerate(tk0):
        # print(d)
        torch.cuda.empty_cache()
        src_entities = d["src_entities"]#.numpy().tolist()
        rels = d["rels"]#.numpy().tolist()
        tgt_entities = d["tgt_entities"]#.numpy().tolist()
        if dataset.dataset == 'WN18RR':
            src_entities = src_entities.numpy().tolist()
            tgt_entities = tgt_entities.numpy().tolist()

        src_input_ids = d["src_input_ids"].to(device, dtype=torch.long)
        src_attention_mask = d["src_attention_mask"].to(device, dtype=torch.long)
        src_token_type_ids = d["src_token_type_ids"].to(device, dtype=torch.long)
        tgt_input_ids = d["tgt_input_ids"].to(device, dtype=torch.long)
        tgt_attention_mask = d["tgt_attention_mask"].to(device, dtype=torch.long)
        tgt_token_type_ids = d["tgt_token_type_ids"].to(device, dtype=torch.long)
        # labels = d["labels"].to(device, dtype=torch.long)

        model.zero_grad()

        src_embedding = model.module.encoder(src_input_ids, attention_mask=src_attention_mask, token_type_ids=src_token_type_ids)
        tgt_embedding = model.module.encoder(tgt_input_ids, attention_mask=tgt_attention_mask, token_type_ids=tgt_token_type_ids)

        src_emb_list, tgt_emb_list = [], []
        for idx in range(src_embedding.shape[0]):
            src_emb_list.append(src_embedding[idx].repeat(tgt_embedding.shape[0]).reshape(tgt_embedding.shape[0], -1))
            tgt_emb_list.append(tgt_embedding)

        src_repeat_embedding = torch.stack(src_emb_list, 0).reshape(-1, tgt_embedding.shape[-1])
        tgt_repeat_embedding = torch.stack(tgt_emb_list, 0).reshape(-1, tgt_embedding.shape[-1])

        repeat_clf_labels, repeat_contrast_weights, repeat_contrast_margins = [], [], []
        weight_num, clf_weights = [1., 0.01], []
        for src_ent, rel_ent in zip(src_entities, rels):
            for tgt_ent in tgt_entities:
                if tgt_ent in dataset.subj2objs[src_ent][rel_ent]:
                    repeat_clf_labels.append(1.)
                    repeat_contrast_weights.append(1)
                    repeat_contrast_margins.append(0)
                    clf_weights.append(weight_num[0])
                else:
                    repeat_clf_labels.append(0.)
                    repeat_contrast_weights.append(-1)
                    repeat_contrast_margins.append(1)
                    clf_weights.append(weight_num[1])

        # src_emb_list2, tgt_emb_list2 = [], []
        # for _i in range(len(src_entities)-1):
        #     for _j in range(_i+1, len(src_entities)):
        #         src_emb_list2.append(src_embedding[_i])
        #         tgt_emb_list2.append(src_embedding[_j])
        #
        #         if (src_entities[_i] == src_entities[_j]) and (rels[_i] == rels[_j]):
        #             repeat_clf_labels.append(1)
        #             repeat_contrast_weights.append(1)
        #             repeat_contrast_margins.append(0)
        #         else:
        #             repeat_clf_labels.append(0)
        #             repeat_contrast_weights.append(-1)
        #             repeat_contrast_margins.append(1)
        # src_embedding2 = torch.stack(src_emb_list2, 0).reshape(-1, tgt_embedding.shape[-1])
        # tgt_embedding2 = torch.stack(tgt_emb_list2, 0).reshape(-1, tgt_embedding.shape[-1])
        # src_repeat_embedding = torch.cat((src_repeat_embedding, src_embedding2), 0)
        # tgt_repeat_embedding = torch.cat((tgt_repeat_embedding, tgt_embedding2), 0)


        # print('repeat_clf_labels: ', len(repeat_clf_labels))
        # print(repeat_clf_labels)
        # print(repeat_contrast_weights)
        # print(repeat_contrast_margins)
        repeat_clf_labels = torch.tensor(repeat_clf_labels, dtype=torch.float).to(device, dtype=torch.float)
        repeat_contrast_weights = torch.tensor(repeat_contrast_weights, dtype=torch.float).to(device, dtype=torch.float)
        repeat_contrast_margins = torch.tensor(repeat_contrast_margins, dtype=torch.float).to(device, dtype=torch.float)

        clf_weights = torch.tensor(clf_weights, dtype=torch.float).to(device, dtype=torch.float)

        outputs = model.module.classifier(src_repeat_embedding, tgt_repeat_embedding)
        distances = model.module.distance_fn(src_repeat_embedding, tgt_repeat_embedding)

        # print('outputs: ', outputs.min(), outputs.max(), outputs)
        clf_loss = loss_fn(outputs, repeat_clf_labels, clf_weights)# * torch.tensor(clf_weights, dtype=torch.float).to(device, dtype=torch.float).unsqueeze(-1)
        dis_loss = torch.mean(repeat_contrast_margins + repeat_contrast_weights * torch.sigmoid(distances.sum(-1)))
        # dis_loss = torch.mean(torch.relu(1 + repeat_contrast_weights * distances)) / 10
        loss = clf_loss + dis_loss#   # F.logsigmoid
        # loss /= ACCUMULATION_STEPS
        # loss = torch.mean(contrast_margin + contrast_weight * F.sigmoid(distances.sum(-1)))
        # loss = loss.mean()
        # print('loss: ', loss)
        loss.backward()
        optimizer.step()
        # # print('loss.backward(): ', loss.backward())
        # if (bi + 1) % ACCUMULATION_STEPS == 0:
        #     optimizer.step()  # 更新参数
        #     optimizer.zero_grad()
        if scheduler:
            scheduler.step()
        torch.cuda.empty_cache()

        losses.update(loss.item(), src_input_ids.size(0))
        clf_losses.update(clf_loss.item(), src_input_ids.size(0))
        dis_losses.update(dis_loss.item(), src_input_ids.size(0))
        tk0.set_postfix(loss=losses.avg, clf_loss=clf_losses.avg, dis_loss=dis_losses.avg)

        # if bi >= 30:
        #     break
def finetuning_eval_fn(dataset, data_loader, model, loss_fn, device):
    pred_list = []
    model.eval()
    losses = AverageMeter()
    clf_losses = AverageMeter()
    dis_losses = AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader), desc="Eval")
    with torch.no_grad():
        for bi, d in enumerate(tk0):
            torch.cuda.empty_cache()
            src_entities = d["src_entities"]#.numpy().tolist()
            rels = d["rels"]
            tgt_entities = d["tgt_entities"]#.numpy().tolist()
            if dataset.dataset == 'WN18RR':
                src_entities = src_entities.numpy().tolist()
                tgt_entities = tgt_entities.numpy().tolist()

            src_input_ids = d["src_input_ids"].to(device, dtype=torch.long)
            src_attention_mask = d["src_attention_mask"].to(device, dtype=torch.long)
            src_token_type_ids = d["src_token_type_ids"].to(device, dtype=torch.long)
            tgt_input_ids = d["tgt_input_ids"].to(device, dtype=torch.long)
            tgt_attention_mask = d["tgt_attention_mask"].to(device, dtype=torch.long)
            tgt_token_type_ids = d["tgt_token_type_ids"].to(device, dtype=torch.long)
            # labels = d["labels"].to(device, dtype=torch.long)

            model.zero_grad()
            src_embedding = model.module.encoder(src_input_ids, attention_mask=src_attention_mask,
                                          token_type_ids=src_token_type_ids)
            tgt_embedding = model.module.encoder(tgt_input_ids, attention_mask=tgt_attention_mask,
                                          token_type_ids=tgt_token_type_ids)

            src_emb_list, tgt_emb_list = [], []
            for idx in range(src_embedding.shape[0]):
                src_emb_list.append(
                    src_embedding[idx].repeat(src_embedding.shape[0]).reshape(src_embedding.shape[0], -1))
                tgt_emb_list.append(tgt_embedding)

            src_repeat_embedding = torch.stack(src_emb_list, 0).reshape(-1, src_embedding.shape[-1])
            tgt_repeat_embedding = torch.stack(tgt_emb_list, 0).reshape(-1, tgt_embedding.shape[-1])

            repeat_clf_labels, repeat_contrast_weights, repeat_contrast_margins = [], [], []
            weight_num, clf_weights = [1., 0.01], []
            for src_ent, rel_ent in zip(src_entities, rels):
                for tgt_ent in tgt_entities:
                    if tgt_ent in dataset.subj2objs[src_ent][rel_ent]:
                        repeat_clf_labels.append(1.)
                        repeat_contrast_weights.append(1)
                        repeat_contrast_margins.append(0)
                        clf_weights.append(weight_num[0])
                    else:
                        repeat_clf_labels.append(0.)
                        repeat_contrast_weights.append(-1)
                        repeat_contrast_margins.append(1)
                        clf_weights.append(weight_num[1])
            repeat_clf_labels = torch.tensor(repeat_clf_labels, dtype=torch.float).to(device, dtype=torch.float)
            repeat_contrast_weights = torch.tensor(repeat_contrast_weights, dtype=torch.float).to(device, dtype=torch.float)
            repeat_contrast_margins = torch.tensor(repeat_contrast_margins, dtype=torch.float).to(device, dtype=torch.float)

            clf_weights = torch.tensor(clf_weights, dtype=torch.float).to(device, dtype=torch.float)

            outputs = model.module.classifier(src_repeat_embedding, tgt_repeat_embedding)
            distances = model.module.distance_fn(src_repeat_embedding, tgt_repeat_embedding)

            clf_loss = loss_fn(outputs, repeat_clf_labels, clf_weights)# * torch.tensor(clf_weights, dtype=torch.float).to(device, dtype=torch.float).unsqueeze(-1)
            dis_loss = torch.mean(repeat_contrast_margins + repeat_contrast_weights * torch.sigmoid(distances.sum(-1)))
            # dis_loss = torch.mean(torch.relu(1 + repeat_contrast_weights * distances)) / 10
            loss = clf_loss + dis_loss #   # F.logsigmoid
            # loss = torch.mean(contrast_margin + contrast_weight * F.sigmoid(distances.sum(-1)))
            # loss = loss.mean()
            outputs = outputs.cpu().detach().numpy().astype(float)
            pred_list.append(outputs)

            losses.update(loss.item(), src_input_ids.size(0))
            clf_losses.update(clf_loss.item(), src_input_ids.size(0))
            dis_losses.update(dis_loss.item(), src_input_ids.size(0))
            tk0.set_postfix(loss=losses.avg, clf_loss=clf_losses.avg, dis_loss=dis_losses.avg)
    return losses.avg, np.concatenate(pred_list).reshape(-1)
