import logging
import threading
from tqdm import trange

import utils
import torch
import torch.nn as nn
import numpy as np
from dataset import convert_idx_to_triple

def evaluate(model, dataloader, config, eval_type='val'):
    """ Evaluate model in given dataset."""

    model.eval()

    items = []
    correct_num, pred_num, gold_num = 1e-10, 1e-10, 1e-10

    t = trange(len(dataloader), ascii=True)
    with torch.no_grad():
        for step, _ in enumerate(t):
            batch = next(iter(dataloader))

            ids, masks, triples, tokens, texts = batch
            ids, masks = ids.to(config.device), masks.to(config.device)

            outputs, cor_pred, rels_pred, ents_pred = model(ids, masks)
            preds = [extract_triples(o, mask, rels) for o, mask, rels in zip(outputs, masks, rels_pred)]

            for pred, gold in zip(preds, triples):
                gold_num += len(gold)
                pred_num += len(pred)
                correct_num += len(set(pred) & set(gold))
            
            if eval_type == 'test':
                items.extend(test_result(preds, triples, tokens, texts, config.idx2rel))

    p = correct_num / pred_num
    r = correct_num / gold_num
    f1 = 2 * p*r / (p+r)

    logging.info("Eval Result: F1: {:.4f}, P: {:.4f}, R: {:.4f}".format(f1, p, r))

    if eval_type == 'test':
        return f1, p, r, items
    else:
        return f1, p, r


def extract_triples(output, mask, rels, threshold=0.5, rel_threshold=0.5):
    """ extract triple by the output of model
      output: L * L * |R|
    """
    triples = []
    output = output.permute(2, 0, 1)
    L = mask.sum()
    # for rel in rels:
    for rel in torch.where(rels > rel_threshold)[0]:
        if output[rel].max() < threshold:
            continue

        visited = np.zeros([L, L], dtype=bool)
        for i in range(L):
            for j in range(L):
                if not visited[i,j] and output[rel, i, j] > threshold:
                    visited[i,j] = 1
                    x, y = find_lower_right_index(i, j, output[rel], threshold)
                    triples.append((rel.item(), i, x+1, j, y+1))
                    visited[i:x+1,j:y+1] = 1

    return triples

def find_lower_right_index(i, j, table, threshold):
    L = len(table[0])
    x, y = i, j
    while True:
        r = table[x,y+1] if y+1 < L else 0
        d = table[x+1,y] if x+1 < L else 0
        c = table[x+1,y+1] if x+1 < L and y+1 < L else 0
        m = max(r,d,c)

        if m < threshold:
            return x, y

        if m == r:
            x, y = x, y+1
        elif m == d:
            x, y = x+1, y
        elif m == c:
            x, y = x+1, y+1
        else:
            return x, y

def test_result(preds, triples, tokens, texts, idx2rel):
    """ just for val stage"""
    items = []
    for pred, gold, token, text in zip(preds, triples, tokens, texts):
        gold = set([convert_idx_to_triple(token, g, idx2rel) for g in sorted(gold)])
        pred = set([convert_idx_to_triple(token, p, idx2rel) for p in sorted(pred)])

        correct = pred & gold
        erro = pred - gold
        lack = gold - pred

        items.append({
            "text": text,
            "token": token,
            "pred": list(pred),
            "gold": list(gold),
            "correct": list(correct),
            "error": list(erro),
            "lack": list(lack)
        })
    return items
        


