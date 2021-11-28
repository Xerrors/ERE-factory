
import utils
import logging
import torch
import torch.nn as nn
from optimization import BertAdam
from tqdm import trange
from evaluate import evaluate
from dataloader import get_dataloader
from utils.logger import save_result_to_csv_and_json

def train(model, config):
    # Prepare optimizer 看不懂的东西，先不管
    param_optimizer = list(model.named_parameters())
    param_pre = [(n, p) for n, p in param_optimizer if 'bert' in n]
    param_downstream = [(n, p) for n, p in param_optimizer if 'bert' not in n]
    no_decay = ['bias', 'LayerNorm', 'layer_norm']
    optimizer_grouped_parameters = [
        # pretrain model param
        {'params': [p for n, p in param_pre if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay_rate, 'lr': config.fin_tuning_lr
         },
        {'params': [p for n, p in param_pre if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': config.fin_tuning_lr
         },
        # downstream model
        {'params': [p for n, p in param_downstream if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay_rate, 'lr': config.downs_en_lr
         },
        {'params': [p for n, p in param_downstream if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': config.downs_en_lr
         }
    ]

    model.to(config.device)

    # dataset
    train_data = get_dataloader(config, 'train')    # 5019
    test_data = get_dataloader(config, 'test')      # 703
    val_data = get_dataloader(config, 'val')        # 500

    # optimizer
    num_steps = len(train_data) * config.args.epoch_num
    optimizer = BertAdam(optimizer_grouped_parameters, warmup=config.warmup_prop, schedule="warmup_cosine",
                         t_total=num_steps)

    best_f1 = 0.0
    early_stop = 0

    for epoch in range(1, config.args.epoch_num+1):
        logging.info("Epoch {}/{}".format(epoch, config.args.epoch_num))

        _train_model(model, train_data, optimizer, config)

        f1, p, r, items = evaluate(model, val_data, config, eval_type='test')
        # f1_test, p, r, items = evaluate(model, test_data, config, eval_type='test')
        save_result_to_csv_and_json(items, config.log_dir, "training_val")

        if f1 > best_f1:
            best_f1 = f1
            early_stop = 0
            logging.warning("Find new best F1 {:.4f}".format(best_f1))
        else:
            early_stop += 1

        if early_stop >= config.early_stop:
            logging.warning("Early stoping in epoch {} and the best F1 is ({:.6f})".format(epoch, best_f1))
            break

    logging.warning("Final evaluate in 'test set':")
    f1, p, r, items = evaluate(model, test_data, config, eval_type='test')
    save_result_to_csv_and_json(items, config.log_dir, "{}".format(int(f1*1e4)))


def _train_model(model, dataloader, optimizer, config):
    """ Training model in one epoch."""

    model.train()

    loss_avg = utils.RunningAverage()
    loss_func = nn.BCELoss(reduction='mean')

    t = trange(len(dataloader), ascii=True)
    for step, _ in enumerate(t):
        batch = next(iter(dataloader))
        batch = tuple(t.to(config.device) for t in batch)

        ids, masks, table_tags, neg_masks, cor_tags, rel_tags, ent_tags = batch
        table, cors, rels, ents = model(ids, masks) # bsz, L, L, |R|

        if config.args.use_negative_mask:
            table = table * neg_masks

        tab_loss = loss_func(table, table_tags)
        rel_loss = loss_func(rels, rel_tags)
        ent_loss = loss_func(ents, ent_tags)
        cor_loss = loss_func(cors, cor_tags)
        
        loss = 10 * tab_loss + 0 * rel_loss + 0 * ent_loss  + 0 * cor_loss

        loss.backward()
        optimizer.step()
        model.zero_grad()

        loss_avg.update(loss.item())
        t.set_postfix(loss='{:.6f}/{:.6f}'.format(loss_avg(), loss.item()))
    
    return loss_avg()

        
