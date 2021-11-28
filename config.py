import os
import json
import random
import numpy as np
import torch

from transformers import BertConfig

rel_num_dict = {
    'WebNLG': 216,
    'WebNLG-star': 171
}

Label2IdxSub = {"B-H": 1, "I-H": 2, "O": 0}
Label2IdxObj = {"B-T": 1, "I-T": 2, "O": 0}

class Config(object):
    """ global config for whole project """

    # global config can just white here

    def __init__(self, args):
        """ common configs that need not to be changed frequently can be predefined here. """
        self.args = args

        index = self.args.index

        # common
        self.early_stop = 30
        self.seed = 12138
        self.seed_torch(self.seed)

        self.hidden_size = 768
        self.max_text_len = 150
        self.drop_prob = 0.3

        # paths
        self.data_dir = os.path.join('data', self.args.corpus_type)
        self.log_dir = os.path.join('logs', index)
        self.model_dir = os.path.join('models', index)
        self.log_path = os.path.join(self.log_dir, 'train.log')

        # bert
        self.bert_model_dir = '../Bert/bert-base-cased'
        self.bert_config = BertConfig.from_pretrained(os.path.join(self.bert_model_dir, 'config.json'))

        # rels
        with open(os.path.join(self.data_dir, 'rel2id.json'), mode='r', encoding='utf-8') as f:
            self.rel2idx = json.load(f)[-1]
            self.idx2rel = dict(zip(self.rel2idx.values(), self.rel2idx.keys()))
            self.rel_num = len(self.rel2idx)

        # device
        torch.cuda.set_device(self.args.device_id)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # learning rate
        self.fin_tuning_lr = 1e-4
        self.downs_en_lr = 1e-3
        self.clip_grad = 2.
        self.drop_prob = 0.3  # dropout
        self.weight_decay_rate = 0.01
        self.warmup_prop = 0.1
        # self.gradient_accumulation_steps = 2

        pass

    def get_args_info(self):
        args_d = vars(self.args)
        info = "\nRunning with the following arguments:\n"
        for k,v in args_d.items():
            info += " >> {} : {}\n".format(k, str(v))
        return info

    def get_config_info(self):
        params = {}
        sent = "\nConfigs: \n"
        for k, v in self.__dict__.items():
            if isinstance(v, (str, int, float, bool)):
                sent += " >> {}: {}\n".format(k, v)
                params[k] = v
        return sent

    def seed_torch(self, seed=12138):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)