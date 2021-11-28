from numpy.lib.twodim_base import tri
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from dataset import CustomDataset, TripleDataset, SentDataset

import numpy as np



def get_dataloader(config, data_sign='train'):
    dataset = SentDataset(config=config, data_sign=data_sign)
    print("{} dataset load finished! ({})".format(data_sign, len(dataset)))

    if data_sign == 'train':
        datasampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=datasampler, batch_size=config.args.batch_size, collate_fn=dataset.collate_fn_train)
    elif data_sign in ['test', 'val']:
        datasampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=datasampler, batch_size=config.args.batch_size * 2, collate_fn=dataset.collate_fn_test)
    else:
        raise ValueError('Can not find "{}" in predefined signs!'.format(data_sign))
    
    return dataloader

