import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class data_set(Dataset):

    def __init__(self, data,config=None):
        self.data = data
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, data):
        label = torch.tensor([item['relation'] for item in data])
        input_ids = torch.cat([item['tokens']['input_ids'] for item in data],dim=0)
        token_type_ids = torch.cat([item['tokens']['token_type_ids'] for item in data],dim=0)
        attention_mask = torch.cat([item['tokens']['attention_mask'] for item in data],dim=0)
        return (
            label,
            input_ids,
            token_type_ids,
            attention_mask
        )

def get_data_loader(config, data, shuffle = False, drop_last = False, batch_size = None):

    dataset = data_set(data, config)

    if batch_size == None:
        batch_size = min(config.batch_size_per_step, len(data))
    else:
        batch_size = min(batch_size, len(data))

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=False,
        num_workers=config.num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=drop_last)

    return data_loader

class data_set_raw(Dataset):

    def __init__(self, data,config=None):
        self.data = data
        self.config = config


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, data):
        label = torch.tensor([item['relation'] for item in data])
        input_sents = [item['tokens'] for item in data]

        return (
            label,
            input_sents
        )

def get_data_loader_raw(config, data, shuffle = False, drop_last = False, batch_size = None):

    dataset = data_set_raw(data, config)

    if batch_size == None:
        batch_size = min(config.batch_size_per_step, len(data))
    else:
        batch_size = min(batch_size, len(data))

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=False,
        num_workers=config.num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=drop_last)

    return data_loader