import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import AutoTokenizer


class AlignCollate:
    def __init__(self, max_length=50):
        self.tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
        self.max_length = max_length

    def __call__(self, batch):
        new_img = []
        new_label = []
        new_len = []
        for data in batch:
            label = data['label']
            input_ids = [1] * (self.max_length + 2)
            ids = self.tokenizer.encode(label)
            input_ids[:len(ids)] = ids
            input_ids = torch.tensor(input_ids)
            new_img.append(data['img'])
            new_label.append(input_ids)
            new_len.append(data['len'])
        result = []
        result.append(torch.stack(new_img))
        result.append(torch.stack(new_label))
        result.append(torch.tensor(new_len))
        return result


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=AlignCollate(max_length=50)):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
