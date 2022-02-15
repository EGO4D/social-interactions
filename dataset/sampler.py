import torch
import numpy as np
from torch.utils.data import Sampler

class SequenceBatchSampler(Sampler):

    def __init__(self, data_source, batch_size, shuffle=True, drop_last=False):

        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.indices = list(range(len(data_source)))
        self.batches = []
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        if self.shuffle:
            np.random.shuffle(self.data_source.segments)
            self.data_source.segments.sort(key=lambda x: x[-2]-x[-3], reverse=True)

        self.batches = []
        start = 0        
        while True:
            segment = self.data_source.segments[start]
            length = int(segment[-2] - segment[-3])
            end = min(len(self.data_source.segments), start + max(int(self.batch_size / length), 1))
            self.batches.append(list(range(start, end)))
            if end == len(self.data_source.segments):
                break
            start = end

        if self.shuffle:
            self.batches = [
                torch.tensor(batch)[torch.randperm(len(batch), generator=g)].tolist() for batch in self.batches
            ]
            np.random.shuffle(self.batches)

        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

    def set_epoch(self, epoch):
        self.epoch = epoch
