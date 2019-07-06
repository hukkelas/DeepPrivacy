import torch
import math 
# Rerference implementation: https://pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html


class BaseSampler(torch.utils.data.Sampler):

    def __init__(self, dataset):
        num_replicas = 1
        rank = 0
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.idx = 0

    def __len__(self):
        return self.num_samples


class ValidationSampler(BaseSampler):

    def __iter__(self):
        indices = torch.arange(0, len(self.dataset)).tolist()
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size, "Was {} indices, expected: {}".format(len(indices), self.total_size)

        # subsample
        sample_size = len(indices) // self.num_replicas
        indices = indices[self.rank*sample_size:(self.rank+1)*sample_size]
        assert len(indices) == self.num_samples
        return iter(indices)


class TrainSampler(BaseSampler):

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.idx)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        indices += indices[:(self.total_size - len(indices))]
 
        assert len(indices) == self.total_size, "Was {} indices, expected: {}".format(len(indices), self.total_size)

        sample_size = len(indices) // self.num_replicas
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        self.idx += 1

        return iter(indices)
