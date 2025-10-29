import torch
import torch.utils.data as t_data
from collections import defaultdict

from utils.logger import get_logger
logger = get_logger(__name__)

class EraDataset(t_data.Dataset):
    def __init__(
        self,
        continous_tensor: torch.Tensor,
        categorical_tensor: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor=None,
        name: str=None,
        era: str=None,
        dataset_type: str=None,
    ):
        self.continous_input = continous_tensor
        self.categorical_input = categorical_tensor
        self.targets = target
        self.weight = weight
        self.name = name
        self.era = era
        self.reset()
        self.dataset_type = dataset_type
        self.sample_size = None

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        return self.continous_input[idx], self.continous_input[idx], self.targets[idx]

    def reset(self):
        self.current_idx = 0
        self.indices = torch.randperm(len(self))

    def sample(self, number=None):
        if number is None:
            number = self.sample_size

        if number > len(self):
            raise ValueError(f"Cannot sample {number} elements from dataset of size {len(self)}")

        start_idx = self.current_idx
        # do not drop last, but instead create smaller batch
        next_idx = min(self.current_idx + number, len(self))

        idx = self.indices[start_idx:next_idx]
        # reset if we reached the end of the dataset and drop last incomplete batch
        try:
            data = self.continous_input[idx], self.categorical_input[idx],self.targets[idx]
        except:
            from IPython import embed; embed(header="string - 55 in datasets.py ")
        self.current_idx = next_idx
        if self.current_idx >= len(self):
            self.reset()
        return data


class EraDatasetSampler(t_data.Sampler):
    def __init__(
        self,
        datasets=None,
        batch_size=1,
        sample_ratio={"dy": 0.25, "tt": 0.25, "hh": 0.5},
        min_size = None,
        ):
        self.total_weight = {"dy":0, "tt":0, "hh":0}
        self.datasets = defaultdict(dict)  # {dataset_type: {(era, process_id): dataset}}
        self.weights = defaultdict(dict)
        self.batch_size = torch.Tensor([batch_size])
        if datasets is not None:
            for dataset in datasets:
                self.add_dataset(dataset)
        self.sample_ratio = sample_ratio
        self.min_size = min_size # index where to split continous and categorical data

    def add_dataset(self, dataset):
        # {era : process_id: {array}}
        weight = dataset.weight
        self.total_weight[dataset.dataset_type] += weight
        self.datasets[dataset.dataset_type][(dataset.era, dataset.name)] = dataset

    @property
    def keys(self):
        return list(self.datasets.keys())

    @property
    def sample_sizes(self):
        _sample_sizes = []
        for ds_type in self.keys:
            _sample_sizes = _sample_sizes + [ds.sample_size for ds in self.datasets[ds_type]]
        return _sample_sizes

    def print_sample_rates(self):
        logger.info(f"Sample rate for Dataset of Era")

        for dataset_type, uid in self.datasets.items():
            total_batch = 0
            logger.info(f"{dataset_type}:")
            for (era, pid), ds in uid.items():
                logger.info(f"\t{pid} | {era} | {ds.sample_size}")
                total_batch += ds.sample_size
            logger.info(f"\tsum/expected: {total_batch}/{(self.sample_ratio[ds.dataset_type] * self.batch_size).item()}")

    def calculate_sample_size(self, dataset_type, dry=False):
        logger.info(f"Calculate sample sizes for: {dataset_type}")
        # from IPython import embed; embed(header="string - 84 in datasets.py ")
        # unpack and convert to tensors for easier handling, get desired_sub_batch_size
        weights, keys = [], []
        for (era, pid), ds in self.datasets[dataset_type].items():
            weights.append(ds.weight.item())
            keys.append((era, pid))
        weights = torch.tensor(weights, dtype=torch.float32)
        min_size = torch.tensor(self.min_size)
        sub_batch_size = int(self.batch_size * self.sample_ratio[dataset_type])

        # start algorithm
        # 2 cases: Phase phases need to be upsampeled OR phase spaces are already reasonable high
        #
        # in 3 steps:
        # 1. downsample calculated batch size and upsample to minimal threshold
        # 2. calculate relative weight of overflow for samples above minimal threshold
        # 3. distribute overflow from 2. by ceil and floor half of the sample according to median
        # 4. reduce  all steps resulting in sub_batch_size batch composition


        # preperation: calculate batch size according to weight
        total_weight = weights.sum()
        ideal_sizes = sub_batch_size * weights / total_weight

        # step 1:

        # check if UPSAMPLING IS NECESSARY AT ALL
        floored_sizes = torch.floor(ideal_sizes) # decreases sub batch size by length of tensor
        floored_sizes = torch.maximum(floored_sizes, min_size) # increase sub batch size by different to threshold
        # calculate overflow of sub batch
        overflow_size = int(floored_sizes.sum()) - sub_batch_size

        # step 2:
        # calculate relative contribution of events above threshold

        mask_above_threshold = floored_sizes > min_size
        floored_above_threshold = floored_sizes[mask_above_threshold]
        floored_overflow_sizes = floored_above_threshold / floored_above_threshold.sum() * overflow_size
        # step 3: floor below median, ceil for weights above
        # sort to be able to use median properlly
        floored_overflow_sizes, floored_oveflow_idx = torch.sort(floored_overflow_sizes)

        # define mask for floor and ceil
        # median -> even: under median so idx len/2 - 1 (change mask to inclusive)
        # median -> uneven: exact middle idx len/2

        # even uneven median treatment
        if len(floored_oveflow_idx) % 2 == 0:
            median = floored_oveflow_idx.median()
            below_median = floored_oveflow_idx <= median
            above_median = floored_oveflow_idx > median
        else:
            median = floored_oveflow_idx.median()
            below_median = floored_oveflow_idx < median
            above_median = floored_oveflow_idx >= median

        # apply floor and ceil for targets below and above median
        floored_overflow_sizes[below_median] = torch.floor(floored_overflow_sizes[below_median])
        floored_overflow_sizes[above_median] = torch.ceil(floored_overflow_sizes[above_median])

        # step 4: apply overflow to original floored
        # combine sort indices and threshold mask
        indices_above_threshold = torch.arange(len(floored_sizes))[mask_above_threshold][floored_oveflow_idx]
        floored_sizes[indices_above_threshold] -= floored_overflow_sizes

        # conver to int but check if values changed, just to be safe
        if (floored_sizes.to(torch.int32).sum() - floored_sizes.sum()) != 0:
            raise TypeError(f"Change of type changed value")
        floored_sizes = floored_sizes.to(torch.int32)

        # if one still mismatch add it on lowest or remove from highest
        very_last_remaining = int(floored_sizes.sum()) - sub_batch_size
        # from IPython import embed; embed(header="string - 144 in datasets.py ")
        if very_last_remaining <= -1:
            floored_sizes[indices_above_threshold[0]] += very_last_remaining
        elif very_last_remaining >= 1:
            floored_sizes[indices_above_threshold[-1]] -= very_last_remaining
        elif very_last_remaining != 0:
            from IPython import embed; embed(header=f"{floored_sizes.sum()} but should be {sub_batch_size}")
            raise ValueError(f"Resampling failed: Created batch size of size: {floored_sizes.sum()} but should be {sub_batch_size}")

        # store batch size per phase space in dataset
        if not dry:
            for k, w in zip(keys, floored_sizes):
                self.datasets[dataset_type][k].sample_size = w.item()
        else:
            logger.info({k:w.item() for k,w in zip(keys, floored_sizes)})


    @property
    def __len__(self):
        return sum([len(ds) for ds in self.datasets])

    def get_batch(self, device=torch.device("cpu"), shuffle_batch=True):
        # loop over dataset classes and ask them to generate a certain number of samples

        # shuffle finished batch optionally
        # Get a batch of data from each dataset
        batch_cont, batch_cat, batch_target = [], [], []

        # from IPython import embed; embed(header="GETBATCH - 66 in datasets.py ")
        for _, uid in self.datasets.items():
            for ds in uid.values():
                cont, cat, target = ds.sample()
                batch_cont.append(cont), batch_cat.append(cat), batch_target.append(target)

        batch_cont = torch.concatenate(batch_cont, dim=0)
        batch_cat = torch.concatenate(batch_cat, dim=0)
        batch_target = torch.concatenate(batch_target, dim=0)

        if shuffle_batch:
            # needs to be depending on data due to last incomplete batch
            indices = torch.randperm(batch_cont.shape[0])
            batch_cont = batch_cont[indices]
            batch_cat = batch_cat[indices]
            batch_target = batch_target[indices]
        return batch_cont.to(device), batch_cat.to(device), batch_target.to(device)
