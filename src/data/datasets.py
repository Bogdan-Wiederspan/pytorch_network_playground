import torch
import torch.utils.data as t_data
from collections import defaultdict

from src.utils.logger import get_logger
logger = get_logger(__name__)

class EraDataset(t_data.Dataset):
    def __init__(
        self,
        inputs: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor=None,
        name: str=None,
        era: str=None,
        dataset_type: str=None,
    ):
        self.inputs = inputs
        self.targets = target
        self.weight = weight
        self.name = name
        self.era = era
        self.size = len(self)
        self.reset()
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def reset(self):
        self.current_idx = 0
        self.indices = torch.randperm(len(self))

    def sample(self, number):
        if number > len(self.inputs):
            raise ValueError(f"Cannot sample {number} elements from dataset of size {len(self)}")

        start_idx = self.current_idx
        # do not drop last, but instead create smaller batch
        next_idx = min(self.current_idx + number, len(self))

        idx = self.indices[start_idx:next_idx]
        # reset if we reached the end of the dataset and drop last incomplete batch
        data = self.inputs[idx], self.targets[idx]
        self.current_idx = next_idx
        if self.current_idx == len(self.inputs):
            self.reset()
        return data


class EraDatasetSampler(t_data.Sampler):
    def __init__(self, datasets=None, batch_size=1, sample_ratio={"dy": 0.25, "tt": 0.25, "hh": 0.5}):
        self.total_weight = {"dy":0, "tt":0, "hh":0}
        self.datasets = defaultdict(dict)  # {dataset_type: {(era, process_id): dataset}}
        self.weights = defaultdict(dict)
        self.batch_size = torch.Tensor([batch_size])
        if datasets is not None:
            for dataset in datasets:
                self.add_dataset(dataset)
        self.sample_ratio = sample_ratio

    def add_dataset(self, dataset):
        # {era : process_id: {array}}
        weight = dataset.weight
        self.total_weight[dataset.dataset_type] += weight
        self.datasets[dataset.dataset_type][(dataset.era, dataset.name)] = dataset

    def print_sample_rate(self, batch=False):
        logger.info(f"Sample rate for Dataset of Era")
        batch_size = 1
        if batch:
            batch_size = self.batch_size
        for era, pid in self.datasets.items():
            for ds, weight in pid.values():
                if weight is not None:
                    rate = torch.ceil(self.batch_size * weight / self.total_weight).to(torch.int32)
                    logger.info(f"{ds.name} | {ds.era} | {rate}")

    def calculate_sample_size(self, dataset_type, min_size=1):
        # setup unpack the dataset dictionary and convert to tensors
        min_size = torch.tensor(min_size)
        unique_identifier = self.datasets[dataset_type]
        total_weight = self.total_weight[dataset_type]
        sample_ratio = self.sample_ratio[dataset_type]
        sub_batch_size = int(self.batch_size * sample_ratio)

        weights = []
        keys = []

        for (era, pid), ds in unique_identifier.items():
            weights.append(ds.weight.item())
            keys.append((era, pid))
        weights = torch.tensor(weights, dtype=torch.float32)

        # start algorithm

        # calculate batch size according to weight
        total_weight = weights.sum()
        ideal_sizes = sub_batch_size * weights / total_weight

        # downsample calculated batch size --> reduces batchsize by length
        # upsample below threshold --> increases batch size
        # downsample biggest batches .--> reduces batch size
        # sum should be exactly initial batch size
        floored = torch.floor(ideal_sizes)
        floored = torch.maximum(floored, min_size)

        num_too_much = int(floored.sum()) - sub_batch_size

        # distribute fraction then floor again and then remove from biggest the residue
        m = floored != min_size

        floored_indices = torch.arange(len(floored))

        sub_set_floored = floored[m]
        sel_floor_values, sel_floor_idx = torch.sort(sub_set_floored / sub_set_floored.sum() * num_too_much)

        # floor below median
        median = sel_floor_idx.median()
        below_median = sel_floor_idx < median
        above_median = sel_floor_idx >= median
        sel_floor_values[below_median] = torch.floor(sel_floor_values[below_median])
        sel_floor_values[above_median] = torch.ceil(sel_floor_values[above_median])

        floored[floored_indices[m][sel_floor_idx]] -= sel_floor_values
        return floored

    @property
    def __len__(self):
        return sum([len(ds) for ds in self.datasets])

    def get_batch(self, shuffle_batch=True):
        # loop over dataset classes and ask them to generate a certain number of samples

        # shuffle finished batch optionally
        # Get a batch of data from each dataset
        batch_input = []
        batch_target = []
        # from IPython import embed; embed(header="GETBATCH - 66 in datasets.py ")
        for era, pid in self.datasets.items():
            for ds in pid.values():
                input, target = ds.sample(torch.ceil(self.batch_size * ds.weight / self.total_weight).to(torch.int32))
                batch_input.append(input)
                batch_target.append(target)
                # batch_target.append(target)
                # batch_weights.append(w)
                # if ds.weight is not None:
                #

        batch_input = torch.concatenate(batch_input, dim=0)
        batch_target = torch.concatenate(batch_target, dim=0)
        if shuffle_batch:
            indices = torch.randperm(len(batch_input))
            batch_input = batch_input[indices]
            batch_target = batch_target[indices]
        return batch_input, batch_target






# _default_categorical_features = expand_braces(
#     "pair_type",
#     "decay_mode1",
#     "decay_mode2",
#     "lepton1.charge",
#     "lepton2.charge",
#     "has_fatjet",
#     "has_jet_pair",
#     "year_flag",
# )
# _default_continuous_features = expand_braces(
#     "bjet1.{btagPNetB,btagPNetCvB,btagPNetCvL,energy,hhbtag,mass,px,py,pz}",
#     "bjet2.{btagPNetB,btagPNetCvB,btagPNetCvL,energy,hhbtag,mass,px,py,pz}",
#     "fatjet.{energy,mass,px,py,pz}",
#     "lepton1.{energy,mass,px,py,pz}",
#     "lepton2.{energy,mass,px,py,pz}",
#     "PuppiMET.{px,py}",
#     "reg_dnn_nu{1,2}_{px,py,pz}",
# )
# _default_lbn_features = expand_braces(
#     "bjet1.{energy,mass,px,py,pz}",
#     "bjet2.{energy,mass,px,py,pz}",
#     "fatjet.{energy,mass,px,py,pz}",
#     "lepton1.{energy,mass,px,py,pz}",
#     "lepton2.{energy,mass,px,py,pz}",
#     "PuppiMET.{energy,mass,px,py}",
#     "reg_dnn_nu{1,2}_{px,py,pz}",
#     )
