import torch
import torch.utils.data as t_data
from collections import defaultdict



class EraDataset(t_data.Dataset):
    def __init__(self, inputs: torch.Tensor, target: torch.Tensor, weight: torch.Tensor=None, name: str=None, era: str=None):
        self.inputs = inputs
        self.targets = target
        self.weight = weight
        self.name = name
        self.era = era
        self.size = len(self)
        self.reset()

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
    def __init__(self, datasets=None, batch_size=1):
        self.datasets = defaultdict(dict)  # {era : {process_id : EraDataset}}
        self.batch_size = torch.Tensor([batch_size])
        if datasets is not None:
            for dataset in datasets:
                self.add_dataset(dataset)
        self._total_weight = None

    def add_dataset(self, dataset):
        # {era : process_id: {array}}
        self.datasets[dataset.era][dataset.name] = dataset

    @property
    def total_weight(self):
        if self._total_weight is None:
            total_weight = 0
            for era, pid in self.datasets.items():
                for ds in pid.values():
                    if ds.weight is not None:
                        total_weight += ds.weight
                    else:
                        total_weight += 1
            self._total_weight = total_weight
        return self._total_weight

    @property
    def __len__(self):
        return sum([len(ds) for ds in self.datasets])

    def get_batch(self, shuffle_batch=True):
        # loop over dataset classes and ask them to generate a certain number of samples

        # shuffle finished batch optionally
        # Get a batch of data from each dataset
        batch_input = []
        batch_target = []
        from IPython import embed; embed(header="GETBATCH - 66 in datasets.py ")
        for era, pid in self.datasets.items():
            for ds in pid.values():
                from IPython import embed; embed(header="LOOP - 82 in datasets.py ")
                input, target = ds.sample(torch.ceil(self.batch_size * ds.weight * self.total_weight))
                batch_input.append(input)
                batch_target.append(target)
                # batch_target.append(target)
                # batch_weights.append(w)
                # if ds.weight is not None:
                #



        batch_data = torch.concatenate(batch_data, dim=0)
        if shuffle_batch:
            indices = torch.randperm(len(batch_data))
            batch_data = batch_data[indices]
        return batch_data






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
