import awkward as ak
import numpy as np
import torch



class DatasetPerConfig(torch.utils.data.Dataset):
    def __init__(self, data_map, input_features, target_map, config_name):
        self.data = ak.concatenate([x.data for x in data_map if x.config_name == config_name])
        self.input_features = input_features
        self.target_map = target_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # extract input features
        inputs = np.stack([feature.apply(item).to_numpy().astype(np.float32) for feature in self.input_features], axis=-1)
        # extract targets
        targets = np.stack([item[target] if target in item else np.array([]) for target in self.target_map.keys()], axis=-1).astype(np.int64)
        return torch.from_numpy(inputs), torch.from_numpy(targets)
