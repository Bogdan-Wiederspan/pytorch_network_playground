import torch
import torch.utils.data as t_data
from collections import defaultdict

from utils.logger import get_logger
logger = get_logger(__name__)

class Process(t_data.Dataset):

    def __init__(
        self,
        continous: torch.Tensor,
        categorical: torch.Tensor,
        target: torch.Tensor,
        normalization_weights: torch.Tensor,
        product_of_weights: torch.Tensor,
        total_normalization_weights: torch.Tensor,
        total_product_of_weights: torch.Tensor,
        evaluation_space_mask: torch.Tensor,
        process_id: str=None,
        process_type: str=None,
        randomize: bool =True,
    ):
        """
        A Process Dataset class represents all data connected to a process identified by *process id*.
        Each process is also associated with a *process type* like "tt", "dy, "hh" etc.
        The randomize flag controls whether the data is shuffled at the beginning of each full pass.
        The process data itself is stored as torch tensors in the attributes *continous*, *categorical* and *targets*.
        A process, knows about its individual *normalization_weights* and *product_of_weights*, but also the total weight of the process type.

        The normalization weights represent the scaling from oversampling of the MC generator.
        The product of weights
        The data is stored as tensors and is hold the actual data and weights corresponding to a process id and also all necessary meta information.


        The *product_of_weights* is the product of all event weights assigned during generation and reconstruction.

        Args:
            continous (torch.Tensor): Tensor with continous features of shape [num_events, num_continous_features]
            categorical (torch.Tensor): Tensor with categorical features of shape [num_events, num_categorical_features]
            target (torch.Tensor): Tensor with target one-hot encoded of shape [num_events, num_classes]
            normalization_weights (torch.Tensor, optional): Tensor with normalization weights per event of shape [num_events].
            product_of_weights (torch.Tensor, optional): Tensor of product of weights per event of shape [num_events].
            total_normalization_weights (Dict[torch.Tensor], optional): Single Tensor describing the sum over normalized weights over the whole MC population.
            total_product_of_weights (torch.Tensor, optional): Single Tensor describing the sum over all product of weights over the whole MC population.
            evaluation_space_mask (torch.Tensor, optional): Bool Tensor to bring events into evaluation phase space
            process_id (int, optional): Unique process id of the process. Defaults to None.
            process_type (str, optional): To which process type does the process belong to (ex. "hh"). Defaults to None.
            randomize (bool, optional): Defines if a new cycle of a process instance is shuffled. Defaults to True.
        """

        self.continous = continous.to(torch.float32)
        self.categorical = categorical.to(torch.int32)
        self.targets = target.to(torch.float32)

        # is product of all weights assigned during generation and reconstruction
        # necessary to transfer to real yield (real events one would see in data)
        self.product_of_weights = product_of_weights.to(torch.float32)

        # mc sampler are generated with higher lumi (to reduce stat. unc.)
        # normalization weights scale down to correct lumi
        self.normalization_weights = normalization_weights.to(torch.float32)

        # IMPORTANT: Due to splitting of events the total weights is the total of the WHOLE POPULATION
        # it is not just the sum over all splitted weights, which is the reason why it must be handed over!
        self.total_product_of_weights = total_product_of_weights.to(torch.float32)
        self.total_normalization_weights = total_normalization_weights.to(torch.float32)
        self.evaluation_space_mask = evaluation_space_mask.to(torch.float32)

        self.process_id = process_id
        self.randomize = randomize
        # initialize indices and current idx
        self.reset()
        self.process_type = process_type
        self.sample_size = len(self)
        self.relative_weight = None

    @property
    def uid(self):
        return (self.process_type, self.process_id)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        return self.continous[idx], self.continous[idx], self.targets[idx]

    def reset(self):
        """
        Resets the index of the sampler, and randomize the indices for the next sample
        """
        self.current_idx = 0
        if self.randomize:
            self.indices = torch.randperm(len(self))
        else:
            self.indices = torch.arange(len(self))

    def sample(self,sample_from: tuple[str], number: str=None ,device=torch.device("cpu"))-> dict[torch.Tensor]:
        """
        Sample *number* of events from attributes defined by *sample_from*.
        Samples from the start again, if maximum number of samples is reached.
        Specified tensors are moved to *device* before returning.

        Args:
            sample_from (tuple[str]): Tuple of strings defining which attributes to sample from.
            number (str, optional): Number of events to sample. If None, sample_size of the process instance is used. Defaults to None.
            device (torch.device, optional): Device to which tensors are moved before returning. Defaults to torch.device("cpu").

        returns: (dict[torch.Tensor]): Dictionary with sampled tensors for each attribute defined in *sample_from*.
        """
        if self.current_idx >= len(self):
            self.reset()

        if number is None:
            number = self.sample_size

        if number > len(self):
            # when required number is too big, set it to len of self to return all data
            number = len(self)

        start_idx = self.current_idx
        # do not drop last, but instead create smaller batch
        next_idx = min(self.current_idx + number, len(self))
        idx = self.indices[start_idx:next_idx]
        self.current_idx = next_idx

        # sample events from sample_from, and normalization weight per default
        sampled_events = {attribute:getattr(self, attribute)[idx].to(device) for attribute in sample_from}
        sampled_events["weights"] = torch.full((len(idx), 1), self.total_normalization_weights / self.sample_size).to(device)
        return sampled_events

        # return self.continous[idx].to(device), self.categorical[idx].to(device), self.targets[idx].to(device)

    def create_sample_generator(self, sample_from, batch_size=None, device=torch.device("cpu")):
        """
        Creates a generator object that returns a dict of torch tensors, defined by *sample_from*.
        If *batch
        If *batch_size* is returned as None, the whole underlying data is returned at once, else only a small batch
        The generator does not randomize the underlying data. The generator and sample methods can be used
        independently.

        Args:
            batch_size (int, optional): Sample size per iteration, if None all data is returned at once. Defaults to None.
            device (torch.device, optional): Device to which tensors are moved before yielding. Defaults to torch.device("cpu").

        Yields:
            tuple (torch.Tensor): Tuple with 3 torch tensors for categorical, continous and target data of the shape [batch_size, num_features]
        """

        # save current state of the sampler to restore after full iteration
        # HINT: Do not mix sample with batch generator, as both manipulate current_idx
        start_idx, next_idx = 0, 0
        indices = torch.arange(len(self))

        if batch_size == -1:
            batch_size = len(self)

        while next_idx < len(self):
            # handle windowing of indices
            next_idx = min(start_idx + batch_size, len(self))
            idx = indices[start_idx:next_idx]
            start_idx = next_idx
            # reset if we reached the end of the dataset and drop last incomplete batch
            sampled_events = {attribute:getattr(self, attribute)[idx].to(device) for attribute in sample_from}
            sampled_events["weights"] = torch.full((len(idx), 1), self.total_normalization_weights / self.sample_size).to(device)
            yield sampled_events


class ProcessSampler(t_data.Sampler):
    signal_background_map = {"hh": "signal", "tt":"background", "dy":"background"}

    def __init__(
        self,
        batch_size: int=1,
        sample_ratio: dict[int]={"dy": 0.25, "tt": 0.25, "hh": 0.5},
        min_size: int=0,
        target_map: dict[int] = {"hh": 0, "tt": 1, "dy": 2},
        ):
        """
        Manager for Process instances to regulate sampling across multiple processes.
        Each process introduces own sampling method, which the ProcessSampler accesses.
        The result it put together as dictionary of multiple torch Tensors.
        The number of events in a tensor per sampler are determined by *batch_size*.
        The composition of the batch is determined in 2 stages:
            1. The desired number of events per process type is calculated according to *sample_ratio*. Needs to be sum to 1.0.
            2. The number of events per process id within a process type is calculated according to the total normalization weights of each process.
        To ensure that each process contributes enough events to the batch, a *min_size* per process id can be defined.
        The *target_map* maps process type to the index in the target tensor.

        Args:
            batch_size (int, optional): The number of events in a tensor per sample. Defaults to 1.
            sample_ratio (dict[int], optional): The contribution of each process type in the batch. Defaults to {"dy": 0.25, "tt": 0.25, "hh": 0.5}.
            min_size (int, optional): The minimum number of events per Process. Defaults to 0.
            target_map (dict[int], optional): Mapping from process type to target index. Defaults to {"hh": 0, "tt": 1, "dy": 2}.
        """
        self.process_inst = defaultdict(dict)  # {process_type: {process_id: dataset}}
        # statistics describing whole population of a process type
        self.total_normalization_weight_per_process_type = {"dy":0, "tt":0, "hh":0}
        self.total_product_of_weights_per_process_type = {"dy":0, "tt":0, "hh":0}

        self.batch_size = torch.tensor([batch_size], dtype=torch.int32)
        self.sample_ratio = sample_ratio
        self.min_size = min_size
        self.target_map = target_map

        # set attributes to access dataset properties directly from sampler, enabling dot notation
        self.set_attr([
            "total_normalization_weights", "relative_weight", "sample_size",
            "process_type", "create_sample_generator", "total_product_of_weights", "evaluation_space_mask",
            "continous", "categorical", "targets", "normalization_weights", "product_of_weights",
            ])

    def __getitem__(self, uid):
        return self.flat_process_dict()[uid]

    def flat_process_dict(self) -> dict[Process]:
        """
        By default process instances are nested as "process_type": {uid : process_inst}.
        This returns a flat version removing outer nesting

        Returns:
            dict[Process] : Dictionary of all process instances managed by the sampler
        """
        datasets = {}
        for _, uid_ds in self.process_inst.items():
            datasets.update(uid_ds)
        return datasets

    def apply_func_on_datasets(self, func):
        # small helper to apply function of managet datasets
        return {uid: func(ds) for uid,ds in self.flat_process_dict().items()}

    def get_attribute_of_datasets(self, attr):
        # small helper to getattributes from managet datasets
        return {uid: getattr(ds, attr) for uid,ds in self.flat_process_dict().items()}

    def events_per_dataset(self):
        return self.apply_func_on_datasets(len)

    def __len__(self):
        return sum(list(self.events_per_dataset().values()))

    def add_process_instance(self, process_inst: Process):
        """
        Adds a *process_inst* to the collection of the sampler and update statitics related to it.

        Args:
            process_inst (Process): Instance of the Dataset class
        """
        self.total_normalization_weight_per_process_type[process_inst.process_type] += process_inst.total_normalization_weights
        self.total_product_of_weights_per_process_type[process_inst.process_type] += process_inst.total_product_of_weights
        self.process_inst[process_inst.process_type][process_inst.process_id] = process_inst

    @property
    def keys(self):
        return list(self.process_inst.keys())

    def print_sample_rates(self):
        logger.info(f"Sample rate for Dataset of Era")

        for dataset_type, uid in self.process_inst.items():
            total_batch = 0
            logger.info(f"{dataset_type}:")
            for (pid), ds in uid.items():
                logger.info(f"\t{pid} | {ds.sample_size}")
                total_batch += ds.sample_size
            logger.info(f"\tsum/expected: {total_batch}/{(self.sample_ratio[ds.dataset_type] * self.batch_size).item()}")

    def calculate_sample_size(self, process_type, dry=False):
        if self.batch_size == -1:
            raise ValueError("Batch Size < 1 is not supported. Try a number big enough to be representative")

        logger.info(f"Calculate sample sizes for: {process_type}")
        # unpack and convert to tensors for easier handling, get desired_sub_batch_size
        weights, keys = [], []
        for pid, process_inst in self.process_inst[process_type].items():
            weights.append(process_inst.total_normalization_weights.item())
            keys.append(pid)
        weights = torch.tensor(weights, dtype=torch.float32)
        min_size = torch.tensor(self.min_size)
        sub_batch_size = int(self.batch_size * self.sample_ratio[process_type])

        # start algorithm
        # 2 cases: Phase phases need to be upsampeled OR phase spaces are already reasonable high
        #
        # in 3 steps:
        # 1. downsample calculated batch size and upsample to minimal threshold
        # 2. calculate relative weight of overflow for samples above minimal threshold
        # 3. distribute overflow from 2. by ceil and floor half of the sample according to median
        # 4. reduce  all steps resulting in sub_batch_size batch composition


        # preperation: calculate batch size according to weight
        total_weight_of_process_type = weights.sum()
        ideal_sizes = sub_batch_size * weights / total_weight_of_process_type

        # add relative weight to datasets
        for pid, ds in self.process_inst[process_type].items():
            ds.relative_weight = ds.total_normalization_weights / total_weight_of_process_type * self.sample_ratio[process_type]
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
        # this returns floored sizes sorted by biggest by indices
        indices_above_threshold = torch.arange(len(floored_sizes))[mask_above_threshold][floored_oveflow_idx]
        floored_sizes[indices_above_threshold] -= floored_overflow_sizes

        # conver to int but check if values changed, just to be safe
        if (floored_sizes.to(torch.int32).sum() - floored_sizes.sum()) != 0:
            raise TypeError(f"Change of type changed value")
        floored_sizes = floored_sizes.to(torch.int32)

        # if one still mismatch add it on lowest or remove from highest
        very_last_remaining = int(floored_sizes.sum()) - sub_batch_size

        # TODO BETTER SCHEME
        # remove remaining from biggest sample or add more to biggest sample

        floored_sizes[indices_above_threshold[0]] -= very_last_remaining
        # if very_last_remaining <= -1:
        #     floored_sizes[indices_above_threshold[0]] -= very_last_remaining
        # elif very_last_remaining >= 1:
        #     floored_sizes[indices_above_threshold[0]] -= very_last_remaining
        # elif very_last_remaining != 0:
        #     from IPython import embed; embed(header=f"{floored_sizes.sum()} but should be {sub_batch_size}")
        #     raise ValueError(f"Resampling failed: Created batch size of size: {floored_sizes.sum()} but should be {sub_batch_size}")

        # store batch size per phase space in dataset
        if not dry:
            for k, w in zip(keys, floored_sizes):
                self.process_inst[process_type][k].sample_size = w.item()
        else:
            logger.info({k:w.item() for k,w in zip(keys, floored_sizes)})

    def sample_batch(self, sample_from: list[str], device: torch.device=torch.device("cpu")):
        """
        This function samples a specific number of events from all datasets instances connected to the sampler.
        Which attribute is samplet is defined by a list of strings in "sample_from". By default ["continous", "categorical", "target"] is sampled.
        In the end all so created tensors are put on *device*
        Returns a dict of tensors with keys coming from *sample_from*.

        Args:
            sample_from (list[str]): List of strings defining which attributes to sample from each dataset. Defaults to ["continous", "categorical", "target"].
            device (torch.device, optional): _description_. Defaults to torch.device("cpu").

        Returns:
            dict[torch.Tensor]: Dict of torch tensor where each process instance contributed according to their sample_size attribute.
        """
        # sample sample_size number of events from different process instances
        events = {}
        for _, ds in sorted(self.flat_process_dict().items()):
            for attribute, sampled_events in ds.sample(sample_from, device=device).items():
                if attribute not in events:
                    events[attribute] = []
                events[attribute].append(sampled_events)

        # combine to single tensors
        for attribute in events.keys():
            events[attribute] = torch.concatenate(events[attribute], dim=0).to(device)
        return events

    def set_attr(self, attribute):
        if isinstance(attribute, str):
            attribute = [attribute]

        for attr in attribute:
            if hasattr(self, attr):
                raise AttributeError(f"Attribute {attr} already exists in sampler class.")

            def _accesor(*args, dataset_attr=attr, **kwargs):
                collector = {}
                for uid, process_inst in self.flat_process_dict().items():
                    process_inst_attr = getattr(process_inst, dataset_attr)
                    # when attribute is a callable function pass args and kwargs
                    if callable(process_inst_attr):
                        collector[uid] = process_inst_attr(*args, **kwargs)
                    else:
                        collector[uid] = process_inst_attr
                return collector

            setattr(self, attr, _accesor)

    def share_weights_between_sampler(self, process_sampler):
        # helper to enable ex. validation sampler to get sampling weight from training sampler
        for process_type, processes in self.process_inst.items():
            for uid, process_inst in processes.items():
                v_process_inst = process_sampler.process_inst[process_type][uid]
                v_process_inst.relative_weight = process_inst.relative_weight


def create_sampler(events, target_map, batch_size, min_size=1, train=True, sample_ratio={"dy": 0.25, "tt": 0.25, "hh": 0.5}):
    # extract data from events and wrap into Datasets
    if not events:
        logger.warning(f"Sampler is not created due to feeding empty events")
        return None

    process_sampler = ProcessSampler(batch_size=batch_size, min_size=min_size, sample_ratio=sample_ratio, target_map = target_map)
    for uid in list(events.keys()):
        (process_type, process_id) = uid
        arrays = events.pop(uid)

        # create target tensor from uid
        num_events = len(arrays["continous"])
        target_value = target_map[process_type]
        target = torch.zeros(size=(num_events, len(target_map)), dtype=torch.float32)
        target[:, target_value] = 1.
        process = Process(
            continous=arrays["continous"],
            categorical=arrays["categorical"],
            target=target,
            normalization_weights=arrays["normalization_weights"],
            product_of_weights=arrays["product_of_weights"],
            total_normalization_weights=arrays["total_normalization_weights"],
            total_product_of_weights=arrays["total_product_of_weights"],
            evaluation_space_mask=arrays["evaluation_mask"],
            process_id=process_id,
            process_type=process_type,
            randomize=True,
        )
        process_sampler.add_process_instance(process)
        logger.info(f"Add {process_type} pid: {process_id}")

    if train:
        for process_type in process_sampler.keys:
            process_sampler.calculate_sample_size(process_type=process_type)
    return process_sampler
