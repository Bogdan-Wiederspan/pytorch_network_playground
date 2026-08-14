def load_data_old(datasets, columns: Union[list[str],str, None]=None, cuts: Union[list[str], None]=None):
    """
    Loads data with given *file_type* in given a *dataset_pattern* and *year_pattern*. If only certain columns are needed, they can be specified in *columns*.
    The data sorted by year and dataset name is returned as a nested dictionary in awkward format.

    Args:
        dataset_patter (str): pattern describing dataset e.g. "dy_*" -> all drell-yan datasets
        columns (list[str], str, optional): columns that should be lodead e.g. ["events", "run"]. If None loads all columns . Defaults to None.
        cuts (list[str]): list of cuts to be applied on top of baseline cut, which are defined in root_to_numpy. Defaults to None.

    Returns:
        dict: {year:{pid: List(Ids)}}
    """
    data = load_data_per_process_id_old(datasets, branches=list(columns), cut=cuts)
    data = merge_per_pid(data)
    return data

def load_data_per_process_id_old(dataset_paths, branches, cut=None):
    # helper to load root data into a dictionary of form:
    # {year:{dataset : array}}, where array is a structured array
    def sort_by_process_id(array):
        # helper to extract data by process id and group them by this
        pids = array["process_id"]
        unique_ids = np.unique(pids)
        p_array = {}
        for uid in unique_ids:
            mask = pids == uid
            p_array[int(uid)] = array[mask]
        return tuple(p_array.items())

    data = {}
    num_of_datasets = len(dataset_paths.keys())
    for current_dataset_num, (dataset, files) in enumerate(dataset_paths.items()):
        logger_inst.info(f"Dataset: {dataset}, {current_dataset_num}/{num_of_datasets}")
        events = root_to_numpy(files, branches=branches, cut=cut)
        for file_array in events:
            p_arrays = sort_by_process_id(file_array)
            for pid, p_array in p_arrays:
                uid = (dataset[:2],pid)
                if uid not in data:
                    data[uid] = []
                data[uid].append(p_array)
                logger_inst.debug(f"{dataset} | PID: {pid} | {len(p_array)}")
    return data



def merge_per_pid(data):
    # helper to merge all process_ids together to continuous array
    logger_inst.info("Start merging arrays per process id")
    keys = list(data.keys())

    for i, uid in enumerate(keys):
        logger_inst.debug(f"{i}/{len(keys)}: {uid}")
        arrays = data.pop(uid)
        concat = np.concatenate(arrays, axis=0)
        data[uid] = concat
    return data
