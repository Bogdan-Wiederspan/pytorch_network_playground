import torch
from utils import utils
from models import layers, create_model
from data import features
from data.load_data import get_data, find_datasets, create_sampler, get_batch_statistics


CPU = torch.device("cpu")
CUDA = torch.device("cuda")
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# load data
# eras = ["22pre", "22post", "23pre", "23post"] if not debugging else ["22pre"]
eras = ["22pre", "22post"]
datasets = find_datasets(["dy_*","tt_*", "hh_ggf_hbb_htt_kl0_kt1*"], eras, "root")
debugging = False

dataset_config = {
    "min_events":3,
    "continous_features" : features.continous_features if not debugging else features.continous_features[:2],
    "categorical_features": features.categorical_features if not debugging else features.categorical_features[:2],
    "eras" : eras,
    "datasets" : datasets,
}

# config of network
layer_config = {
    "ref_phi_columns": ("res_dnn_pnet_vis_tau1", "res_dnn_pnet_vis_tau2"),
    "rotate_columns": ("res_dnn_pnet_bjet1", "res_dnn_pnet_bjet2", "res_dnn_pnet_fatjet", "res_dnn_pnet_vis_tau1", "res_dnn_pnet_vis_tau2"),
    "empty_value": 15,
    "nodes": 128,
    "activation_functions": "PReLU",
    "skip_connection_init": 1,
    "freeze_skip_connection": False,
}

config = {

    "lr":1e-2,
    "gamma":0.9,
    "label_smoothing":0,
}

# load data in format {pid : torch}
# print("LOAD CACHE")
# p = "/data/dust/user/wiedersb/HH_DNN/cache/test_cache"
# import pickle
# with open(p, "rb") as file:
#     events = pickle.load(file)
# from IPython import embed; embed(header="AFTER - 44 in train.py ")


events = get_data(dataset_config, cache_path=None, overwrite=True)
layer_config["mean"],layer_config["std"] = get_batch_statistics(events, padding_value=-99999)

sampler = create_sampler(
    events,
    min_size=3,
)



# training loop:
# TODO: use split of parameters
models_input_layer, model = create_model.init_layers(dataset_config["continous_features"], dataset_config["categorical_features"], config=layer_config)
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config["gamma"])
model = model.to(DEVICE)
# HINT: requires only logits, no softmax at end
loss_fn = torch.nn.CrossEntropyLoss(weight=None, size_average=None,label_smoothing=config["label_smoothing"])


    # loss = loss_fn(pred, target)
    # from IPython import embed; embed(header="string - 614 in bognet.py ")
    # loss.backward()
    # optimizer.first_step(zero_grad=True)

    # # second forward step with disabled bachnorm running stats in second forward step
    # disable_running_stats(model)
    # pred_2 = self(categorical_x, continous_x)
    # loss_fn(pred_2, target).backward()
    # optimizer.second_step(zero_grad=True)
max_iteration = 100
LOG_INTERVAL = 10
model.train()
running_loss = 0.0

#### PREPARE DATA

# sampler = prepare_data.prepare_data(
#     find_datasets(dataset_config["dataset_pattern"], dataset_config["eras"], "root"),
#     dataset_config["continous_features"] + dataset_config["categorical_features"],
#     dtype=torch.float32,
#     file_type="root",
#     min_size=dataset_config["min_events"],
# )

for iteration in range(max_iteration):
    # from IPython import embed; embed(header="string - 65 in train.py ")
    optimizer.zero_grad()

    cont, cat, targets = sampler.get_batch()
    cont, cat, targets = cont.to(DEVICE), cat.to(DEVICE), targets.to(DEVICE)
    pred = model((cat,cont))

    loss = loss_fn(pred, targets.reshape(-1,3))
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    if iteration % LOG_INTERVAL == 0:
        print(f"Step {iteration} Loss: {loss.item():.4f}")

from IPython import embed; embed(header="END - 89 in train.py ")

def torch_export(model, dst_path, input_tensors):
    from pathlib import Path
    model = model.eval()

    categorical_input, continous_inputs = input_tensors

    # HINT: input is chosen since model takes a tuple of inputs, normally name of inputs is used
    dim = torch.export.dynamic_shapes.Dim.AUTO
    dynamic_shapes = {
        "input": ((dim, categorical_input.shape[-1]), (dim, continous_inputs.shape[-1]))
    }

    exp = torch.export.export(
        model,
        args=((categorical_input, continous_inputs),),
        dynamic_shapes=dynamic_shapes,
    )

    p = Path(f"{dst_path}").with_suffix(".pt2")
    torch.export.save(exp, p)


def run_exported_tensor_model(pt2_path, input_tensors):
    exp = torch.export.load(pt2_path)
    scores = exp.module()(input_tensors)
    return scores
