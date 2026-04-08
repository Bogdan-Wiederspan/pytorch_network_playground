import torch

import kernel
from models import layers
from data import features
from utils import utils

#TODO shared memory between classes that are connected together! That should AUX define.
class Aux():
    x = {}
    def __init__(self):
        pass

    def set_aux(self, *args):
        cls_name = self.__class__.__name__
        self.x


class BNet(torch.nn.Module):
    def __init__(self, continuous_features, categorical_features, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_layers(config=config, continuous_features=continuous_features, categorical_features=categorical_features)

    def init_layers(self, continuous_features, categorical_features, config):
        # increasing eps helps to stabilize training to counter batch norm and L2 reg counterplay when used together.
        # eps = 0.5e-5

        # activate weight normalization on linear layer weights
        normalize = False

        # helper where all layers are defined
        # std layers are filled when statitics are known
        std_layer = layers.StandardizeLayer(mean=config["mean"], std=config["std"])

        continuous_padding = layers.PaddingLayer(padding_value=0, mask_value=utils.EMPTY_FLOAT)
        categorical_padding = layers.PaddingLayer(padding_value=config["empty_value"], mask_value=utils.EMPTY_INT)

        if config["enable_rotation"]:
            rotation_layer = layers.RotatePhiLayer(
                columns=list(map(str, continuous_features)),
                ref_phi_columns=config["ref_phi_columns"],
                rotate_columns=config["rotate_columns"],
            )
        else:
            rotation_layer = None

        self.input_layer = layers.InputLayer(
            continuous_inputs=continuous_features,
            categorical_inputs=categorical_features,
            embedding_dim=10,
            expected_categorical_inputs=features.expected_embedding_inputs(),
            empty=config["empty_value"],
            std_layer=std_layer,
            rotation_layer=rotation_layer,
            padding_categorical_layer=categorical_padding,
            padding_continuous_layer=continuous_padding,
        )

        self.transition_dense_1 = layers.DenseBlock(input_nodes = self.input_layer.ndim, output_nodes = config["nodes"], activation_functions=config["activation_functions"], eps=config["batch_norm_eps"], normalize=normalize) # noqa
        self.resnet_block_1 = layers.ResNetPreactivationBlock(config["nodes"], config["activation_functions"], config["skip_connection_init"], config["freeze_skip_connection"], eps=config["batch_norm_eps"], normalize=normalize)
        self.resnet_block_2 = layers.ResNetPreactivationBlock(config["nodes"], config["activation_functions"], config["skip_connection_init"], config["freeze_skip_connection"], eps=config["batch_norm_eps"], normalize=normalize)
        self.resnet_block_3 = layers.ResNetPreactivationBlock(config["nodes"], config["activation_functions"], config["skip_connection_init"], config["freeze_skip_connection"], eps=config["batch_norm_eps"], normalize=normalize)
        self.last_linear = torch.nn.Linear(config["nodes"], 3)

    def forward(self, categorical_inputs, continuous_inputs):
        x = self.input_layer(categorical_inputs=categorical_inputs, continuous_inputs=continuous_inputs)
        x = self.transition_dense_1(x)
        x = self.resnet_block_1(x)
        x = self.resnet_block_2(x)
        x = self.resnet_block_3(x)
        x = self.last_linear(x)
        return x


class BNetDenseNet(torch.nn.Module):
    def __init__(self, continuous_features, categorical_features, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_layers(config=config, continuous_features=continuous_features, categorical_features=categorical_features)
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features

    def init_layers(self, continuous_features, categorical_features, config):
        # increasing eps helps to stabilize training to counter batch norm and L2 reg counterplay when used together.
        eps = 0.001
        # activate weight normalization on linear layer weights
        normalize = False

        # helper where all layers are defined
        # std layers are filled when statitics are known
        if config["mean"] is None and config["std"] is None:
            std_layer = layers.StandardizeLayer(
                mean = torch.zeros(len(continuous_features)),
                std = torch.ones(len(continuous_features))
            )
        else:
            std_layer = layers.StandardizeLayer(mean=config["mean"], std=config["std"])


        if config["continuous_padding_value"] is None:
            continuous_padding = None
        else:
            continuous_padding = layers.PaddingLayer(padding_value=-4, mask_value=utils.EMPTY_FLOAT)


        if config["categorical_padding_value"] is None:
            categorical_padding = None
        else:
            categorical_padding = layers.PaddingLayer(padding_value=config["categorical_padding_value"], mask_value=utils.EMPTY_INT)


        if config["enable_rotation"]:
            rotation_layer = layers.RotatePhiLayer(
                columns=list(map(str, continuous_features)),
                ref_phi_columns=config["ref_phi_columns"],
                rotate_columns=config["rotate_columns"],
            )
        else:
            rotation_layer = None

        self.input_layer = layers.InputLayer(
            continuous_inputs=continuous_features,
            categorical_inputs=categorical_features,
            embedding_dim=10,
            expected_categorical_inputs=features.expected_embedding_inputs(),
            empty=config["categorical_padding_value"],
            std_layer=std_layer,
            rotation_layer=rotation_layer,
            padding_categorical_layer=categorical_padding,
            padding_continuous_layer=continuous_padding,
        )

        dense_config = {
            "skip_connection_init":config["skip_connection_init"],
            "freeze_skip_connection":config["freeze_skip_connection"],
            "activation_functions":"ELU",
            "eps":eps,
            "normalize":False,
        }

        self.transition_dense_1 = layers.DenseBlock(input_nodes = self.input_layer.ndim, output_nodes = config["nodes"], activation_functions=config["activation_functions"], eps=eps, normalize=normalize) # noqa
        self.dense_block_1 = layers.DenseNetBlock(input_nodes = self.transition_dense_1.output_dim, output_nodes = int((config["nodes"])), **dense_config)
        self.dense_block_2 = layers.DenseNetBlock(input_nodes = self.dense_block_1.output_dim, output_nodes = int((config["nodes"])), **dense_config)
        self.dense_block_3 = layers.DenseNetBlock(input_nodes = self.dense_block_2.output_dim, output_nodes = int((config["nodes"])), **dense_config)
        self.dense_block_4 = layers.DenseNetBlock(input_nodes = self.dense_block_3.output_dim, output_nodes = int((config["nodes"])), **dense_config)
        self.dense_block_5 = layers.DenseNetBlock(input_nodes = self.dense_block_4.output_dim, output_nodes = int((config["nodes"])), **dense_config)
        self.last_linear = torch.nn.Linear(self.dense_block_5.output_dim, 3)

    def forward(self, categorical_inputs, continuous_inputs):
        x = self.input_layer(categorical_inputs=categorical_inputs, continuous_inputs=continuous_inputs)
        x = self.transition_dense_1(x)
        x = self.dense_block_1(x)
        x = self.dense_block_2(x)
        x = self.dense_block_3(x)
        x = self.dense_block_4(x)
        x = self.dense_block_5(x)
        x = self.last_linear(x)
        return x

class BNetLBNDenseNet(torch.nn.Module):
    def __init__(self, continuous_features, categorical_features, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_layers(config=config, continuous_features=continuous_features, categorical_features=categorical_features)
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features

    # def mark_attributes(self):
    #     return {attr:getattr(self, attr) for attr in [
    #         "categorical_features", "continuous_features"
    #         ]}

    def init_layers(self, continuous_features, categorical_features, config):
        # increasing eps helps to stabilize training to counter batch norm and L2 reg counterplay when used together.
        eps = 0.001
        # activate weight normalization on linear layer weights
        normalize = False

        # helper where all layers are defined
        # std layers are filled when statitics are known
        if config["mean"] is None and config["std"] is None:
            std_layer = layers.StandardizeLayer(
                mean = torch.zeros(len(continuous_features)),
                std = torch.ones(len(continuous_features))
            )
        else:
            std_layer = layers.StandardizeLayer(mean=config["mean"], std=config["std"])


        if config["continuous_padding_value"] is None:
            continuous_padding = None
        else:
            continuous_padding = layers.PaddingLayer(padding_value=-4, mask_value=utils.EMPTY_FLOAT)


        if config["categorical_padding_value"] is None:
            categorical_padding = None
        else:
            categorical_padding = layers.PaddingLayer(padding_value=config["categorical_padding_value"], mask_value=utils.EMPTY_INT)

        if config["enable_rotation"]:
            rotation_layer = layers.RotatePhiLayer(
                columns=list(map(str, continuous_features)),
                ref_phi_columns=config["ref_phi_columns"],
                rotate_columns=config["rotate_columns"],
            )
        else:
            rotation_layer = None

        self.input_layer = layers.InputLayer(
            continuous_inputs=continuous_features,
            categorical_inputs=categorical_features,
            embedding_dim=10,
            expected_categorical_inputs=features.expected_embedding_inputs(),
            empty=config["categorical_padding_value"],
            std_layer=std_layer,
            rotation_layer=rotation_layer,
            padding_categorical_layer=categorical_padding,
            padding_continuous_layer=continuous_padding,
        )

        self.lbn = layers.LBN_DNN(
            continuous_features = continuous_features,
            M  = config["LBN_M"],
            weight_init_scale = 1.0,
            clip_weights = False,
            eps = 1.0e-5,
        )

        dense_config = {
            "skip_connection_init":config["skip_connection_init"],
            "freeze_skip_connection":config["freeze_skip_connection"],
            "activation_functions":"ELU",
            "eps":eps,
            "normalize":False,
        }

        self.transition_dense_1 = layers.DenseBlock(input_nodes = self.input_layer.ndim + self.lbn.ndim, output_nodes = config["nodes"], activation_functions=config["activation_functions"], eps=eps, normalize=normalize) # noqa
        self.dense_block_1 = layers.DenseNetBlock(input_nodes = self.transition_dense_1.output_dim, output_nodes = int((config["nodes"])), **dense_config)
        self.dense_block_2 = layers.DenseNetBlock(input_nodes = self.dense_block_1.output_dim, output_nodes = int((config["nodes"])), **dense_config)
        self.dense_block_3 = layers.DenseNetBlock(input_nodes = self.dense_block_2.output_dim, output_nodes = int((config["nodes"])), **dense_config)
        self.dense_block_4 = layers.DenseNetBlock(input_nodes = self.dense_block_3.output_dim, output_nodes = int((config["nodes"])), **dense_config)
        self.dense_block_5 = layers.DenseNetBlock(input_nodes = self.dense_block_4.output_dim, output_nodes = int((config["nodes"])), **dense_config)
        self.last_linear = torch.nn.Linear(self.dense_block_5.output_dim, 3)

    def forward(self, categorical_inputs, continuous_inputs):
        # preprocessing
        x = torch.concatenate(
            (self.input_layer(categorical_inputs=categorical_inputs, continuous_inputs=continuous_inputs),
            self.lbn(continuous_inputs)),
            axis=1
        )

        # dnn
        x = self.transition_dense_1(x)
        x = self.dense_block_1(x)
        x = self.dense_block_2(x)
        x = self.dense_block_3(x)
        x = self.dense_block_4(x)
        x = self.dense_block_5(x)
        x = self.last_linear(x)
        return x

class AddBinning(torch.nn.Module):
    def __init__(self, model, kernel_cls, kernel_cfg, binning_edges, signal_cls=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unbinned_model = model
        self.binning_layer = layers.BinningLayer(
            init_edges=binning_edges,
            kernel_cls=kernel_cls,
            kernel_cfg=kernel_cfg,
            )
        self.signal_cls = ... if signal_cls is None else signal_cls # used for class slicing
        # for attr in self.model.mark_attributes():
        #     setattr(self, attr)

    def forward(self, categorical_inputs, continuous_inputs):
        x = self.unbinned_model(categorical_inputs, continuous_inputs)[:, self.signal_cls] # normal prediction as probabilities
        x = self.binning_layer(x)
        return x

class AddActFnToModel(torch.nn.Module):
    def __init__(self, model, act_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        # self.categorical_features = model.categorical_features
        # self.continuous_features = model.continuous_features

        self.act_func = self._get_attr(torch.nn.modules.activation, act_fn)(dim=1)

    def _get_attr(self, obj, attr):
        for o in dir(obj):
            if o.lower() == attr.lower():
                return getattr(obj, o)
        else:
            raise AttributeError(f"Object has no attribute '{attr}'")

    def forward(self, categorical_inputs, continuous_inputs):
        x = self.model(categorical_inputs, continuous_inputs)
        x = self.act_func(x)
        return x

# TODO finished calib layer
class AddCalibrationToModel(torch.nn.Module):
    def __init__(self, model, act_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.categorical_features = model.categorical_features
        self.continuous_features = model.continuous_features
        self.calibration_layer = layers.TemperaturCalibrationLayer()


    def calibrate(self, sampler, num_samples):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.calibration_layer.temperature], lr=0.001, max_iter=10000, line_search_fn='strong_wolfe')

        # for i in range(num_samples):
        #     sampler.get
        # # Create tensors
        # logits_list = torch.cat(logits_list).to(device)
        # labels_list = torch.cat(labels_list).to(device)

        # def _eval():
        # loss = criterion(T_scaling(logits_list, temperature), labels_list)
        # loss.backward()
        # return loss

        # optimizer.step(_eval)

    def forward(self, categorical_inputs, continuous_inputs):
        x = self.model(categorical_inputs, continuous_inputs)
        x = self.act_func(x)
        return x


class BinnedLBNDenseNet(torch.nn.Module):
    def __init__(
        self,
        continuous_features,
        categorical_features,
        config,
        binning_config=None,
        *args,
        **kwargs):
        super().__init__(*args, **kwargs)
        self.init_layers(
            config=config,
            continuous_features=continuous_features,
            categorical_features=categorical_features,
            binning_config=binning_config
            )
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.config = config
        self.binning_config = binning_config

    def init_layers(self, continuous_features, categorical_features, config, binning_config):
        # increasing eps helps to stabilize training to counter batch norm and L2 reg counterplay when used together.
        eps = 0.001
        # activate weight normalization on linear layer weights
        normalize = False

        # helper where all layers are defined
        # std layers are filled when statitics are known
        if config["mean"] is None and config["std"] is None:
            std_layer = layers.StandardizeLayer(
                mean = torch.zeros(len(continuous_features)),
                std = torch.ones(len(continuous_features))
            )
        else:
            std_layer = layers.StandardizeLayer(mean=config["mean"], std=config["std"])


        if config["continuous_padding_value"] is None:
            continuous_padding = None
        else:
            continuous_padding = layers.PaddingLayer(padding_value=-4, mask_value=utils.EMPTY_FLOAT)


        if config["categorical_padding_value"] is None:
            categorical_padding = None
        else:
            categorical_padding = layers.PaddingLayer(padding_value=config["categorical_padding_value"], mask_value=utils.EMPTY_INT)

        if config["enable_rotation"]:
            rotation_layer = layers.RotatePhiLayer(
                columns=list(map(str, continuous_features)),
                ref_phi_columns=config["ref_phi_columns"],
                rotate_columns=config["rotate_columns"],
            )
        else:
            rotation_layer = None

        self.input_layer = layers.InputLayer(
            continuous_inputs=continuous_features,
            categorical_inputs=categorical_features,
            embedding_dim=10,
            expected_categorical_inputs=features.expected_embedding_inputs(),
            empty=config["categorical_padding_value"],
            std_layer=std_layer,
            rotation_layer=rotation_layer,
            padding_categorical_layer=categorical_padding,
            padding_continuous_layer=continuous_padding,
        )

        self.lbn = layers.LBN_DNN(
            continuous_features = continuous_features,
            M  = config["LBN_M"],
            weight_init_scale = 1.0,
            clip_weights = False,
            eps = 1.0e-5,
        )

        dense_config = {
            "skip_connection_init":config["skip_connection_init"],
            "freeze_skip_connection":config["freeze_skip_connection"],
            "activation_functions":"ELU",
            "eps":eps,
            "normalize":False,
        }

        self.transition_dense_1 = layers.DenseBlock(input_nodes = self.input_layer.ndim + self.lbn.ndim, output_nodes = config["nodes"], activation_functions=config["activation_functions"], eps=eps, normalize=normalize) # noqa
        self.dense_block_1 = layers.DenseNetBlock(input_nodes = self.transition_dense_1.output_dim, output_nodes = int((config["nodes"])), **dense_config)
        self.dense_block_2 = layers.DenseNetBlock(input_nodes = self.dense_block_1.output_dim, output_nodes = int((config["nodes"])), **dense_config)
        self.dense_block_3 = layers.DenseNetBlock(input_nodes = self.dense_block_2.output_dim, output_nodes = int((config["nodes"])), **dense_config)
        self.dense_block_4 = layers.DenseNetBlock(input_nodes = self.dense_block_3.output_dim, output_nodes = int((config["nodes"])), **dense_config)
        self.dense_block_5 = layers.DenseNetBlock(input_nodes = self.dense_block_4.output_dim, output_nodes = int((config["nodes"])), **dense_config)
        self.last_linear = torch.nn.Linear(self.dense_block_5.output_dim, 3)

        # pick correct last activaton function
        last_activation_fn =  getattr(torch.nn.modules.activation, config["last_activation_fn"]) if config["last_activation_fn"] else torch.nn.Identity()
        if config["last_activation_fn"] == "Softmax":
            self.last_activation = last_activation_fn(dim=1)
        else:
            self.last_activation = self.last_activation()

        # add binning layer if necessary
        if binning_config is None:
            self.binning_layer = torch.nn.Identity()
        else:
            self.binning_layer = layers.BinningLayer(
                init_edges=binning_config["init_edges"],
                kernel_cls=getattr(kernel, binning_config["kernel_cls"]),
                kernel_cfg=binning_config["kernel_config"][binning_config["kernel_cls"]],
                )

    def set_learning_mode(self, mode):
        # freeze everything and set depending on mode specific layer on trainable
        for name, layer in self.named_children():
            layer.requires_grad = False

        if mode == "bin_mode":
            # only binning is trained
            all_layers = dict(self.named_children())
            all_layers["binning_layer"].requires_grad = True
        elif mode == "default":
            # everything that is not binning is trained
            all_layers_except_binning = dict(self.named_children())
            all_layers_except_binning.pop("binning_layer")
            for name, layer in all_layers_except_binning.items():
                layer.requires_grad = True
        elif mode == "train_all":
            # every weight is trained
            for name, layer in self.named_children():
                layer.requires_grad = True

    def forward(self, categorical_inputs, continuous_inputs, **kwargs):
        # preprocessing
        x = torch.concatenate(
            (self.input_layer(categorical_inputs=categorical_inputs, continuous_inputs=continuous_inputs),
            self.lbn(continuous_inputs)),
            axis=1
        )

        # dnn
        x = self.transition_dense_1(x)
        x = self.dense_block_1(x)
        x = self.dense_block_2(x)
        x = self.dense_block_3(x)
        x = self.dense_block_4(x)
        x = self.dense_block_5(x)
        x = self.last_linear(x)

        # probability or identity
        x = self.last_activation(x)
        if kwargs.get("without_bin"):
            return x
        # binning or identity
        x = self.binning_layer(x)
        return x
