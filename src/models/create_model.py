from __future__ import annotations

import torch

from loss import kernel
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
        self.categorical_features = model.categorical_features
        self.continuous_features = model.continuous_features
        # for attr in self.model.mark_attributes():
        #     setattr(self, attr)

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


class ResidualDNN(torch.nn.Module):
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
        std_layer = layers.StandardizeLayer(mean=config.mean, std=config.std)

        continuous_padding = layers.PaddingLayer(padding_value=0, mask_value=utils.EMPTY_FLOAT)
        categorical_padding = layers.PaddingLayer(padding_value=config.empty_value, mask_value=utils.EMPTY_INT)

        if config.enable_rotation:
            rotation_layer = layers.RotatePhiLayer(
                columns=list(map(str, continuous_features)),
                ref_phi_columns=config.ref_phi_columns,
                rotate_columns=config.rotate_columns,
            )
        else:
            rotation_layer = None

        self.input_layer = layers.InputLayer(
            continuous_inputs=continuous_features,
            categorical_inputs=categorical_features,
            embedding_dim=10,
            expected_categorical_inputs=features.expected_embedding_inputs(),
            empty=config.empty_value,
            std_layer=std_layer,
            rotation_layer=rotation_layer,
            padding_categorical_layer=categorical_padding,
            padding_continuous_layer=continuous_padding,
        )

        self.transition_dense_1 = layers.DenseBlock(input_nodes = self.input_layer.ndim, output_nodes = config.nodes, activation_functions=config.activation_functions, eps=config.batch_norm_eps, normalize=normalize) # noqa
        self.resnet_block_1 = layers.ResNetPreactivationBlock(config.nodes, config.activation_functions, config.skip_connection_init, config.freeze_skip_connection, eps=config.batch_norm_eps, normalize=normalize)
        self.resnet_block_2 = layers.ResNetPreactivationBlock(config.nodes, config.activation_functions, config.skip_connection_init, config.freeze_skip_connection, eps=config.batch_norm_eps, normalize=normalize)
        self.resnet_block_3 = layers.ResNetPreactivationBlock(config.nodes, config.activation_functions, config.skip_connection_init, config.freeze_skip_connection, eps=config.batch_norm_eps, normalize=normalize)
        self.last_linear = torch.nn.Linear(config.nodes, 3)

    def forward(self, categorical_inputs, continuous_inputs):
        x = self.input_layer(categorical_inputs=categorical_inputs, continuous_inputs=continuous_inputs)
        x = self.transition_dense_1(x)
        x = self.resnet_block_1(x)
        x = self.resnet_block_2(x)
        x = self.resnet_block_3(x)
        x = self.last_linear(x)
        return x


class DenseNet(torch.nn.Module):
    def __init__(self, full_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # extract configs from combined dataclass config
        self.model_building_config = full_config.model_building_config
        self.dataset_config = full_config.dataset_config
        self.binning_config = full_config.binning_config

        # init layers
        self.dense_config = {
            "skip_connection_init" : self.model_building_config.skip_connection_init,
            "freeze_skip_connection" : self.model_building_config.freeze_skip_connection,
            "activation_functions" : self.model_building_config.activation_functions,
            "eps": self.model_building_config.eps_batchnorm, # increasing eps helps to stabilize training, to counter batch norm and L2 reg counter play when used together
            "normalize" : self.model_building_config.normalize_linear, # activate weight normalization on linear layer weights
        }
        self.init_layers()
        self.categorical_features = self.dataset_config.categorical_features
        self.continuous_features = self.dataset_config.continuous_features



    def init_rotation_layer(self):
        if self.model_building_config.enable_rotation:
            rotation_layer = layers.RotatePhiLayer(
                columns=list(map(str, self.dataset_config.continuous_features)),
                ref_phi_columns=self.model_building_config.ref_phi_columns,
                rotate_columns=self.model_building_config.rotate_columns,
            )
        else:
            rotation_layer = None
        return rotation_layer

    def init_standardization_layer(self):
        is_unset_mean = (self.model_building_config.mean is None)
        is_unset_std = (self.model_building_config.std is None)
        if is_unset_mean and is_unset_std:
            # if statistics are unknown create a dummy standardize layer that does nothing
            std_layer = layers.StandardizeLayer(
                mean = torch.zeros(len(self.dataset_config.continuous_features)),
                std = torch.ones(len(self.dataset_config.continuous_features))
            )
        else:
            std_layer = layers.StandardizeLayer(mean=self.model_building_config.mean, std=self.model_building_config.std)
        return std_layer

    def init_padding_layer(self):
        if self.model_building_config.continuous_padding_value is None:
            continuous_padding = None
        else:
            continuous_padding = layers.PaddingLayer(padding_value=-4, mask_value=utils.EMPTY_FLOAT)


        if self.model_building_config.categorical_padding_value is None:
            categorical_padding = None
        else:
            categorical_padding = layers.PaddingLayer(padding_value=self.model_building_config.categorical_padding_value, mask_value=utils.EMPTY_INT)
        return continuous_padding, categorical_padding

    def init_input_layer(self):
        d_cfg = self.dataset_config
        m_cfg = self.model_building_config

        std_layer = self.init_standardization_layer()
        rot_layer = self.init_rotation_layer()
        cont_pad_layer, cat_pad_layer = self.init_padding_layer()
        input_layer = layers.InputLayer(
            continuous_inputs=d_cfg.continuous_features,
            categorical_inputs=d_cfg.categorical_features,
            embedding_dim=m_cfg.embedding_dim,
            expected_categorical_inputs=m_cfg.expected_embedding_inputs,
            empty=m_cfg.categorical_padding_value,
            std_layer=std_layer,
            rotation_layer=rot_layer,
            padding_categorical_layer=cat_pad_layer,
            padding_continuous_layer=cont_pad_layer,
        )

        return input_layer

    def init_layers(self):
        m_cfg = self.model_building_config

        self.input_layer = self.init_input_layer()

        self.transition_dense_1 = layers.DenseBlock(
            input_nodes = self.input_layer.ndim,
            output_nodes = m_cfg.nodes,
            activation_functions=m_cfg.activation_functions,
            eps=self.dense_config["eps"],
            normalize=self.dense_config["normalize"],
            )
        self.dense_block_1 = layers.DenseNetBlock(input_nodes = self.transition_dense_1.output_dim, output_nodes = int((m_cfg.nodes)), **self.dense_config)
        self.dense_block_2 = layers.DenseNetBlock(input_nodes = self.dense_block_1.output_dim, output_nodes = int((m_cfg.nodes)), **self.dense_config)
        self.dense_block_3 = layers.DenseNetBlock(input_nodes = self.dense_block_2.output_dim, output_nodes = int((m_cfg.nodes)), **self.dense_config)
        self.dense_block_4 = layers.DenseNetBlock(input_nodes = self.dense_block_3.output_dim, output_nodes = int((m_cfg.nodes)), **self.dense_config)
        self.dense_block_5 = layers.DenseNetBlock(input_nodes = self.dense_block_4.output_dim, output_nodes = int((m_cfg.nodes)), **self.dense_config)
        self.last_linear = torch.nn.Linear(self.dense_block_5.output_dim, len(self.dataset_config.target_map.keys()))

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

class LBNDenseNet(DenseNet):
    def __init__(self, full_config, *args, **kwargs):
        # has same init as DenseNet
        super().__init__(full_config, *args, **kwargs)

    def init_layers(self):
        # create normal DenseNet but also LBN
        super().init_layers()
        # old transition layer needs combined input dimension lbn and input_layer
        self.lbn = layers.LBN_DNN(
            continuous_features = self.dataset_config.continuous_features,
            M  = self.model_building_config.LBN_M,
            weight_init_scale = 1.0,
            clip_weights = False,
            eps = 1.0e-5,
        )

        self.transition_dense_1 = layers.DenseBlock(
            input_nodes = (self.input_layer.ndim + self.lbn.ndim),
            output_nodes = self.model_building_config.nodes,
            activation_functions=self.model_building_config.activation_functions,
            eps=self.model_building_config.eps_batchnorm,
            normalize=self.model_building_config.normalize_linear
            )


    def forward(self, categorical_inputs, continuous_inputs):
        # preprocessing with lbn
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


class BinnedLBNDenseNetV2(LBNDenseNet):
    def __init__(self, full_config, *args, **kwargs):
        super().__init__(
            self.dataset_config,
            self.model_building_config,
            *args,
            **kwargs
            )

        self.config = config
        self.binning_config = binning_config

    def init_layers(self, dataset_config, config):
        # create normal LBN DenseNet
        super().init_layers(
            dataset_config.continuous_features,
            dataset_config.categorical_features,
            config
            )

        # init binning layer
        # add binning layer if necessary
        if self.binning_config is None:
            self.binning_layer = torch.nn.Identity()
        else:
            from IPython import embed; mbed(header="MESSAGE Line 443 | File: create_model.py")
            self.binning_layer = layers.BinningLayerRight(
                num_bins=binning_config.num_bins,
                lower_bound=binning_config.lower_edge,
                upper_bound=binning_config.upper_edge,
                binning_fn=binning_config.binning_fn,
                kernel_cls=getattr(kernel, binning_config.kernel_cls),
                kernel_cfg=binning_config.kernel_config[binning_config.kernel_cls],
                )


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
        if config.mean is None and config.std is None:
            std_layer = layers.StandardizeLayer(
                mean = torch.zeros(len(continuous_features)),
                std = torch.ones(len(continuous_features))
            )
        else:
            std_layer = layers.StandardizeLayer(mean=config.mean, std=config.std)


        if config.continuous_padding_value is None:
            continuous_padding = None
        else:
            continuous_padding = layers.PaddingLayer(padding_value=-4, mask_value=utils.EMPTY_FLOAT)


        if config.categorical_padding_value is None:
            categorical_padding = None
        else:
            categorical_padding = layers.PaddingLayer(padding_value=config.categorical_padding_value, mask_value=utils.EMPTY_INT)

        if config.enable_rotation:
            rotation_layer = layers.RotatePhiLayer(
                columns=list(map(str, continuous_features)),
                ref_phi_columns=config.ref_phi_columns,
                rotate_columns=config.rotate_columns,
            )
        else:
            rotation_layer = None

        self.input_layer = layers.InputLayer(
            continuous_inputs=continuous_features,
            categorical_inputs=categorical_features,
            embedding_dim=10,
            expected_categorical_inputs=features.expected_embedding_inputs(),
            empty=config.categorical_padding_value,
            std_layer=std_layer,
            rotation_layer=rotation_layer,
            padding_categorical_layer=categorical_padding,
            padding_continuous_layer=continuous_padding,
        )

        self.lbn = layers.LBN_DNN(
            continuous_features = continuous_features,
            M  = config.LBN_M,
            weight_init_scale = 1.0,
            clip_weights = False,
            eps = 1.0e-5,
        )

        dense_config = {
            "skip_connection_init":config.skip_connection_init,
            "freeze_skip_connection":config.freeze_skip_connection,
            "activation_functions":"ELU",
            "eps":eps,
            "normalize":False,
        }

        self.transition_dense_1 = layers.DenseBlock(input_nodes = self.input_layer.ndim + self.lbn.ndim, output_nodes = config.nodes, activation_functions=config.activation_functions, eps=eps, normalize=normalize) # noqa
        self.dense_block_1 = layers.DenseNetBlock(input_nodes = self.transition_dense_1.output_dim, output_nodes = int((config.nodes)), **dense_config)
        self.dense_block_2 = layers.DenseNetBlock(input_nodes = self.dense_block_1.output_dim, output_nodes = int((config.nodes)), **dense_config)
        self.dense_block_3 = layers.DenseNetBlock(input_nodes = self.dense_block_2.output_dim, output_nodes = int((config.nodes)), **dense_config)
        self.dense_block_4 = layers.DenseNetBlock(input_nodes = self.dense_block_3.output_dim, output_nodes = int((config.nodes)), **dense_config)
        self.dense_block_5 = layers.DenseNetBlock(input_nodes = self.dense_block_4.output_dim, output_nodes = int((config.nodes)), **dense_config)
        self.last_linear = torch.nn.Linear(self.dense_block_5.output_dim, 3)

        # pick correct last activaton function
        last_activation_fn =  getattr(torch.nn.modules.activation, config.last_activation_fn) if config.last_activation_fn else torch.nn.Identity()
        if config.last_activation_fn == "Softmax":
            self.last_activation = last_activation_fn(dim=1)
        else:
            self.last_activation = self.last_activation()

        # add binning layer if necessary
        if binning_config is None:
            self.binning_layer = torch.nn.Identity()
        else:
            from IPython import embed; mbed(header="MESSAGE Line 443 | File: create_model.py")
            self.binning_layer = layers.BinningLayerRight(
                num_bins=binning_config.num_bins,
                lower_bound=binning_config.lower_edge,
                upper_bound=binning_config.upper_edge,
                binning_fn=binning_config.binning_fn,
                kernel_cls=getattr(kernel, binning_config.kernel_cls),
                kernel_cfg=binning_config.kernel_config[binning_config.kernel_cls],
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
