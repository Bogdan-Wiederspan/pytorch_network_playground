from __future__ import annotations

import torch

from loss import kernel
from models import layers
from data import features
from utils import utils

MODEL_REGISTRY = {}

def register_model(name):
    def wrapper(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return wrapper


class BaseModel(torch.nn.Module):
    LEARNING_MODES = {}
    def __init__(self, full_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_building_config = full_config.model_building_config
        self.dataset_config = full_config.dataset_config
        self.binning_config = full_config.binning_config

        self.categorical_features = self.dataset_config.categorical_features
        self.continuous_features = self.dataset_config.continuous_features

        # use the last activation function when it exist. Can be deactivated
        self.use_last_activation = self.model_building_config.use_last_activation
        self.is_binned = False

    def init_layers(self):
        raise NotImplementedError("init_layers needs to be implemented in child class, where all layers of the model are defined")

    @property
    def is_parametrized(self) -> bool:
        """
        If any layer is parametrized, the whole model is considered parametrized.
        Parametrization can break some operations, like serialization, e.g. torch.save(model_instance).

        Returns:
            bool: Boolean that indicates if the model is parametrized, which is the case if any layer is parametrized.
        """
        for name, module in self.named_modules():
            if torch.nn.utils.parametrize.is_parametrized(module):
                return True
        return False

    def set_last_fn_mode(self, on=True):
        """
        Activate/Deactivate the last activation function. Use full if one wants the logits.

        Args:
            on (bool, optional): Bool to determine if one uses the last activation function. Defaults to True.
        """
        self.use_last_activation = on

    def init_rotation_layer(self) -> torch.nn.Module | None:
        """
        Helper to initialize a rotation layer instance. This layer rotates given columns relative to other columns, given by name.
        If rotation is not enabled in the model building config, None is returned.

        Returns:
            torch.nn.Module: Rotation layer instance or None if rotation is not enabled in the model building config.
        """
        if self.model_building_config.enable_rotation:
            rotation_layer = layers.RotatePhiLayer(
                columns=list(map(str, self.dataset_config.continuous_features)),
                ref_phi_columns=self.model_building_config.ref_phi_columns,
                rotate_columns=self.model_building_config.rotate_columns,
            )
        else:
            rotation_layer = None
        return rotation_layer

    def init_standardization_layer(self)-> torch.nn.Module:
        """
        Helper to initialize a Standardization Layer instance with mean and std as buffer.
        Mean and STD are fixed values that need to be set before the training.
        If mean and std in model building config are set to None, an Identity transformation is returned.

        Returns:
            torch.nn.Module: Standardization layer instance with mean and std as buffer.
        """
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

    def init_padding_layer(self)-> tuple[torch.nn.Module | None, torch.nn.Module | None]:
        """
        Helper function to initialize two padding layers, one to pad continous values and the other for categorical information.
        The actual padding value is defined in the model building config, if the value is None, no padding layer is created.

        Returns:
            tuple[torch.nn.Module | None, torch.nn.Module | None]: None or tuple of PaddingLayer instances.
        """
        if self.model_building_config.continuous_padding_value is None:
            continuous_padding = None
        else:
            continuous_padding = layers.PaddingLayer(padding_value=-4, mask_value=utils.EMPTY_FLOAT)

        if self.model_building_config.categorical_padding_value is None:
            categorical_padding = None
        else:
            categorical_padding = layers.PaddingLayer(padding_value=self.model_building_config.categorical_padding_value, mask_value=utils.EMPTY_INT)
        return continuous_padding, categorical_padding

    def init_input_layer(self) -> torch.nn.Module:
        """
        Helper function to initialize the input layer, which consist of a preprocessing pipeline.
        This preprocessing pipeline consist of standardization, rotation and padding layer, which are initialized with the corresponding helper functions.

        Returns:
            torch.nn.Module: input layer instance with preprocessing pipeline
        """
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

    def init_cat_embedding_layer(self) -> torch.nn.Module:
        embedding_layer = layers.CatEmbeddingLayer(
            embedding_dim = self.model_building_config.embedding_dim,
            categories = self.dataset_config.categorical_features,
            expected_categorical_inputs = self.model_building_config.expected_embedding_inputs,
            empty = self.model_building_config.categorical_padding_value,
            # category_dims = self.model_building_config.category_dims,
            )
        return embedding_layer

    def init_optional_input_layer(self) -> torch.nn.Module:
        """
        Helper function to initialize the input layer, which consist of a preprocessing pipeline.
        This preprocessing pipeline consist of standardization, rotation and padding layer, which are initialized with the corresponding helper functions.

        Returns:
            torch.nn.Module: input layer instance with preprocessing pipeline
        """
        d_cfg = self.dataset_config
        m_cfg = self.model_building_config

        cont_pad_layer, cat_pad_layer = self.init_padding_layer()
        embedding_layer = self.init_cat_embedding_layer()
        std_layer = self.init_standardization_layer()
        rot_layer = self.init_rotation_layer()

        if d_cfg.continuous_features:
            cont_input_layer = layers.ContinuousInputLayer(
                continuous_inputs=d_cfg.continuous_features,
                empty=m_cfg.continuous_padding_value,
                rotation_layer=rot_layer,
                std_layer=std_layer,
                padding_continuous_layer=cont_pad_layer,
            )
        else:
            cont_input_layer = layers.EmptyLayer()

        if d_cfg.categorical_features:
            cat_input_layer = layers.CategoricalInputLayer(
                embedding_layer=embedding_layer,
                empty=m_cfg.categorical_padding_value,
                padding_categorical_layer=cat_pad_layer,
                )
        else:
            cat_input_layer = layers.EmptyLayer()

        input_layer = layers.OptionalInputLayer(
            continuous_layer_inst=cont_input_layer,
            categorical_layer_inst=cat_input_layer,
        )
        return input_layer

    def init_last_activation_layer(self) -> torch.nn.Module | None:
        """
        Helper to init the very last activation function.

        Returns:
            torch.nn.Module | None: Either an instance of the last activation layer or None.
        """
        # pick activation layer from torch by string,
        if self.model_building_config.last_activation_fn:
            last_activation_fn = getattr(torch.nn.modules.activation, self.model_building_config.last_activation_fn)
            if self.model_building_config.last_activation_fn == "Softmax":
                return last_activation_fn(dim=1)
            elif self.model_building_config.last_activation_fn == "Sigmoid":
                return last_activation_fn()
        return None


    @classmethod
    def register_learning_mode(cls, fn) -> None:
        """
        Helper function *fn* to register learning modes.
        These function take a model and modify their training status.

        Args:
            fn (function): Function that takes a model instance as input and sets the requires_grad attribute of the layers accordingly.
        """
        # helper method to register learning modes
        # defines a function as learning mode, which can be set with set_learning_mode. The function needs to take the model instance as input and set the requires_grad attribute of the layers accordingly.
        cls.LEARNING_MODES[fn.__name__] = fn
        return fn

    def set_learning_mode(self, mode):
        """
        Set the learning mode of the model, which determines which layers are trainable.

        Args:
            mode (str): String that determines the learning mode. Needs to be registered with the register_learning_mode decorator.
        """
        mode = f"_{mode}"
        if mode not in self.LEARNING_MODES:
            raise ValueError(f"Learning mode {mode} is not registered. Available learning modes are: {list(self.LEARNING_MODES.keys())}")
        # set all layers to non trainable
        self.LEARNING_MODES["_freeze_all"](self)
        # set specific layers to trainable depending on mode
        self.LEARNING_MODES[mode](self)

@BaseModel.register_learning_mode
def _freeze_all(model):
    for name, layer in model.named_children():
        layer.requires_grad = False

@BaseModel.register_learning_mode
def _unfreeze_all(model):
    for name, layer in model.named_children():
        layer.requires_grad = True

@register_model("residual")
class ResidualDNN(BaseModel):
    def __init__(self, full_config, *args, **kwargs):
        super().__init__(full_config, *args, **kwargs)
        self.init_layers()

    def init_layers(self):
        # increasing eps helps to stabilize training to counter batch norm and L2 reg counterplay when used together.
        # eps = 0.5e-5

        # helper where all layers are defined
        # std layers are filled when statitics are known
        self.init_standardization_layer()
        self.init_padding_layer()

        self.init_rotation_layer()
        self.init_input_layer()

        cfg = {
            "nodes" : self.model_building_config.nodes,
            "activation_functions" : self.model_building_config.activation_functions,
            "skip_connection_init" : self.model_building_config.skip_connection_init,
            "freeze_skip_connection" :self.model_building_config.freeze_skip_connection,
            "eps" : self.model_building_config.batch_norm_eps,
            "normalize" : False # activate weight normalization on linear layer weights
        }

        self.transition_dense_1 = layers.DenseBlock(
            input_nodes = self.input_layer.ndim,
            output_nodes = self.model_building_config.nodes,
            **cfg,
            )
        self.resnet_block_1 = layers.ResNetPreactivationBlock(**cfg)
        self.resnet_block_2 = layers.ResNetPreactivationBlock(**cfg)
        self.resnet_block_3 = layers.ResNetPreactivationBlock(**cfg)
        self.last_linear = torch.nn.Linear(self.model_building_config.nodes, 3)

    def forward(self, categorical_inputs, continuous_inputs):
        x = self.input_layer(categorical_inputs=categorical_inputs, continuous_inputs=continuous_inputs)
        x = self.transition_dense_1(x)
        x = self.resnet_block_1(x)
        x = self.resnet_block_2(x)
        x = self.resnet_block_3(x)
        x = self.last_linear(x)
        return x

@register_model("dense")
class DenseNet(BaseModel):
    def __init__(self, full_config, *args, **kwargs):
        super().__init__(full_config, *args, **kwargs)

        # init layers
        self.dense_config = {
            "skip_connection_init" : self.model_building_config.skip_connection_init,
            "freeze_skip_connection" : self.model_building_config.freeze_skip_connection,
            "activation_functions" : self.model_building_config.activation_functions,
            "eps": self.model_building_config.eps_batchnorm, # increasing eps helps to stabilize training, to counter batch norm and L2 reg counter play when used together
            "normalize" : self.model_building_config.normalize_linear, # activate weight normalization on linear layer weights
        }
        self.init_layers()


    def init_layers(self):
        m_cfg = self.model_building_config

        # self.input_layer = self.init_input_layer()
        self.input_layer = self.init_optional_input_layer()

        self.transition_dense_1 = layers.DenseBlock(
            input_nodes=self.input_layer.ndim,
            output_nodes=int(m_cfg.nodes),
            activation_functions=m_cfg.activation_functions,
            eps=self.dense_config["eps"],
            normalize=self.dense_config["normalize"],
            )
        self.dense_block_1 = layers.DenseNetBlock(input_nodes=self.transition_dense_1.output_dim, output_nodes=int(m_cfg.nodes), **self.dense_config)
        self.dense_block_2 = layers.DenseNetBlock(input_nodes=self.dense_block_1.output_dim, output_nodes=int(m_cfg.nodes), **self.dense_config)
        self.dense_block_3 = layers.DenseNetBlock(input_nodes=self.dense_block_2.output_dim, output_nodes=int(m_cfg.nodes), **self.dense_config)
        self.dense_block_4 = layers.DenseNetBlock(input_nodes=self.dense_block_3.output_dim, output_nodes=int(m_cfg.nodes), **self.dense_config)
        self.dense_block_5 = layers.DenseNetBlock(input_nodes=self.dense_block_4.output_dim, output_nodes=int(m_cfg.nodes), **self.dense_config)
        self.last_linear = torch.nn.Linear(self.dense_block_5.output_dim, len(self.dataset_config.target_map.keys()))

        # can only be sigmoid or softmax, uses only dim as configuration
        self.last_activaton_fn = self.init_last_activation_layer()

    def forward(self, categorical_inputs, continuous_inputs):
        x = self.input_layer(categorical_inputs=categorical_inputs, continuous_inputs=continuous_inputs)
        x = self.transition_dense_1(x)
        x = self.dense_block_1(x)
        x = self.dense_block_2(x)
        x = self.dense_block_3(x)
        x = self.dense_block_4(x)
        x = self.dense_block_5(x)
        x = self.last_linear(x)
        if self.use_last_activation:
            self.last_activaton_fn(x)
        return x

@register_model("lbn_dense")
class LBNDenseNet(DenseNet):
    def __init__(
        self,
        full_config,
        *args,
        **kwargs
        ):
        # has same init as DenseNet
        super().__init__(full_config, *args, **kwargs)

    def init_layers(self):
        # create normal DenseNet but also LBN
        super().init_layers()
        # old transition layer needs combined input dimension lbn and input_layer
        self.lbn = layers.LBN_DNN(
            continuous_features=self.dataset_config.continuous_features,
            M =self.model_building_config.LBN_M,
            weight_init_scale=1.0,
            clip_weights=False,
            eps=1.0e-5,
        )

        self.transition_dense_1 = layers.DenseBlock(
            input_nodes=(self.input_layer.ndim + self.lbn.ndim),
            output_nodes=self.model_building_config.nodes,
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
        if self.use_last_activation:
            x = self.last_activaton_fn(x)
        return x

@register_model("binned_lbn_dense")
class BinnedLBNDenseNetV2(LBNDenseNet):
    def __init__(
        self,
        full_config,
        *args,
        **kwargs,
        ):
        super().__init__(full_config, *args, **kwargs)
        self.is_binned = True


    def init_layers(self):
        # create normal LBN DenseNet
        super().init_layers()
        # init binning layer
        # add binning layer if necessary
        if self.binning_config is None:
            self.binning_layer = torch.nn.Identity()
        else:
            self.binning_layer = layers.BinningLayerRight(
                num_bins=self.binning_config.num_bins,
                bounds=self.binning_config.bounds,
                binning_fn=self.binning_config.binning_fn,
                kernel_cls=getattr(kernel, self.binning_config.kernel_cls),
                kernel_cfg=self.binning_config.kernel_config[self.binning_config.kernel_cls],
                )

    def forward(self, categorical_inputs, continuous_inputs):
        normal_network_output = super().forward(categorical_inputs, continuous_inputs)
        binned_output = self.binning_layer(normal_network_output) # increases dimension at axis 0
        return normal_network_output, binned_output

@BinnedLBNDenseNetV2.register_learning_mode
def _bin_only(model):
    all_layers = dict(model.named_children())
    all_layers["binning_layer"].requires_grad = True

@BinnedLBNDenseNetV2.register_learning_mode
def _model_only(model):
    all_layers_except_binning = dict(model.named_children())
    all_layers_except_binning.pop("binning_layer")
    for name, layer in all_layers_except_binning.items():
        layer.requires_grad = True



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
            # always go on object dimension, which is last
            self.last_activation = last_activation_fn(dim=-1)
        else:
            self.last_activation = self.last_activation()

        # add binning layer if necessary
        if binning_config is None:
            self.binning_layer = torch.nn.Identity()
        else:
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
