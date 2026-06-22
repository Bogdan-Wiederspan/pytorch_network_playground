from __future__ import annotations

import torch

from ..input import CategoricalInputLayer, ContinuousInputLayer, InputLayer, OptionalInputLayer
from ..preprocessing import CatEmbeddingLayer, EmptyLayer, PaddingLayer, RotatePhiLayer, StandardizeLayer


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
        self.is_binned = False # this flag indicates if the network is two-headed

    def __init_subclass__(cls):
        super().__init_subclass__()

        modes = {}

        # traverse the mro diagram classes to get all learning modes from previous models.
        # if a more modern version of a learning_mode exist it will overwrite by default
        # the older version of the learning mode, due to reverse order of the mro (starting from base ending at current)
        # HINT: this creates unbound fn, shared by class - this is very cheap in memory, due to being run once!
        # but this also means that first argument (cls) is not set, so fn(self) needs to be used when called
        for parent in reversed(cls.mro()):
            for name, fn in parent.__dict__.items():
                if name.startswith("learning_mode_"):
                    modes[name.removeprefix("learning_mode_")] = fn

        cls.LEARNING_MODES = modes


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
        for _, module in self.named_modules():
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
            rotation_layer = RotatePhiLayer(
                columns=list(map(str, self.dataset_config.continuous_features)),
                ref_phi_columns=self.model_building_config.ref_phi_columns,
                rotate_columns=self.model_building_config.rotate_columns,
            )
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
            std_layer = StandardizeLayer(
                mean = torch.zeros(len(self.dataset_config.continuous_features)),
                std = torch.ones(len(self.dataset_config.continuous_features))
            )
        else:
            std_layer = StandardizeLayer(mean=self.model_building_config.mean, std=self.model_building_config.std)
        return std_layer

    def init_padding_layer(self)-> tuple[torch.nn.Module | None, torch.nn.Module | None]:
        """
        Helper function to initialize continuous and categorical padding layer.
        The actual padding value is defined in the model building config, if the value is None, no padding layer is created.

        Returns:
            tuple[torch.nn.Module | None, torch.nn.Module | None]: None or tuple of PaddingLayer instances.
        """
        m_cfg = self.model_building_config
        if self.model_building_config.continuous_target_value is None:
            continuous_padding = None
        else:
            continuous_padding = PaddingLayer(
                target_value=m_cfg.continuous_target_value,
                padding_value=m_cfg.continuous_masking_value,
            )

        if self.model_building_config.categorical_target_value is None:
            categorical_padding = None
        else:
            categorical_padding = PaddingLayer(
                target_value=m_cfg.categorical_target_value,
                padding_value=m_cfg.categorical_masking_value,
                )

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
        input_layer = InputLayer(
            continuous_inputs=d_cfg.continuous_features,
            categorical_inputs=d_cfg.categorical_features,
            embedding_dim=m_cfg.embedding_dim,
            expected_categorical_inputs=m_cfg.expected_embedding_inputs,
            empty=m_cfg.categorical_masking_value,
            std_layer=std_layer,
            rotation_layer=rot_layer,
            padding_categorical_layer=cat_pad_layer,
            padding_continuous_layer=cont_pad_layer,
        )
        return input_layer

    def init_cat_embedding_layer(self) -> torch.nn.Module:
        embedding_layer = CatEmbeddingLayer(
            embedding_dim = self.model_building_config.embedding_dim,
            categories = self.dataset_config.categorical_features,
            expected_categorical_inputs = self.model_building_config.expected_embedding_inputs,
            empty = self.model_building_config.tokenizer_add_unknown_category,
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
        # m_cfg = self.model_building_config

        cont_pad_layer, cat_pad_layer = self.init_padding_layer()

        if d_cfg.continuous_features:
            std_layer = self.init_standardization_layer()
            rot_layer = self.init_rotation_layer()
            cont_input_layer = ContinuousInputLayer(
                continuous_inputs=d_cfg.continuous_features,
                rotation_layer=rot_layer,
                std_layer=std_layer,
                padding_continuous_layer=cont_pad_layer,
            )
        else:
            cont_input_layer = EmptyLayer()

        if d_cfg.categorical_features:
            embedding_layer = self.init_cat_embedding_layer()
            cat_input_layer = CategoricalInputLayer(
                embedding_layer=embedding_layer,
                padding_categorical_layer=cat_pad_layer,
                )
        else:
            cat_input_layer = EmptyLayer()

        input_layer = OptionalInputLayer(
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

    def set_learning_mode(self, mode):
        """
        Set the learning mode of the model, which determines which layers are trainable.

        Args:
            mode (str): String that determines the learning mode. Needs to be registered with the register_learning_mode decorator.
        """
        if mode not in self.LEARNING_MODES:
            raise ValueError(f"Learning mode {mode} is not registered. Available learning modes are: {list(self.LEARNING_MODES.keys())}")
        # set all layers to non trainable
        self.LEARNING_MODES["freeze_all"](self)
        # set specific layers to trainable depending on mode
        self.LEARNING_MODES[mode](self)

    def learning_mode_freeze_all(self):
        for _, layer in self.named_children():
            layer.requires_grad = False

    def learning_mode_unfreeze_all(self):
        for _, layer in self.named_children():
            layer.requires_grad = True
