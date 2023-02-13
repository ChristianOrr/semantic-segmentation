import jax
import flax.linen as nn
import jax.numpy as jnp
import resnet_models
from functools import partial
from typing import Any, Callable, Sequence, Tuple
ModuleDef = Any


class _pspnet(nn.Module):
    num_classes: int
    initializer: str = "yilmaz_normal"
    act: Callable = nn.relu
    backbone: str = "ResNet50"
    bins: tuple[tuple] = ((1, 1), (2, 2), (3, 3), (6, 6))
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        # Scale values to be between 0-1
        x = jnp.array(x / 256, dtype=self.dtype)
        height, width = x.shape[-3], x.shape[-2] 

        # Select the initializer
        init_kwargs = {"in_axis":-2, "out_axis":-1, "batch_axis":0, "dtype": self.dtype}
        if self.initializer == "he_normal":
            initializer = nn.initializers.he_normal(**init_kwargs)
        elif self.initializer == "he_uniform":
            initializer = nn.initializers.he_uniform(**init_kwargs)
        elif self.initializer == "xavier_normal":
            initializer = nn.initializers.xavier_normal(**init_kwargs)
        elif self.initializer == "xavier_uniform":
            initializer = nn.initializers.xavier_uniform(**init_kwargs)
        elif self.initializer == "kumar_normal":
            # From: https://arxiv.org/abs/1704.08863
            initializer = nn.initializers.variance_scaling(scale=3.6**2, mode="fan_avg", distribution="truncated_normal", **init_kwargs)
        elif self.initializer == "yilmaz_normal":
            # From: https://www.sciencedirect.com/science/article/abs/pii/S0893608022002040
            init1 = nn.initializers.variance_scaling(scale=8, mode="fan_avg", distribution="truncated_normal", **init_kwargs)
            init2 = nn.initializers.constant(-1)
            def initializer(*args, **kwargs):
                return jnp.maximum(-init1(*args, **kwargs), init2(*args, **kwargs))
        else:
            raise NotImplementedError(f"The initializer {self.initializer} is not supported.")

        act = self.act
        conv_kwargs = {"padding":"SAME", "kernel_init":initializer, "use_bias":False, "dtype": self.dtype}
        norm_kwargs = {"num_groups":32, "dtype":self.dtype}
        pool_kwargs = {"padding":"VALID", "count_include_pad":True}

        ########################### Feature Map ############################
        backbone = getattr(resnet_models, self.backbone)
        feature_map = backbone()(x)
        batch_size, height_features, width_features, channel_features = feature_map.shape


        ###################### Pyramid Pooling Module #######################
        ppm_channels = channel_features // len(self.bins)
        pyramid_layers = [feature_map]
        for bin in self.bins:
            layer_out = nn.Sequential([
                partial(nn.avg_pool, window_shape=bin, strides=bin, **pool_kwargs), 
                nn.Conv(features=ppm_channels, kernel_size=(1, 1), **conv_kwargs), 
                nn.GroupNorm(**norm_kwargs), 
                act
            ])(feature_map)
            layer_out = jax.image.resize(
                layer_out, 
                shape=(batch_size, height_features, width_features, ppm_channels), 
                method="bilinear"
            )
            pyramid_layers.append(layer_out)
        x = jnp.concatenate(pyramid_layers, axis=-1)

        ########################### CLS ############################
        x = nn.Sequential([ 
            nn.Conv(features=ppm_channels, kernel_size=(3, 3), **conv_kwargs), 
            nn.GroupNorm(**norm_kwargs), 
            act,
            nn.Dropout(rate=0.1, deterministic=True),
            nn.Conv(features=self.num_classes, kernel_size=(1, 1), **conv_kwargs)
        ])(x)


        x = jax.image.resize(
            x, 
            shape=(batch_size, height, width, self.num_classes), 
            method="bilinear"
        )
        x = jax.nn.softmax(x)
        return x