import jax
import flax.linen as nn
import jax.numpy as jnp
from functools import partial
from typing import Any, Callable, Sequence, Tuple
ModuleDef = Any


class _unet(nn.Module):
    num_classes: int
    initializer: str = "yilmaz_normal"
    act: Callable = nn.relu
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        # Scale values to be between 0-1
        x = jnp.array(x / 256, dtype=self.dtype) 

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
        pool_kwargs = {"window_shape":(2, 2), "strides":(2, 2), "padding":"SAME"}
        ########################### Encoders ############################
        # Encoder block 1
        encoder_block1 = nn.Sequential([
            nn.Conv(features=64, kernel_size=(3, 3), **conv_kwargs), nn.GroupNorm(**norm_kwargs), act,
            nn.Conv(features=64, kernel_size=(3, 3), **conv_kwargs), nn.GroupNorm(**norm_kwargs), act
        ])
        encoder1 = encoder_block1(x)
        x = encoder1
        x = nn.max_pool(inputs=x, **pool_kwargs)
        # Encoder block 2
        encoder_block2 = nn.Sequential([
            nn.Conv(features=128, kernel_size=(3, 3), **conv_kwargs), nn.GroupNorm(**norm_kwargs), act,
            nn.Conv(features=128, kernel_size=(3, 3), **conv_kwargs), nn.GroupNorm(**norm_kwargs), act
        ])
        encoder2 = encoder_block2(x)
        x = encoder2
        x = nn.max_pool(inputs=x, **pool_kwargs)
        # Encoder block 3
        encoder_block3 = nn.Sequential([
            nn.Conv(features=256, kernel_size=(3, 3), **conv_kwargs), nn.GroupNorm(**norm_kwargs), act,
            nn.Conv(features=256, kernel_size=(3, 3), **conv_kwargs), nn.GroupNorm(**norm_kwargs), act
        ])
        encoder3 = encoder_block3(x)
        x = encoder3
        x = nn.max_pool(inputs=x, **pool_kwargs)
        # Encoder block 4
        encoder_block4 = nn.Sequential([
            nn.Conv(features=512, kernel_size=(3, 3), **conv_kwargs), nn.GroupNorm(**norm_kwargs), act,
            nn.Conv(features=512, kernel_size=(3, 3), **conv_kwargs), nn.GroupNorm(**norm_kwargs), act,
            nn.Dropout(rate=0.3, deterministic=True)
        ])
        encoder4 = encoder_block4(x)
        x = encoder4
        x = nn.max_pool(inputs=x, **pool_kwargs)
        # Encoder block 5
        encoder_block5 = nn.Sequential([
            nn.Conv(features=1024, kernel_size=(3, 3), **conv_kwargs), nn.GroupNorm(**norm_kwargs), act,
            nn.Conv(features=1024, kernel_size=(3, 3), **conv_kwargs), nn.GroupNorm(**norm_kwargs), act,
            nn.Dropout(rate=0.3, deterministic=True)
        ])
        x = encoder_block5(x)


        ########################### Decoders ############################
        # Decoder block 4
        x = nn.ConvTranspose(features=512, kernel_size=(2, 2), strides=(2, 2), **conv_kwargs)(x)
        x = nn.GroupNorm(**norm_kwargs)(x)
        x = act(x)
        x = jnp.concatenate((x, encoder4), axis=-1)
        decoder_block3 = nn.Sequential([
            nn.Conv(features=512, kernel_size=(3, 3), **conv_kwargs), nn.GroupNorm(**norm_kwargs), act,   
            nn.Conv(features=512, kernel_size=(3, 3), **conv_kwargs), nn.GroupNorm(**norm_kwargs), act, 
        ])  
        x = decoder_block3(x)    
        # Decoder block 3
        x = nn.ConvTranspose(features=256, kernel_size=(2, 2), strides=(2, 2), **conv_kwargs)(x)
        x = nn.GroupNorm(**norm_kwargs)(x)
        x = act(x)
        x = jnp.concatenate((x, encoder3), axis=-1)
        decoder_block3 = nn.Sequential([
            nn.Conv(features=256, kernel_size=(3, 3), **conv_kwargs), nn.GroupNorm(**norm_kwargs), act,   
            nn.Conv(features=256, kernel_size=(3, 3), **conv_kwargs), nn.GroupNorm(**norm_kwargs), act, 
        ])  
        x = decoder_block3(x)  
        # Decoder block 2
        x = nn.ConvTranspose(features=128, kernel_size=(2, 2), strides=(2, 2), **conv_kwargs)(x)
        x = nn.GroupNorm(**norm_kwargs)(x)
        x = act(x)
        x = jnp.concatenate((x, encoder2), axis=-1)
        decoder_block2 = nn.Sequential([
            nn.Conv(features=128, kernel_size=(3, 3), **conv_kwargs), nn.GroupNorm(**norm_kwargs), act,   
            nn.Conv(features=128, kernel_size=(3, 3), **conv_kwargs), nn.GroupNorm(**norm_kwargs), act, 
        ])  
        x = decoder_block2(x)  
        # Decoder block 1
        x = nn.ConvTranspose(features=64, kernel_size=(2, 2), strides=(2, 2), **conv_kwargs)(x)
        x = nn.GroupNorm(**norm_kwargs)(x)
        x = act(x)
        x = jnp.concatenate((x, encoder1), axis=-1)
        decoder_block1 = nn.Sequential([
            nn.Conv(features=64, kernel_size=(3, 3), **conv_kwargs), nn.GroupNorm(**norm_kwargs), act,   
            nn.Conv(features=64, kernel_size=(3, 3), **conv_kwargs), act, 
            nn.Conv(features=self.num_classes, kernel_size=(1, 1), **conv_kwargs), jax.nn.softmax,
        ])  
        x = decoder_block1(x)  

        return x