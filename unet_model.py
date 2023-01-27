import jax
import flax.linen as nn
import jax.numpy as jnp


class _unet(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x):
        # Scale values to be between 0-1
        x = x / 256
        initializer = nn.initializers.normal(0.5)
        ########################### Encoders ############################
        # Encoder block 1
        encoder_block1 = nn.Sequential([
            nn.Conv(features=64, kernel_size=(3, 3), padding="SAME", kernel_init=initializer), jax.nn.leaky_relu,
            nn.Conv(features=64, kernel_size=(3, 3), padding="SAME", kernel_init=initializer), jax.nn.leaky_relu,
            nn.GroupNorm(num_groups=32)
        ])
        encoder1 = encoder_block1(x)
        x = encoder1
        x = nn.max_pool(inputs=x, window_shape=(2, 2), strides=(2, 2), padding="SAME")
        # Encoder block 2
        encoder_block2 = nn.Sequential([
            nn.Conv(features=128, kernel_size=(3, 3), padding="SAME", kernel_init=initializer), jax.nn.leaky_relu,
            nn.Conv(features=128, kernel_size=(3, 3), padding="SAME", kernel_init=initializer), jax.nn.leaky_relu,
            nn.GroupNorm(num_groups=32)
        ])
        encoder2 = encoder_block2(x)
        x = encoder2
        x = nn.max_pool(inputs=x, window_shape=(2, 2), strides=(2, 2), padding="SAME")
        # Encoder block 3
        encoder_block3 = nn.Sequential([
            nn.Conv(features=256, kernel_size=(3, 3), padding="SAME", kernel_init=initializer), jax.nn.leaky_relu,
            nn.Conv(features=256, kernel_size=(3, 3), padding="SAME", kernel_init=initializer), jax.nn.leaky_relu,
            nn.GroupNorm(num_groups=32)
        ])
        encoder3 = encoder_block3(x)
        x = encoder3
        x = nn.max_pool(inputs=x, window_shape=(2, 2), strides=(2, 2), padding="SAME")
        # Encoder block 4
        encoder_block4 = nn.Sequential([
            nn.Conv(features=512, kernel_size=(3, 3), padding="SAME", kernel_init=initializer), jax.nn.leaky_relu,
            nn.Conv(features=512, kernel_size=(3, 3), padding="SAME", kernel_init=initializer), jax.nn.leaky_relu,
            nn.GroupNorm(num_groups=32),
            nn.Dropout(rate=0.3, deterministic=True)
        ])
        encoder4 = encoder_block4(x)
        x = encoder4
        x = nn.max_pool(inputs=x, window_shape=(2, 2), strides=(2, 2), padding="SAME")
        # Encoder block 5
        encoder_block5 = nn.Sequential([
            nn.Conv(features=1024, kernel_size=(3, 3), padding="SAME", kernel_init=initializer), jax.nn.leaky_relu,
            nn.Conv(features=1024, kernel_size=(3, 3), padding="SAME", kernel_init=initializer), jax.nn.leaky_relu,
            nn.GroupNorm(num_groups=32),
            nn.Dropout(rate=0.3, deterministic=True)
        ])
        x = encoder_block5(x)


        ########################### Decoders ############################
        # Decoder block 4
        x = nn.ConvTranspose(features=512, kernel_size=(2, 2), strides=(2, 2), padding="SAME", kernel_init=initializer)(x)
        x = jax.nn.leaky_relu(x)
        x = nn.GroupNorm(num_groups=32)(x)
        x = jnp.concatenate((x, encoder4), axis=-1)
        decoder_block3 = nn.Sequential([
            nn.Conv(features=512, kernel_size=(3, 3), padding="SAME", kernel_init=initializer), jax.nn.leaky_relu,   
            nn.Conv(features=512, kernel_size=(3, 3), padding="SAME", kernel_init=initializer), jax.nn.leaky_relu, 
            nn.GroupNorm(num_groups=32)
        ])  
        x = decoder_block3(x)    
        # Decoder block 3
        x = nn.ConvTranspose(features=256, kernel_size=(2, 2), strides=(2, 2), padding="SAME", kernel_init=initializer)(x)
        x = jax.nn.leaky_relu(x)
        x = nn.GroupNorm(num_groups=32)(x)
        x = jnp.concatenate((x, encoder3), axis=-1)
        decoder_block3 = nn.Sequential([
            nn.Conv(features=256, kernel_size=(3, 3), padding="SAME", kernel_init=initializer), jax.nn.leaky_relu,   
            nn.Conv(features=256, kernel_size=(3, 3), padding="SAME", kernel_init=initializer), jax.nn.leaky_relu, 
            nn.GroupNorm(num_groups=32)
        ])  
        x = decoder_block3(x)  
        # Decoder block 2
        x = nn.ConvTranspose(features=128, kernel_size=(2, 2), strides=(2, 2), padding="SAME", kernel_init=initializer)(x)
        x = jax.nn.leaky_relu(x)
        x = nn.GroupNorm(num_groups=32)(x)
        x = jnp.concatenate((x, encoder2), axis=-1)
        decoder_block2 = nn.Sequential([
            nn.Conv(features=128, kernel_size=(3, 3), padding="SAME", kernel_init=initializer), jax.nn.leaky_relu,   
            nn.Conv(features=128, kernel_size=(3, 3), padding="SAME", kernel_init=initializer), jax.nn.leaky_relu, 
            nn.GroupNorm(num_groups=32)
        ])  
        x = decoder_block2(x)  
        # Decoder block 1
        x = nn.ConvTranspose(features=64, kernel_size=(2, 2), strides=(2, 2), padding="SAME", kernel_init=initializer)(x)
        x = jax.nn.leaky_relu(x)
        x = nn.GroupNorm(num_groups=32)(x)
        x = jnp.concatenate((x, encoder1), axis=-1)
        decoder_block1 = nn.Sequential([
            nn.Conv(features=64, kernel_size=(3, 3), padding="SAME", kernel_init=initializer), jax.nn.leaky_relu,   
            nn.Conv(features=64, kernel_size=(3, 3), padding="SAME", kernel_init=initializer), jax.nn.leaky_relu, 
            nn.GroupNorm(num_groups=32),
            nn.Conv(features=self.num_classes, kernel_size=(1, 1), padding="SAME", kernel_init=initializer), jax.nn.sigmoid,
        ])  
        x = decoder_block1(x)  

        return x