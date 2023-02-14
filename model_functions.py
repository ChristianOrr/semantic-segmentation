import flax.linen as nn
import jax.numpy as jnp


def get_initializer(init_name, dtype):
    """
    Function for extracting the weight initialization function.
    Args:
        init_name: String, name of the intialization function.
        dtype: Data type of the initialization function.
    Returns:
        Initialization function.
    
    """
    init_kwargs = {"in_axis":-2, "out_axis":-1, "batch_axis":0, "dtype": dtype}
    if init_name == "he_normal":
        initializer = nn.initializers.he_normal(**init_kwargs)
    elif init_name == "he_uniform":
        initializer = nn.initializers.he_uniform(**init_kwargs)
    elif init_name == "xavier_normal":
        initializer = nn.initializers.xavier_normal(**init_kwargs)
    elif init_name == "xavier_uniform":
        initializer = nn.initializers.xavier_uniform(**init_kwargs)
    elif init_name == "kumar_normal":
        # From: https://arxiv.org/abs/1704.08863
        initializer = nn.initializers.variance_scaling(scale=3.6**2, mode="fan_avg", distribution="truncated_normal", **init_kwargs)
    elif init_name == "yilmaz_normal":
        # From: https://www.sciencedirect.com/science/article/abs/pii/S0893608022002040
        init1 = nn.initializers.variance_scaling(scale=8, mode="fan_avg", distribution="truncated_normal", **init_kwargs)
        init2 = nn.initializers.constant(-1)
        def initializer(*args, **kwargs):
            return jnp.maximum(-init1(*args, **kwargs), init2(*args, **kwargs))
    else:
        raise NotImplementedError(f"The initializer {init_name} is not supported.")

    return initializer