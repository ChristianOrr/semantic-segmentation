import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnames=['height', 'width'])
def prep_data(input, target_ids, height, width):
    """
    Prepares the data by resizing to the requested height/width.
    Args:
        input: A single RGB input image.
        target_ids: A single segmentation mask for the image.
        height: Height dimension for resizing the images.
        width: Width dimension for resizing the images.
        num_classes: Total number of distinct classes.
        dtype: Array data type.
    Returns:
        The resized input image and resized target ID's.
    """
    input = jax.image.resize(input, shape=(height, width, 3), method="bilinear")
    target_ids = jax.image.resize(target_ids, shape=(height, width), method="nearest")
    return input, target_ids


def prep_data_batch(data_generator, batch_size, height, width, dtype=jnp.float32):
    """
    Prepares a batch of data for training/validation.
    Args:
        data_generator: Generator object containing the 
            training/validation data.
        batch_size: Number of data to process in parallel. 
        height: Height dimension for resizing the images.
        width: Width dimension for resizing the images.
        num_classes: Total number of distinct classes.
        dtype: Array data type.
    Returns:
        A Jax array of input images and target images.
    """
    inputs = []
    targets = []
    for _ in range(batch_size):
        data = next(data_generator)
        input, target_ids = jnp.array(data["image"][0], dtype=dtype), jnp.array(data["annotation"][0], dtype=jnp.uint16)
        # Don't add the image if its not RGB
        if len(input.shape) == 2 or input.shape[-1] == 1:
            continue
        input, target_ids = prep_data(input, target_ids, height, width)
        inputs.append(input)
        targets.append(target_ids)
    inputs = jnp.array(inputs, dtype=dtype)
    targets = jnp.array(targets, dtype=jnp.uint16)
    return inputs, targets
    

@jax.jit
def grads_vanished_or_exploded(params, max_mean_grad, min_mean_grad):
    """
    Checks if the gradients have vanished or exploded.
    Args:
        params: The model weight gradinents from the loss function.
        max_mean_grad: Maximum mean gradient allowed before 
            being considered as expoding gradients. 
        min_mean_grad: Minimum mean gradient allowed before 
            being considered as vanishing gradients. 
    Returns:
        has_vanished: Boolean, True for vanished gradients, false otherwise.  
        has_exploded: Boolean, True for exploded gradients, false otherwise. 
        mean_grads: Mean gradients value.
    """
    grads = jax.tree_util.tree_leaves(params)
    mean_grads = jnp.absolute(jnp.array([grad.mean() for grad in grads]))
    mean_grads = mean_grads.mean()

    has_vanished = mean_grads < min_mean_grad
    has_exploded = mean_grads > max_mean_grad
    return has_vanished,  has_exploded, mean_grads


def create_infinite_generator(dataset):
    """
    Converts a hugginface dataset into an infinite generator.
    The dataset is shuffled and batched into single samples.
    Args:
        dataset: Huggingface dataset.
    Yields:
        Next random sample.
    """
    while True:
        gen = dataset.shuffle().iter(batch_size=1)
        for _ in range(dataset.num_rows - 1):
            yield next(gen)


def dict_mean(dict_list):
    """
    Calculates the mean values of a list of dictionaries.
    Args:
        dict_list: List of dictionaries.
    Returns:
        Dictionary with mean values rounded to 4 decimal places.
    """
    mean_dict = {}
    for key in dict_list[0].keys():
        value = sum(float(d[key]) for d in dict_list) / len(dict_list)
        mean_dict[key] = f"{value :.4f}"
    return mean_dict