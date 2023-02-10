import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnames=['num_classes', 'dtype'])
def binarize_target(target_ids, num_classes, dtype=jnp.float32):
    """
    Converts a single array of target IDs to binary form.
    Each ID will be represented in a seperate channel.
    Args:
        target_ids: A single array of class IDs, 
            with values between 0 and num_classes.
        num_classes: Total number of distinct classes.
        dtype: Array data type.
    Returns:
        A binarized array of class IDs. 
    """
    binary_labels = []
    for class_id in range(num_classes):
        binary_labels.append(target_ids == class_id)
    binary_labels = jnp.array(binary_labels, dtype=dtype)
    # Move class labels to last dimension
    binary_labels = jnp.moveaxis(binary_labels, source=0, destination=-1)
    return binary_labels

@partial(jax.jit, static_argnames=['num_classes', 'height', 'width', 'dtype'])
def prep_data(input, target_ids, height, width, num_classes, dtype=jnp.float32):
    """
    Prepares the data by resizing to the requested height/width and
    binarizing the target ID's.
    Args:
        input: A single RGB input image.
        target_ids: A single segmentation mask for the image.
        height: Height dimension for resizing the images.
        width: Width dimension for resizing the images.
        num_classes: Total number of distinct classes.
        dtype: Array data type.
    Returns:
        The resized input image and resized binarized target ID's.
    """
    # input, target_ids = jnp.array(data["image"][0]), jnp.array(data["annotation"][0])
    target_ids_binary = binarize_target(target_ids, num_classes, dtype=dtype)
    # Downsample the image
    input = jax.image.resize(input, shape=(height, width, 3), method="bilinear")
    target_ids_binary = jax.image.resize(target_ids_binary, shape=(height, width, num_classes), method="nearest")
    return input, target_ids_binary


def prep_data_batch(data_generator, batch_size, height, width, num_classes, dtype=jnp.float32):
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
        input, target_ids = jnp.array(data["image"][0], dtype=dtype), jnp.array(data["annotation"][0], dtype=dtype)
        # Don't add the image if its not RGB
        if len(input.shape) == 2 or input.shape[-1] == 1:
            continue
        input, target_ids_binary = prep_data(input, target_ids, height, width, num_classes, dtype=dtype)
        inputs.append(input)
        targets.append(target_ids_binary)
    inputs = jnp.array(inputs, dtype=dtype)
    targets = jnp.array(targets, dtype=dtype)
    return inputs, targets


@jax.jit
def grads_vanished_or_exploded(grads, max_mean_grad, min_mean_grad):
    """
    Checks if the gradients have vanished or exploded.
    Args:
        grads: The gradinents from the loss function.
        max_mean_grad: Maximum mean gradient allowed before 
            being considered as expoding gradients. 
        min_mean_grad: Minimum mean gradient allowed before 
            being considered as vanishing gradients. 
    Returns:
        has_vanished: Boolean, True for vanished gradients, false otherwise.  
        has_exploded: Boolean, True for exploded gradients, false otherwise. 
        mean_grads: Mean gradients value.
    """
    params = grads["params"]
    mean_grads = []

    for layer in params.values():
        for weights in layer.values():
            # Check minimum weight and update if necessary
            layer_mean = weights.mean()
            mean_grads.append(layer_mean)

    mean_grads = jnp.absolute(jnp.array(mean_grads).mean())
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