import numpy as np
import jax
import jax.numpy as jnp


def binarize_target(target_ids, num_classes):
    """
    Converts a single array of target IDs to binary form.
    Each ID will be represented in a seperate channel.
    Args:
        target_ids: A single array of class IDs, 
            with values between 0 and num_classes.
        num_classes: Total number of distinct classes.
    Returns:
        A binarized array of class IDs. 
    """
    (dim_x, dim_y) = target_ids.shape
    binary_labels = np.zeros(shape=(dim_x, dim_y, num_classes))
    for class_id in range(num_classes):
        binary_labels[:, :, class_id] = target_ids == class_id
    return jnp.array(binary_labels)


def prep_data(data, height, width, num_classes):
    """
    Prepares the data by resizing to the requested height/width and
    binarizing the target ID's.
    Args:
        data: Dictionary containing a single input image and annotation.
        height: Height dimension for resizing the images.
        width: Width dimension for resizing the images.
        num_classes: Total number of distinct classes.
    Returns:
        The resized input image and resized binarized target ID's.
    """
    input, target_ids = jnp.array(data["image"][0]), np.array(data["annotation"][0])
    target_ids_binary = binarize_target(target_ids, num_classes)
    # Downsample the image
    input = jax.image.resize(input, shape=(height, width, 3), method="bilinear")
    target_ids_binary = jax.image.resize(target_ids_binary, shape=(height, width, num_classes), method="nearest")
    return input, target_ids_binary


def prep_data_batch(data_generator, batch_size, height, width, num_classes):
    """
    Prepares a batch of data for training/validation.
    Args:
        data_generator: Generator object containing the 
            training/validation data.
        batch_size: Number of data to process in parallel. 
        height: Height dimension for resizing the images.
        width: Width dimension for resizing the images.
        num_classes: Total number of distinct classes.
    Returns:
        A Jax array of input images and target images.
    """
    inputs = []
    targets = []
    for _ in range(batch_size):
        data = next(data_generator)
        input, target_ids_binary = prep_data(data, height, width, num_classes)
        inputs.append(input)
        targets.append(target_ids_binary)
    inputs = jnp.array(inputs)
    targets = jnp.array(targets)
    return inputs, targets


def grads_zeroed(grads):
    """
    Checks if the gradients have zeroed,
    (due to the vanishing gradient problem).
    Args:
        grads: The gradinents from the loss function.
    Returns:
        Boolean, True for zeroed gradients, false otherwise.
    """
    params = grads["params"]
    max_grad = 0
    min_grad = 0

    for layer in params.values():
        for weights in layer.values():
            # Check minimum weight and update if necessary
            layer_min = weights.min()
            if layer_min < min_grad:
                min_grad = layer_min
            # Check minimum weight and update if necessary
            layer_max = weights.min()
            if layer_max > max_grad:
                max_grad = layer_max
    
    return min_grad == max_grad == 0