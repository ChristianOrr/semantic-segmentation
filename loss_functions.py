import jax
import jax.numpy as jnp
import optax
from functools import partial
from flax.training import common_utils


def loss_fn(
    variables, 
    state, 
    inputs, 
    labels, 
    num_classes, 
    loss_method="cross_entropy", 
    ignore_label=0, 
    weight_decay=4e-5, 
    epsilon=1e-15
):
    """
    Calculates the loss for a batch of images.
    Args:
        variables: The segmentation models parameters.
        state: State of the semantic segmentation model.
        inputs: A batch of raw input images.
        labels: A batch of arrays of segmentation mask ID's.
        num_classes: Total number of distinct classes.
        loss_method: Loss calculation method.
        ignore_label: Background class label to ignore.
        weight_decay: Regularization coefficient.
        epsilon: A small number to prevent loss function from dividing by zero.
    Returns:
        The loss from a batch of images and the model log odds for the batch.
    """
    forward_fn = state.apply_fn
    logits = forward_fn(variables, inputs)

    if loss_method == "cross_entropy":
        loss_calc = cross_entropy_loss
    elif loss_method == "dice":
        loss_calc = dice_loss
    else:
        raise NotImplementedError(f"The loss method {loss_method} is not supported.")

    loss = loss_calc(logits, labels, num_classes, ignore_label, epsilon=epsilon)
    # Regularization
    weight_penalty_params = jax.tree_util.tree_leaves(variables["params"])
    weight_l2 = sum([jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1])
    weight_penalty = weight_decay * 0.5 * weight_l2
    loss = loss + weight_penalty
    return loss, logits



@partial(jax.jit, static_argnames=['num_classes', 'ignore_label', 'class_weights'])
def dice_loss(
    logits,
    labels,
    num_classes,
    ignore_label=0,
    class_weights=None,
    label_smoothing=1e-3,
    epsilon=1e-15,
):
    """
    Calculates the dice loss for a batch of images.
    Args:
        logits: The log odds for a batch of images.
        labels: A batch of arrays of segmentation mask ID's.
        num_classes: Total number of distinct classes.
        ignore_label: Background class label to ignore.
        class_weights: List of importance weightings for the classes.
        label_smoothing: Prevents overconfident predictions.
        epsilon: A small number to prevent loss function from dividing by zero.
    Returns:
        The dice loss from a batch of images.
    """
    one_hot_labels = common_utils.onehot(labels, num_classes=num_classes)

    smoothed_one_hot_labels = (
        one_hot_labels * (1 - label_smoothing) + label_smoothing / num_classes
    )

    if class_weights is None:
        class_weights = jnp.ones(num_classes, dtype=jnp.float32)

    probs = jax.nn.softmax(logits, axis=-1)

    numerator = 2 * (probs * smoothed_one_hot_labels).sum(axis=(-2, -3))
    denominator = probs.sum(axis=(-2, -3)) + smoothed_one_hot_labels.sum(axis=(-2, -3))
    dice_coefficient = (numerator + epsilon) / (denominator + epsilon)
    dice_coefficient = dice_coefficient.mean(axis=0)
    # Remove the background class from the loss function
    dice_coefficient = jnp.delete(dice_coefficient, ignore_label)
    class_weights = jnp.delete(class_weights, ignore_label)
    # Get average coefficient using the weighting provided for each class
    mean_dice_coeff = (dice_coefficient * class_weights).sum() / class_weights.sum()

    dice_loss = 1 - mean_dice_coeff
    return dice_loss


@partial(jax.jit, static_argnames=['num_classes', 'class_weights'])
def cross_entropy_loss(
    logits,
    labels,
    num_classes,
    ignore_label=0,
    class_weights=None,
    label_smoothing=1e-3,
    epsilon=1e-15,
):
    """
    Calculates the cross entropy loss for a batch of images.
    Args:
        logits: The log odds for a batch of images.
        labels: A batch of arrays of segmentation mask ID's.
        num_classes: Total number of distinct classes.
        ignore_label: Background class label to ignore.
        class_weights: List of importance weightings for the classes.
        label_smoothing: Prevents overconfident predictions. 
        epsilon: A small number to prevent loss function from dividing by zero.
    Returns:
        The dice loss from a batch of images.

    Obtained from: https://github.com/NobuoTsukamoto/jax_examples/blob/4711d70bdf6ce707c8c5130a1e57fe4741176198/segmentation/train.py#L58
    """
    valid_mask = jnp.not_equal(labels, ignore_label)
    normalizer = jnp.sum(valid_mask.astype(jnp.float32)) + epsilon
    labels = jnp.where(valid_mask, labels, jnp.zeros_like(labels))

    one_hot_labels = common_utils.onehot(labels, num_classes=num_classes)

    smoothed_one_hot_labels = (
        one_hot_labels * (1 - label_smoothing) + label_smoothing / num_classes
    )

    if class_weights is None:
        class_weights = jnp.ones(num_classes, dtype=jnp.float32)

    weight_mask = jnp.einsum(
        "...y,y->...",
        one_hot_labels,
        class_weights,
    )
    valid_mask *= weight_mask

    xentropy = optax.softmax_cross_entropy(
        logits=logits, labels=smoothed_one_hot_labels
    )

    xentropy *= valid_mask.astype(jnp.float32)
    loss = jnp.sum(xentropy) / normalizer
    return loss
