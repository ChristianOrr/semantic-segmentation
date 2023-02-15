import jax
import jax.numpy as jnp
from flax.core.frozen_dict import freeze, unfreeze


@jax.jit
def dice_loss(variables, state, inputs, targets, epsilon):
    """
    Calculates the dice loss for a batch of images.
    Args:
        variables: The segmentation models parameters.
        state: State of the semantic segmentation model.
        inputs: A batch of raw input images.
        targets: A batch of arrays with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        epsilon: A small number to prevent loss function from dividing by zero.
    Returns:
        The dice loss from a batch of images.
    """
    forward_fn = state.apply_fn
    preds = forward_fn(variables, inputs)
    numerator = 2 * (preds * targets).sum(axis=(-2, -3))
    denominator = preds.sum(axis=(-2, -3)) + targets.sum(axis=(-2, -3))
    loss = 1 - jnp.mean((numerator + epsilon) / (denominator + epsilon))
    return loss


@jax.jit
def dice_loss_and_preds(variables, state, inputs, targets, epsilon):
    """
    Calculates the dice loss for a batch of images and 
    returns the loss and preds.
    Args:
        variables: The segmentation models parameters.
        state: State of the semantic segmentation model.
        inputs: A batch of raw input images.
        targets: A batch of arrays with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        epsilon: A small number to prevent loss function from dividing by zero.
    Returns:
        The dice loss and preds from a batch of images.
    """
    forward_fn = state.apply_fn
    preds = forward_fn(variables, inputs)
    numerator = 2 * (preds * targets).sum(axis=(-2, -3))
    denominator = preds.sum(axis=(-2, -3)) + targets.sum(axis=(-2, -3))
    loss = 1 - jnp.mean((numerator + epsilon) / (denominator + epsilon))
    return loss, preds


@jax.jit
def mse_loss(variables, state, inputs, targets):
    """
    Calculates the mean squared error loss for a batch of images.
    Args:
        variables: The segmentation models parameters.
        state: State of the semantic segmentation model.
        inputs: A batch of raw input images.
        targets: A batch of arrays with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        The dice loss from a batch of images.
    """
    forward_fn = state.apply_fn
    preds = forward_fn(variables, inputs)
    loss = jnp.mean((preds - targets)**2)
    return loss



@jax.jit
def dice_loss_bn(variables, state, inputs, targets, epsilon):
    """
    Calculates the dice loss for a batch of images.
    Args:
        variables: The segmentation models parameters.
        state: State of the semantic segmentation model.
        inputs: A batch of raw input images.
        targets: A batch of arrays with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        epsilon: A small number to prevent loss function from dividing by zero.
    Returns:
        The dice loss from a batch of images.
    """
    forward_fn = state.apply_fn
    preds, mutated_vars = forward_fn(variables, inputs, mutable=['batch_stats'])
    # variables = unfreeze(variables)
    # variables["batch_stats"] = mutated_vars["batch_stats"]
    # variables = freeze(variables)
    numerator = 2 * (preds * targets).sum(axis=(-2, -3))
    denominator = preds.sum(axis=(-2, -3)) + targets.sum(axis=(-2, -3))
    loss = 1 - jnp.mean((numerator + epsilon) / (denominator + epsilon))
    return loss


@jax.jit
def dice_loss_and_preds_bn(variables, state, inputs, targets, epsilon):
    """
    Calculates the dice loss for a batch of images and 
    returns the loss and preds.
    Args:
        variables: The segmentation models parameters.
        state: State of the semantic segmentation model.
        inputs: A batch of raw input images.
        targets: A batch of arrays with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        epsilon: A small number to prevent loss function from dividing by zero.
    Returns:
        The dice loss and preds from a batch of images.
    """
    forward_fn = state.apply_fn
    preds, mutated_vars = forward_fn(variables, inputs, mutable=['batch_stats'])
    # variables = unfreeze(variables)
    # variables["batch_stats"] = mutated_vars["batch_stats"]
    # variables = freeze(variables)
    numerator = 2 * (preds * targets).sum(axis=(-2, -3))
    denominator = preds.sum(axis=(-2, -3)) + targets.sum(axis=(-2, -3))
    loss = 1 - jnp.mean((numerator + epsilon) / (denominator + epsilon))
    return loss, preds