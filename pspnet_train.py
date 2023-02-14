import os
import numpy as np
import jax
from jax import value_and_grad
import jax.numpy as jnp
from flax.core.frozen_dict import freeze, unfreeze
import optax
from datasets import load_dataset
from flax.training import train_state, checkpoints
import augmax
import orbax.checkpoint as orbax
from pspnet_model import _pspnet
from loss_functions import dice_loss, dice_loss_and_preds
from preprocessing_functions import prep_data_batch, create_infinite_generator, grads_vanished_or_exploded
import nest_asyncio
nest_asyncio.apply()
import argparse



parser = argparse.ArgumentParser(description='Script for training PSPNet')
parser.add_argument("--dataset_name", help='Dataset name to load from huggingface hub', default="scene_parse_150", required=False)
parser.add_argument("--num_classes", help="Number of classes in the dataset", default=151, type=int, required=False)
parser.add_argument("--weights_path", help='Path to save and load checkpoints', default="./checkpoints/pspnet_adjusted/", required=False)
parser.add_argument("--init_function", 
                    help='Function for initializing the weights (not needed if restoring weights)' 
                    'Options include: "he_normal", "he_uniform", "xavier_normal", "xavier_uniform", "kumar_normal", "yilmaz_normal"',
                    default="yilmaz_normal", 
                    required=False)
parser.add_argument("--rng", help="Random number generator key", default=64, type=int, required=False)
parser.add_argument("--print_freq", help="Frequency for displaying training loss", default=1, type=int, required=False)
parser.add_argument("--lr", help="Initial value for learning rate.", default=0.0001, type=float, required=False)
parser.add_argument("--min_lr", help="Minimum learning rate cap.", default=0.000001, type=float, required=False)
parser.add_argument("--decay", help="Exponential decay rate.", default=0.99, type=float, required=False)
parser.add_argument("--decay_start", help="The epoch to start decaying the learning rate.", default=100, type=int, required=False)
parser.add_argument("--decay_steps", help="Transition steps before next lr decay.", default=100, type=int, required=False)
parser.add_argument("--height", help='Model image input height resolution', type=int, default=256)
parser.add_argument("--width", help='Model image input height resolution', type=int, default=256)
parser.add_argument("--batch_size", help='Batch size to use during training',type=int, default=1)
parser.add_argument("--float_precision", help='Floating point precision, Options are 32 and 16.',type=int, default=32)
parser.add_argument("--num_epochs", help='Number of training epochs', type=int, default=100000)
parser.add_argument("--save_freq", help='Model saving frequncy per steps', type=int, default=100)
parser.add_argument("--val_epochs", help='Frequency for running evaluation', type=int, default=500)
parser.add_argument("--val_batches", help='Number of batches to process per evaluation', type=int, default=5)
parser.add_argument("--dont_augment", help="Prevents augmentation on the RGB images.", action="store_true")
parser.add_argument("--dont_restore", help="Prevents restoring the latest checkpoint from the weights_path", action="store_true")
args = parser.parse_args()



def main(args):

    dataset = load_dataset(args.dataset_name)
    train_dataset, val_dataset  = dataset["train"], dataset["validation"]

    # Number of segmentation classes
    num_classes = args.num_classes
    # Image height and width
    height, width = args.height, args.width
    # Location for saving and loading checkpoints
    checkpoint_dir = args.weights_path
    os.makedirs(checkpoint_dir, exist_ok=True)

    if args.float_precision == 32:
        dtype = jnp.float32
    elif args.float_precision == 16:
        dtype = jnp.float16
    else:
        raise NotImplementedError(f"Floating point precision {args.float_precision} is not supported.")

    # Create the model object
    pspnet = _pspnet(
        num_classes, 
        initializer=args.init_function, 
        use_bn=False,
        dtype=dtype
    )
    # Display the model details
    dummy_x = jnp.array(train_dataset[0]["image"], dtype=dtype)
    # Downsample the image
    dummy_x = jax.image.resize(dummy_x, shape=(height, width, 3), method="bilinear")
    dummy_x = dummy_x[None, ...]
    rng_key = jax.random.PRNGKey(args.rng)

    augment = not args.dont_augment
    augment_images = jax.jit(augmax.Chain(
        augmax.RandomContrast(range=(0, 0.3), p=1.0),
        augmax.RandomBrightness(range=(-0.6, 0.6), p=1.0),
    ))


    # Number of training epochs
    epochs = args.num_epochs
    # Display training loss frequency
    print_freq = args.print_freq
    # Validation frequency
    val_epochs = args.val_epochs
    val_batches = args.val_batches
    # Checkpointing frequency
    save_steps = args.save_freq
    batch_size = args.batch_size
    # A small number to prevent loss function from dividing by zero
    epsilon = 0.0001
    # Exponential decay learning rate schedule defined by:
    # learning_rate * decay_rate ^ (global_step / decay_steps)
    schedule = optax.exponential_decay(
        init_value=args.lr,
        transition_steps=args.decay_steps,
        decay_rate=args.decay,
        transition_begin=args.decay_start,
        end_value=args.min_lr,
    )
    optimizer = optax.adam(learning_rate=schedule)

    # Restore latest checkpoint
    restore_latest = not args.dont_restore
    orbax_checkpointer = orbax.Checkpointer(orbax.PyTreeCheckpointHandler())

    variables = pspnet.init(rng_key, dummy_x)
    state = train_state.TrainState.create(
        apply_fn=pspnet.apply,
        params=variables['params'], 
        tx=optimizer,
    )

    if restore_latest:
        state = checkpoints.restore_checkpoint(
            ckpt_dir=checkpoint_dir, 
            target=state, 
            orbax_checkpointer=orbax_checkpointer
        )

    train_data_generator = create_infinite_generator(train_dataset)
    val_data_generator = create_infinite_generator(val_dataset)

    losses = []
    val_losses = []

    for epoch in range(epochs + 1):
        if epoch % print_freq == 0: print(f"Epoch: {epoch}")
        # Evaluate the models performance
        if epoch % val_epochs == 0:
            val_batch_losses = []
            for _ in range(val_batches):
                # Prepare a batch of data
                val_inputs = []
                # Loop until a valid batch is found
                while len(val_inputs) == 0:
                    val_inputs, val_targets = prep_data_batch(val_data_generator, batch_size, height, width, num_classes, dtype=jnp.float32)
                # Perform augmentation
                rng_key, subkey = jax.random.split(rng_key)
                if augment: val_inputs = augment_images(subkey, val_inputs)
                # Get loss and preds
                variables = freeze({"params": state.params})
                val_batch_loss, val_batch_preds = dice_loss_and_preds(variables, state, val_inputs, val_targets, epsilon)
                val_batch_losses.append(val_batch_loss)
            val_loss = jnp.array(val_batch_losses).mean()
            val_losses.append(val_loss)
            print(f"\t\tValidation Loss: {val_loss :.2f}")

        # Prepare a batch of data
        inputs = []
        # Loop until a valid batch is found
        while len(inputs) == 0:
            inputs, targets = prep_data_batch(train_data_generator, batch_size, height, width, num_classes, dtype=jnp.float32)
        # Perform augmentation
        rng_key, subkey = jax.random.split(rng_key)
        if augment: inputs = augment_images(subkey, inputs)
        # Perform backpropagation 
        variables = freeze({"params": state.params})
        loss, grads = value_and_grad(dice_loss, argnums=0)(variables, state, inputs, targets, epsilon)
        state = state.apply_gradients(grads=grads["params"])

        losses.append(loss)
        last_mean_loss = np.array(losses[-print_freq:]).mean()
        if epoch % print_freq == 0: print(f"\tLoss: {last_mean_loss :.2f} \tLearning Rate: {schedule(epoch) :.8f}")

        has_vanished,  has_exploded, mean_grads = grads_vanished_or_exploded(
            unfreeze(grads["params"]), 
            max_mean_grad=1e9, 
            min_mean_grad=1e-9
        )
        if has_vanished:
            print("Gradients have vanished, exiting training run now...")
            break
        elif has_exploded or jnp.isinf(mean_grads):
            print("Gradients have exploded, exiting training run now...")
            break
        elif jnp.isnan(mean_grads):
            print("Gradients are nan, exiting training run now...")
            break

        if epoch % save_steps == 0 and epoch != 0:
            checkpoints.save_checkpoint(
                ckpt_dir=checkpoint_dir, 
                target=state, 
                step=epoch,
                overwrite=True,
                keep=10,
                orbax_checkpointer=orbax_checkpointer
                )


if __name__ == "__main__":
    main(args)
    