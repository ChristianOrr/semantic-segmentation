import os
import json
import numpy as np
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import freeze, unfreeze
import optax
from datasets import load_dataset
from flax.training import train_state, checkpoints
import augmax
import orbax.checkpoint as orbax
from loss_functions import loss_fn
from preprocessing_functions import prep_data_batch, create_infinite_generator, grads_vanished_or_exploded, dict_mean
from miou_metrics import compute_metrics
import nest_asyncio
nest_asyncio.apply()




def train(args, model, dtype):

    dataset = load_dataset(args.dataset_name)
    train_dataset, val_dataset  = dataset["train"], dataset["validation"]

    # Number of segmentation classes
    num_classes = args.num_classes
    # Image height and width
    height, width = args.height, args.width
    # Location for saving and loading checkpoints
    checkpoint_dir = args.weights_path
    os.makedirs(checkpoint_dir, exist_ok=True)

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
    # Background class
    ignore_label = args.ignore_label
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

    variables = model.init(rng_key, dummy_x)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
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

    for epoch in range(epochs + 1):
        if epoch % print_freq == 0: print(f"Epoch: {epoch}")
        # Evaluate the models performance
        if epoch % val_epochs == 0:
            eval_batch_metrics = []
            for _ in range(val_batches):
                val_inputs = []
                # Loop until a valid batch is found
                while len(val_inputs) == 0:
                    val_inputs, val_targets = prep_data_batch(val_data_generator, batch_size, height, width, dtype=jnp.float32)
                # Perform augmentation
                rng_key, subkey = jax.random.split(rng_key)
                if augment: val_inputs = augment_images(subkey, val_inputs)
                # Get loss and preds
                variables = freeze({"params": state.params})
                val_loss, val_logits = loss_fn(variables, state, val_inputs, val_targets, num_classes, args.loss_method, ignore_label)

                eval_metrics = compute_metrics(val_logits, val_targets, num_classes, ignore_label)
                eval_metrics["loss"] = val_loss
                eval_batch_metrics.append(eval_metrics)

            eval_batch_metrics = dict_mean(eval_batch_metrics)
            print("Validation Metrics:")
            print(json.dumps(eval_batch_metrics, indent=4))

        inputs = []
        # Loop until a valid batch is found
        while len(inputs) == 0:
            inputs, targets = prep_data_batch(train_data_generator, batch_size, height, width, dtype=jnp.float32)
        # Perform augmentation
        rng_key, subkey = jax.random.split(rng_key)
        if augment: inputs = augment_images(subkey, inputs)
        # Perform backpropagation 
        variables = freeze({"params": state.params})
        grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
        (loss, logits), grads = grad_fn(variables, state, inputs, targets, num_classes, args.loss_method, ignore_label)
        state = state.apply_gradients(grads=grads["params"])

        losses.append(loss)
        last_mean_loss = np.array(losses[-print_freq:]).mean()
        if epoch % print_freq == 0: print(f"\tLoss: {last_mean_loss :.4f} \tLearning Rate: {schedule(epoch) :.8f}")

        has_vanished,  has_exploded, mean_grads = grads_vanished_or_exploded(
            grads["params"], 
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