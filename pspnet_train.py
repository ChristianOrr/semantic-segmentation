import jax.numpy as jnp
from pspnet_model import _pspnet
import argparse
from jax.config import config
from train_functions import train


parser = argparse.ArgumentParser(description='Script for training PSPNet')
parser.add_argument("--dataset_name", help='Dataset name to load from huggingface hub', default="scene_parse_150", required=False)
parser.add_argument("--num_classes", help="Number of classes in the dataset", default=150, type=int, required=False)
parser.add_argument("--ignore_label", help="Label to ignore for training", default=0, type=int, required=False)
parser.add_argument("--weights_path", help='Path to save and load checkpoints', default="./checkpoints/pspnet/", required=False)
parser.add_argument("--init_function", 
                    help='Function for initializing the weights (not needed if restoring weights)' 
                    'Options include: "he_normal", "he_uniform", "xavier_normal", "xavier_uniform", "kumar_normal", "yilmaz_normal"',
                    default="yilmaz_normal", 
                    required=False)
parser.add_argument("--rng", help="Random number generator key", default=64, type=int, required=False)
parser.add_argument("--print_freq", help="Frequency for displaying training loss", default=1, type=int, required=False)
parser.add_argument("--loss_method", help='Loss calculation method. Options include: "dice" and "cross_entropy".', default="dice", required=False)
parser.add_argument("--reg_coeff", help="Weight decay L2 regularization coefficient.", default=1e-7, type=float, required=False)
parser.add_argument("--lr", help="Initial value for learning rate.", default=1e-4, type=float, required=False)
parser.add_argument("--min_lr", help="Minimum learning rate cap.", default=1e-7, type=float, required=False)
parser.add_argument("--decay", help="Exponential decay rate.", default=0.99, type=float, required=False)
parser.add_argument("--decay_start", help="The epoch to start decaying the learning rate.", default=100, type=int, required=False)
parser.add_argument("--decay_steps", help="Transition steps before next lr decay.", default=10, type=int, required=False)
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
parser.add_argument("--debugging", help="Runs in debugging mode.", action="store_true")
args = parser.parse_args()



def main(args):
    # Disable jit when debugging
    if args.debugging:
        config.update("jax_disable_jit", True)

    if args.float_precision == 32:
        dtype = jnp.float32
    elif args.float_precision == 16:
        dtype = jnp.float16
    else:
        raise NotImplementedError(f"Floating point precision {args.float_precision} is not supported.")

    # Create the model object
    pspnet = _pspnet(
        args.num_classes, 
        initializer=args.init_function, 
        use_bn=False,
        dtype=dtype
    )

    train(args, pspnet, dtype)


if __name__ == "__main__":
    main(args)
    