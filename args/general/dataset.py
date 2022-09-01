from argparse import ArgumentParser


def dataset_args(parser: ArgumentParser):
    """Adds dataset-related arguments to a parser.

    Args:
        parser (ArgumentParser): parser to add dataset args to.
    """

    parser.add_argument("--dataset", default='', type=str)
    parser.add_argument("--dataset_path", default='', type=str)
    parser.add_argument("--batch_size", default=128, type=int, help="number of data samples in the mini_batch")
    parser.add_argument("--dali", action="store_true", help="use DALI")


def augmentations_args(parser: ArgumentParser):
    """Adds augmentation-related arguments to a parser.

    Args:
        parser (ArgumentParser): parser to add augmentation args to.
    """

    # cropping
    parser.add_argument("--multicrop", action="store_true")
    parser.add_argument("--num_crops", type=int, default=2)
    parser.add_argument("--num_small_crops", type=int, default=0)

    # augmentations
    parser.add_argument("--brightness", type=float, required=True, nargs="+")
    parser.add_argument("--contrast", type=float, required=True, nargs="+")
    parser.add_argument("--saturation", type=float, required=True, nargs="+")
    parser.add_argument("--hue", type=float, required=True, nargs="+")
    parser.add_argument("--gaussian_prob", type=float, default=[0.5], nargs="+")
    parser.add_argument("--solarization_prob", type=float, default=[0.0], nargs="+")
    parser.add_argument("--min_scale", type=float, default=[0.08], nargs="+")

    # for imagenet or custom dataset
    parser.add_argument("--size", type=int, default=[224], nargs="+")

    # for custom dataset
    parser.add_argument("--mean", type=float, default=[0.485, 0.456, 0.406], nargs="+")
    parser.add_argument("--std", type=float, default=[0.228, 0.224, 0.225], nargs="+")

    # debug
    parser.add_argument("--debug_augmentations", action="store_true")
