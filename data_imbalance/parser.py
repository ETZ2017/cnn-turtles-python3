import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="General configuration for micromind.")

    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate."
    )
    
    parser.add_argument(
        "--optimizer",
        dest="opt",
        default="adam",
        choices=["adam", "sgd", "adamW"],
        help="Optimizer name.",
    )
    parser.add_argument(
        "--experiment_name", default="exp", help="Name of the experiment."
    )
    parser.add_argument(
        "--output_folder", default="results", help="Output folder path."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode to check train and validation steps.",
    )
    parser.add_argument(
        "--epochs",
        default=5,
        type=int
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int
    )
    parser.add_argument(
        "--label_smoothing",
        default=0.3,
        type=float
    )
    parser.add_argument(
        "--l2_lambda",
        default=0.005,
        type=float
    )
    parser.add_argument(
        "--enable_aug_rhf",
        default=False,
        type=bool
    )
    parser.add_argument(
        "--enable_aug_rvf",
        default=False,
        type=bool
    )
    parser.add_argument(
        "--enable_aug_rr",
        default=False,
        type=bool
    )
    parser.add_argument(
        "--enable_ins_weights",
        default=False,
        type=bool
    )
    parser.add_argument(
        "--enable_root_weights",
        default=False,
        type=bool
    )
    parser.add_argument(
        "--enable_label_smoothing",
        default=False,
        type=bool
    )
    args = parser.parse_args()
    return args
