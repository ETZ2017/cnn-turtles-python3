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
        default=10,
        type=int
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int
    )
    parser.add_argument(
        "--label_smoothing",
        default=0.2,
        type=float
    )
    parser.add_argument(
        "--l2_lambda",
        default=0.01,
        type=float
    )
    
    args = parser.parse_args()
    return args
