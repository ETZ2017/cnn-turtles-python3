import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="General configuration for micromind.")

    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate."
    )
    parser.add_argument(
        "--experiment_name", default="exp", help="Name of the experiment."
    )
    parser.add_argument(
        "--output_folder", default="results", help="Output folder path."
    )
    parser.add_argument(
        "--epochs",
        default=10,
        type=int
    )
    parser.add_argument(
        "--batch_size",
        default=128,
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
    parser.add_argument(
        "--is_test",
        default=False,
        type=bool
    )
    
    args = parser.parse_args()
    return args
