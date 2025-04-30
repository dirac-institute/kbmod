"""A proof of concept program for generating fake data and results
to use for testing.
"""

import argparse
import logging
import numpy as np

from kbmod.fake_data.fake_data_creator import create_fake_times, FakeDataSet
from kbmod.results import Results


def execute(args):
    """Run the program from the given arguments.

    Parameters
    ----------
    args : `argparse.Namespace`
        The input arguments.
    """
    if args.verbose:
        print("KBMOD Fake Data and Results Generation:")
        for key, val in vars(args).items():
            print(f"  {key}: {val}")
        logging.basicConfig(level=logging.DEBUG)

    # Validate all the parameters.
    if args.num_times <= 0:
        raise ValueError(f"Invalid number of times: {args.num_times}. Must be >= 1.")
    if args.height <= 0:
        raise ValueError(f"Invalid image height: {args.height}. Must be >= 1.")
    if args.width <= 0:
        raise ValueError(f"Invalid image width: {args.width}. Must be >= 1.")
    if args.num_trjs < 0:
        raise ValueError(f"Invalid number of trajectories: {args.num_trjs}. Must be >= 0.")

    # Generate and save the fake data.
    times = create_fake_times(args.num_times)
    fake_ds = FakeDataSet(args.width, args.height, times)
    for id_x in range(args.num_trjs):
        fake_ds.insert_random_object(200.0)
    fake_ds.save_fake_data_to_work_unit(args.workunit)

    # Assign a reasonable (random) likelihood to each trajectory, build
    # a results table, and save it.
    for trj in fake_ds.trajectories:
        trj.lh = 5.0 + np.random.random() * 5.0
        trj.obs_count = args.num_times
    results = Results.from_trajectories(fake_ds.trajectories)
    results.write_table(args.results)


def main():
    parser = argparse.ArgumentParser(
        prog="kbmod-generate-fake-data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="A program to create a fake WorkUnit and Results data for testing purposes.",
    )
    parser.add_argument(
        "--workunit",
        default="./fake_wu.fits",
        dest="workunit",
        type=str,
        help="The file path for the generated workunit file with the image data.",
    )
    parser.add_argument(
        "--results",
        default="./fake_results.ecsv",
        dest="results",
        type=str,
        help="The file path for the generated results file.",
    )
    parser.add_argument(
        "--num_times",
        default=20,
        dest="num_times",
        type=int,
        help="The number of fake images (time steps) generated.",
    )
    parser.add_argument(
        "--height",
        default=400,
        dest="height",
        type=int,
        help="The height of the generated images (in pixels).",
    )
    parser.add_argument(
        "--width",
        default=400,
        dest="width",
        type=int,
        help="The width of the generated images (in pixels).",
    )    
    parser.add_argument(
        "--num_trjs",
        default=20,
        dest="num_trjs",
        type=int,
        help="The number of fake trajectories to insert into the data.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        dest="verbose",
        action="store_true",
        help="Output verbose status messages.",
    )

    # Run the actual program.
    args = parser.parse_args()
    execute(args)
