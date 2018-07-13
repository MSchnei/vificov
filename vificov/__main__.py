"""
Entry point.

References
----------
https://chriswarrick.com/blog/2014/09/15/python-apps-the-right-way-entry_points-and-scripts/

Notes
-----
Use config.csv to set analysis parameters.
"""

import os
import argparse
from vificov.vificov_main import run_vificov

# Get path of this file:
strDir = os.path.dirname(os.path.abspath(__file__))


def main():
    """vificov entry point."""

    # Create parser object:
    objParser = argparse.ArgumentParser()

    # Add argument to namespace - config file path:
    objParser.add_argument('-config',
                           metavar='config.csv',
                           help='Absolute file path of config file with \
                                 parameters for pRF analysis. Ignored if in \
                                 testing mode.'
                           )

    # Namespace object containign arguments and values:
    objNspc = objParser.parse_args()

    # Get path of config file from argument parser:
    strCsvCnfg = objNspc.config

    # Print info if no config argument is provided.
    if strCsvCnfg is None:
        print('Please provide the file path to a config file, e.g.:')
        print('   vificov -config /path/to/my_config_file.csv')

    else:

        # Call to main function, to invoke vificov analysis:
        run_vificov(strCsvCnfg)


if __name__ == "__main__":
    main()
