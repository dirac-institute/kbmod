# Modified from the original Astropy code to add a card format to the
# tabular output. All rights belong to the original authors.

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Modified Astropy ``archiveheaders`` utility that adds a datatype column for each
header card to the output. All rights belong to the original authors.

``archiveheaders`` is a command line script based on astropy.io.fits for printing
the header(s) of one or more FITS file(s) to the standard output in a human-
readable format.

The modifications to the script include:
- supporting only tabular output
- making "ascii.ecsv" the default output format
- appending a datatype to each serialized header card, describing the type of
  the cards value.

Example uses of ``archiveheaders``:

1. Print the header of all the HDUs of a .fits file::

    $ archiveheaders filename.fits

2. Dump the header keywords of all the files in the current directory into a
   machine-readable ecsv file::

    $ archiveheaders *.fits > keywords.csv

3. Sorting the output along a specified keyword::

    $ archiveheaders -f -s DATE-OBS *.fits

4. Sort first by OBJECT, then DATE-OBS::

    $ archiveheaders -f -s OBJECT -s DATE-OBS *.fits

Note that compressed images (HDUs of type
:class:`~astropy.io.fits.CompImageHDU`) really have two headers: a real
BINTABLE header to describe the compressed data, and a fake IMAGE header
representing the image that was compressed. Astropy returns the latter by
default. You must supply the ``--compressed`` option if you require the real
header that describes the compression.

With Astropy installed, please run ``archiveheaders --help`` to see the full usage
documentation.
"""

import argparse
import sys

import numpy as np

from astropy import __version__, log
from astropy.io import fits

DESCRIPTION = """
Print the header(s) of a FITS file. Optional arguments allow the desired
extension(s), keyword(s), and output format to be specified.
Note that in the case of a compressed image, the decompressed header is
shown by default.

This script is part of the Astropy package. See
https://docs.astropy.org/en/latest/io/fits/usage/scripts.html#module-astropy.io.fits.scripts.fitsheader
for further documentation.
""".strip()


class ExtensionNotFoundException(Exception):
    """Raised if an HDU extension requested by the user does not exist."""


class HeaderFormatter:
    """Class to format the header(s) of a FITS file for display by the
    `fitsheader` tool; essentially a wrapper around a `HDUList` object.

    Example usage:
    fmt = HeaderFormatter('/path/to/file.fits')
    print(fmt.parse())

    Parameters
    ----------
    filename : str
        Path to a single FITS file.
    verbose : bool
        Verbose flag, to show more information about missing extensions,
        keywords, etc.

    Raises
    ------
    OSError
        If `filename` does not exist or cannot be read.
    """

    def __init__(self, filename, verbose=True):
        self.filename = filename
        self.verbose = verbose
        self._hdulist = fits.open(filename)

    def parse(self, compressed=False):
        """Returns the FITS file header(s) in a readable format.

        Parameters
        ----------
        compressed : bool, optional
            If True, shows the header describing the compression, rather than
            the header obtained after decompression. (Affects FITS files
            containing `CompImageHDU` extensions only.)

        Returns
        -------
        formatted_header : str or astropy.table.Table
            Traditional 80-char wide format in the case of `HeaderFormatter`;
            an Astropy Table object in the case of `TableHeaderFormatter`.
        """
        hdukeys = range(len(self._hdulist))  # Display all by default
        return self._parse_internal(hdukeys, compressed)

    def _parse_internal(self, hdukeys, compressed):
        """The meat of the formatting; in a separate method to allow overriding."""
        result = []
        for idx, hdu in enumerate(hdukeys):
            try:
                cards = self._get_cards(hdu, compressed)
            except ExtensionNotFoundException:
                continue

            if idx > 0:  # Separate HDUs by a blank line
                result.append("\n")
            result.append(f"# HDU {hdu} in {self.filename}:\n")
            for c in cards:
                result.append(f"{c}\n")
        return "".join(result)

    def _get_cards(self, hdukey, compressed):
        """Returns a list of `astropy.io.fits.card.Card` objects.

        This function will return the desired header cards, taking into
        account the user's preference to see the compressed or uncompressed
        version.

        Parameters
        ----------
        hdukey : int or str
            Key of a single HDU in the HDUList.

        compressed : bool, optional
            If True, shows the header describing the compression.

        Raises
        ------
        ExtensionNotFoundException
            If the hdukey does not correspond to an extension.
        """
        # First we obtain the desired header
        try:
            if compressed:
                # In the case of a compressed image, return the header before
                # decompression (not the default behavior)
                header = self._hdulist[hdukey]._header
            else:
                header = self._hdulist[hdukey].header
        except (IndexError, KeyError):
            message = f"{self.filename}: Extension {hdukey} not found."
            if self.verbose:
                log.warning(message)
            raise ExtensionNotFoundException(message)

        # return all cards
        cards = header.cards
        return cards

    def close(self):
        self._hdulist.close()


class TableHeaderFormatter(HeaderFormatter):
    """Class to convert the header(s) of a FITS file into a Table object.
    The table returned by the `parse` method will contain four columns:
    filename, hdu, keyword, and value.

    Subclassed from HeaderFormatter, which contains the meat of the formatting.
    """

    def _parse_internal(self, hdukeys, compressed):
        """Method called by the parse method in the parent class."""
        tablerows = []
        for hdu in hdukeys:
            try:
                for card in self._get_cards(hdu, compressed):
                    tablerows.append(
                        {
                            "filename": self.filename,
                            "hdu": hdu,
                            "keyword": card.keyword,
                            "value": str(card.value),
                            "format": type(card.value).__name__,
                        }
                    )
            except ExtensionNotFoundException:
                pass

        if tablerows:
            from astropy import table

            return table.Table(tablerows)
        return None


def print_headers_as_table(args):
    """Prints FITS header(s) in a machine-readable table format.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments passed from the command-line as defined below.
    """
    tables = []
    # Create a Table object for each file
    for filename in args.filename:  # Support wildcards
        formatter = None
        try:
            formatter = TableHeaderFormatter(filename)
            tbl = formatter.parse(args.compressed)
            if tbl:
                tables.append(tbl)
        except OSError as e:
            log.error(str(e))  # file not found or unreadable
        finally:
            if formatter:
                formatter.close()

    # Concatenate the tables
    if len(tables) == 0:
        return False
    elif len(tables) == 1:
        resulting_table = tables[0]
    else:
        from astropy import table

        resulting_table = table.vstack(tables)
    # Print the string representation of the concatenated table
    resulting_table.write(sys.stdout, format=args.table)


def main(args=None):
    """This is the main function called by the `fitsheader` script."""
    parser = argparse.ArgumentParser(
        description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "-t",
        "--table",
        nargs="?",
        default="ascii.ecsv",
        metavar="FORMAT",
        help=(
            '"The default format is "ascii.ecsv" (can be "ascii.csv", "ascii.html", '
            '"ascii.latex", "fits", etc)'
        ),
    )
    mode_group.add_argument(
        "-f",
        "--fitsort",
        action="store_true",
        help=("print the headers as a table with each unique keyword in a given column (fitsort format) "),
    )
    parser.add_argument(
        "-s",
        "--sort",
        metavar="SORT_KEYWORD",
        action="append",
        type=str,
        help=(
            "sort output by the specified header keywords, can be repeated to "
            "sort by multiple keywords; Only supported with -f/--fitsort"
        ),
    )
    parser.add_argument(
        "-c",
        "--compressed",
        action="store_true",
        help=(
            "for compressed image data, show the true header which describes "
            "the compression rather than the data"
        ),
    )
    parser.add_argument(
        "filename",
        nargs="+",
        help="path to one or more files; wildcards are supported",
    )
    args = parser.parse_args()

    if args.sort:
        args.sort = [key.replace(".", " ") for key in args.sort]
        if not args.fitsort:
            log.error("Sorting with -s/--sort is only supported in conjunction with" " -f/--fitsort")
            # 2: Unix error convention for command line syntax
            sys.exit(2)

    # Now print the desired headers
    try:
        print_headers_as_table(args)
    except OSError:
        # A 'Broken pipe' OSError may occur when stdout is closed prematurely,
        # eg. when calling `fitsheader file.fits | head`. We let this pass.
        pass
