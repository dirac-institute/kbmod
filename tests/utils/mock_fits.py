import abc
import warnings
import tarfile

import numpy as np
from astropy.table import Table, MaskedColumn
from astropy.utils.exceptions import AstropyUserWarning
from astropy.io.fits import HDUList, PrimaryHDU, CompImageHDU, BinTableHDU, Column

from .utils_for_tests import get_absolute_data_path


__all__ = [
    "DECamImdiffFactory",
    "MockFitsFileFactory",
]


class MockFitsFileFactory(abc.ABC):
    """Mocks a collection of FITS file given a prescription.

    To mock a FITS file, as if it were opened by AstroPy, we require the raw
    header data and a simple description of the layout of the FITS headers. For
    a single file this consists of the ``filename``, ``hdu`` index, header
    ``keyword``s, header keyword ``values``; mimicking the output format of
    AstroPy's ``fitsheader`` tool not accounting for the addition of the '
    ``format`` column.

    To dump the raw data use the `dump_headers.py` script:

        python dump_header -t ascii.ecsv *fits > output_file.ecsv

    duplicating the invocation of AstroPy's ``fitsheader`` tool. That file now
    contains the described columns, with the filename keys repeated for each
    HDU in the FITS file, and the HDU index repeated for each keyword in the
    header of that HDU.

    This class then reads, groups, casts and builds the HDUList object as if
    the file was opened by `astropy.io.fits.open` function.

    To use the class, you must inherit from it and specify the property,
    `hdu_types` which should, possibly construct and, return a list of HDU
    classes (f.e. `PrimaryHDU`, `CompImageHDU`, `BinTableHDU` etc.). The length
    of the list must match the length of the HDU's within each HDUList in the
    dumped data.

    This class does not construct the ``.data`` attribute.

    Parameters
    ----------
    archive_name : `str`
       Name of the TAR archive in ``data`` dir containing raw data.
    fname : `str`
       Filename within the archive that contains the wanted raw data.
    compression : `str`, optional
       Compression used to compress the archive. Assumed to be ``bz2``.
    format : `str`, optional
       Format in which the data was dumped. One of AstroPy's supported
       ``io`` formats. Defaults to ``ecsv``.
    """

    # will almost never be anything else. Rather, it would be a miracle if it
    # were something else, since FITS standard shouldn't allow it. Further
    # casting by some packages will always be casting implemented in terms of
    # parsing these types.
    lexical_type_map = {
        "int": int,
        "str": str,
        "float": float,
        "bool": bool,
    }
    """Map between type names and types themselves."""

    def __init__(self, archive_name, fname, compression="bz2", format="ascii.ecsv"):
        archive_path = get_absolute_data_path(archive_name)
        with tarfile.open(archive_path, f"r:{compression}") as archive:
            tblfile = archive.extractfile(fname)
            table = Table.read(tblfile.read().decode(), format=format)
            # sometimes empty strings get serialized as masked, to cover that
            # eventuality we'll just substitute an empty string
            if isinstance(table["value"], MaskedColumn):
                table["value"].fill_value = ""
                table["value"] = table["value"].filled()

        # Create HDU groups for easier iteration
        self.table = table.group_by("filename")
        self.n_files = len(self.table.groups)

        # Internal counter for the current fits index,
        # so that we may auto-increment it and avoid returning
        # the same data all the time.
        self._current = 0

    @abc.abstractproperty
    def hdu_types(self):
        """A list of HDU types for each HDU in the HDUList"""
        # index and type of HDU map
        raise NotImplementedError()

    def spoof_data(self, hdul, **kwargs):
        """Write fits(es) to file. See HDUList.writeto"""
        raise NotImplementedError("This mock factory did not implement data spoofing.")

    # good for debugging, leave it
    def get_item(self, group_idx, key, hdu_idx=None):
        """Get (key, value, hdu_idx) triplet from the raw data grouped by
        filename and, optionally, also grouped by HDU index."""
        group = self.table.groups[group_idx]
        mask = group["keyword"] == key
        if hdu_idx is not None:
            mask = mask & group["hdu"] == hdu_idx
        return (key, group[mask]["value"], group[mask]["hdu"])

    def get_value(self, group_idx, key, hdu_idx=None):
        """Get (value, hdu_idx) from the raw data grouped by filename and,
        optionally, also grouped by HDU index."""
        return self.get_item(group_idx, key, hdu_idx)[1:]

    def lexical_cast(self, value, format):
        """Cast str literal of a type to the type itself. Supports just
        the builtin Python types.
        """
        if format in self.lexical_type_map:
            return self.lexical_type_map[format](value)
        return value

    def get_fits(self, fits_idx, spoof_data=False):
        """Create a FITS file using the raw data selected by the `fits_idx`.

        **IMPORTANT**: MockFactories guarantee to return Headers that match the
        original header raw data. Spoofed data is, however, not guaranteed to
        respect the original data dimensions and size, just the data layout.
        The practical implication of this is that, for example:
        header["NAXIS1"] (or 2) does not match hdu.data.shape[0] (or 1); or
        that writing the file to disk and reading it again is not guaranteed to
        roundtrip the header data anymore, even with output_verify set to
        ``ignore``:

        hdu = get_fits(0)
        hdu.writeto("test.fits", output_verify="ignore")
        hdu2 = fitsio.open("test.fits", output_verify="ignore")
        hdu2 == hdu --> False

        The change usually affects NAXIS1/2 cards for all HDU types, but could
        also alter PGCOUNT, GCOUNT, TFIELDS, NAXIS as well as endianness.
        """
        hdul = HDUList()
        file_group = self.table.groups[fits_idx % self.n_files]
        hdu_group = file_group.group_by("hdu")

        # nearly every following command will be issuing warnings, but they
        # are not important - all are HIERARCH card creation warnings for keys
        # that are longer than 8 characters.
        warnings.filterwarnings("ignore", category=AstropyUserWarning)
        for hdr_idx, HDUClass in enumerate(self.hdu_types):
            hdr = HDUClass()
            for k, v, f in hdu_group.groups[hdr_idx]["keyword", "value", "format"]:
                hdr.header[k] = self.lexical_cast(v, f)
            hdul.append(hdr)
        warnings.resetwarnings()

        if spoof_data:
            hdul = self.spoof_data(hdul)

        return hdul

    def get_range(self, start_idx, end_idx, spoof_data=False):
        """Get a list of HDUList objects from the specified range.
        When range exceeds the number of available serialized headers it's
        wrapped back to start.

        Does not update the current index counter.
        """
        if not (start_idx < end_idx):
            raise ValueError(
                "Expected starting index to be smaller than the "
                f"ending index. Got start={start_idx}, end={end_idx}"
            )
        files = []
        for i in range(start_idx, end_idx):
            files.append(self.get_fits(i % self.n_files, spoof_data))
        return files

    def get_n(self, n, spoof_data=False):
        """Get next n fits files. Wraps around when available `n_files`
        is exceeded. Updates the current index counter."""
        files = self.get_range(self._current, self._current + n, spoof_data)
        self._current = (self._current + n) % self.n_files
        return files

    def mock_fits(self, spoof_data=False):
        """Return new mocked FITS.

        Raw data is read sequentially, once exhausted it's reset and starts
        over again.
        """
        self._current = (self._current + 1) % self.n_files
        return self.get_fits(self._current, spoof_data)


class DECamImdiffFactory(MockFitsFileFactory):
    """Mocks a Vera C. Rubin Science Pipelines Imdiff Data Product, as it was
    produced by the Science Pipelines and procedure described in
    arXiv:2109.03296

    The FITS file contained 16 HDUs, one PRIMARY, 3 images (image, mask and
    variance) and various supporting data such as PSF, ArchiveId etc. stored
    in `BinTableHDU`s.

    The raw data was exported from DEEP B1a field, as described in
    arXiv:2310.03678.

    The raw data for this factory can be found in the
    ``data/decam_imdif_headers.tar.bz2`` file, which approximately, contains
    real header data from 60-odd different FITS files produced by Rubin Science
    Pipelines. The used files contained ``imdiff`` data product.

    Examples
    --------
    >>>> fitsFactory = DECamImdiffFactory()

    Get the next FITS file from the list. Once all files are exhausted
    `mock_fits` will start from the beginning again.

    >>>> fitsFactory.mock_fits()
    [astropy.io.fits.hdu.image.... ]

    Get a particular file from from the list of all files. In this example the
    zeroth file in the list. Useful repeatability and predictability of the
    results is wanted:

    >>>> fitsFactory.get_fits(fits_idx=0)
    """

    def __init__(self, archive_name="decam_imdiff_headers.ecsv.tar.bz2", fname="decam_imdiff_headers.ecsv"):
        super().__init__(archive_name, fname)

        # don't let Black have its way with these lines because it's a massacre
        # fmt: off
        self.hdus = [PrimaryHDU, ]
        self.hdus.extend([CompImageHDU, ] * 3)
        self.hdus.extend([BinTableHDU, ] * 12)
        # fmt: on

    @property
    def hdu_types(self):
        return self.hdus

    def spoof_data(self, hdul):
        # Mocking FITS files is hard. The data is, usually, well described by
        # the header, to the point where it's possible to construct byte
        # offsets and memmap the fits, like FITS readers usually do. Spoofing
        # the data attribute requires us to respect the data type, size and
        # shape. It's of course silly to have to write, f.e., a 10 by 2k
        # BinTable because the header says so. We get by mostly because our
        # Standardizers don't need to read them, so AstroPy never raises an
        # error. Eventually it is possible, that one of them could want to read
        # the PSF HDU, which we will then have to spoof correctly.
        # On top of that AstroPy will, rightfully, be very opinionated about
        # checking metadata of the data matches the data itself and will throw
        # warnings and errors. We have to match data type using dtypes because
        # they default to 64bit representation in pure Python as well as number
        # of columns and rows - even if we leave the HDU data itself empty!
        # Of course, storing data takes too much space, so now we have to reverse
        # engineer the data, on a per-standardizer level, taking care we don't
        # hit any of these roadblocks.
        empty_array = np.zeros((5, 5), np.float32)
        hdul["IMAGE"].data = empty_array
        hdul["VARIANCE"].data = empty_array
        hdul["MASK"].data = empty_array.astype(np.int32)

        # These are the 12 BinTableHDUs we're not using atm
        for i, hdu in enumerate(hdul[4:]):
            nrows = hdu.header["TFIELDS"]
            hdul[i + 4] = BinTableHDU(data=np.zeros((nrows,), dtype=hdu.data.dtype), header=hdu.header)

        return hdul
