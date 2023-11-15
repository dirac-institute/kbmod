import abc
import warnings
import tarfile

from astropy.table import Table, MaskedColumn
from astropy.utils.exceptions import AstropyUserWarning
from astropy.io.fits import (HDUList,
                             PrimaryHDU,
                             CompImageHDU,
                             BinTableHDU)

from .utils_for_tests import get_absolute_data_path


__all__ = ["DECamImdiffFactory", "MockFitsFileFactory", ]


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
       Name of the TAR archive in ``data`` dir ontaining raw data.
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

        # internal counter, start with -1 so we skip
        # in if in the `mock_fits`
        self._current = -1

    @abc.abstractproperty
    def hdu_types(self):
        """A list of HDU types for each HDU in the HDUList """
        # index and type of HDU map
        raise NotImplementedError()

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
        """Cast literal text form of a type to the type itself. Supports just
        the builtin Python types.
        """
        if format in self.lexical_type_map:
            return self.lexical_type_map[format](value)
        return value

    def create_fits(self, fits_idx):
        """Create a FITS file using the raw data selected by the `fits_idx`."""
        hdul = HDUList()
        file_group = self.table.groups[fits_idx]
        hdu_group = file_group.group_by("hdu")

        # nearly every following command will be issuing warnings, but they
        # are not important - all are HIERARCH card creation warnings for keys
        # that are longer than 8 characters.
        warnings.filterwarnings('ignore', category=AstropyUserWarning)
        for hdr_idx, HDUClass in enumerate(self.hdu_types):
            hdr = HDUClass()
            for k, v, f in hdu_group.groups[hdr_idx]["keyword", "value", "format"]:
                hdr.header[k] = self.lexical_cast(v, f)
            hdul.append(hdr)
        warnings.resetwarnings()

        return hdul

    def mock_fits(self):
        """Return new mocked FITS.

        Raw data is read sequentially, once exhausted it's reset and starts
        over again.
        """
        if self._current > self.n_files:
            self._current = -1
        self._current += 1
        return self.create_fits(self._current)


class DECamImdiffFactory(MockFitsFileFactory):
    """Mocks a Vera C. Rubin Science Pipelines Imdiff Data Product, as it was
    produced by the Science Pipelines and procedure described in
    arXiv:2109.03296

    The FITS file contained 16 HDUs, one PRIMARY, 3 images (image, mask and
    variance) and various supporting data such as PSF, ArchiveId etc. stored
    in `BinTableHDU`s.

    The raw data was exported from DEEP B1a field, as described in
    arXiv:2310.03678.
    """
    def __init__(self,
                 archive_name="decam_imdiff_headers.ecsv.tar.bz2",
                 fname="decam_imdiff_headers.ecsv"):
        super().__init__(archive_name, fname)

        self.hdus = [PrimaryHDU, ]
        self.hdus.extend([CompImageHDU, ]*3)
        self.hdus.extend([BinTableHDU, ]*12)

    @property
    def hdu_types(self):
        return self.hdus
