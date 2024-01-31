"""A class for creating fake data sets.

The FakeDataSet class allows the user to create fake data sets
for testing, including generating images with random noise and
adding artificial objects. The fake data can be saved to files
or used directly.
"""

import os
import random
from pathlib import Path

from astropy.io import fits

from kbmod.configuration import SearchConfiguration
from kbmod.file_utils import *
from kbmod.search import *
from kbmod.wcs_utils import append_wcs_to_hdu_header, make_fake_wcs_info
from kbmod.work_unit import WorkUnit


def add_fake_object(img, x, y, flux, psf=None):
    """Add a fake object to a LayeredImage or RawImage

    Parameters
    ----------
    img : `RawImage` or `LayeredImage`
        The image to modify.
    x : `float`
        The x pixel location of the fake object.
    y : `float`
        The y pixel location of the fake object.
    flux : `float`
        The flux value.
    psf : `PSF`
        The PSF for the image.
    """
    if type(img) is LayeredImage:
        sci = img.get_science()
    else:
        sci = img

    if psf is None:
        sci.interpolated_add(x, y, flux)
    else:
        dim = psf.get_dim()
        initial_x = x - psf.get_radius()
        initial_y = y - psf.get_radius()
        for i in range(dim):
            for j in range(dim):
                sci.interpolated_add(float(initial_x + i), float(initial_y + j), flux * psf.get_value(i, j))


class FakeDataSet:
    """This class creates fake data sets for testing and demo notebooks."""

    def __init__(self, width, height, num_times, noise_level=2.0, psf_val=0.5, obs_per_day=3, use_seed=False):
        """The constructor.

        Parameters
        ----------
        width : `int`
            The width of the images in pixels.
        height : `int`
            The height of the images in pixels.
        num_times : `int`
            The number of time steps (number of images).
        noise_level : `float`
            The level of the background noise.
        psf_val : `float`
            The value of the default PSF.
        obs_per_day : `int`
            The number of observations on the same night.
        use_seed : `bool`
            Use a deterministic seed to avoid flaky tests.
        """
        self.width = width
        self.height = height
        self.psf_val = psf_val
        self.noise_level = noise_level
        self.num_times = num_times
        self.use_seed = use_seed
        self.trajectories = []

        # Generate times with multiple observations per night
        # separated by ~15 minutes.
        self.times = []
        seen_on_day = 0
        day_num = 0
        for i in range(num_times):
            t = 57130.2 + day_num + seen_on_day * 0.01
            self.times.append(t)

            seen_on_day += 1
            if seen_on_day == obs_per_day:
                seen_on_day = 0
                day_num += 1

        # Make the image stack.
        self.stack = self.make_fake_ImageStack()

    def make_fake_ImageStack(self):
        """Make a stack of fake layered images.

        Returns
        -------
        stack : `ImageStack`
        """
        p = PSF(self.psf_val)

        image_list = []
        for i in range(self.num_times):
            img = LayeredImage(
                self.width,
                self.height,
                self.noise_level,
                self.noise_level**2,
                self.times[i],
                p,
                i if self.use_seed else -1,
            )
            image_list.append(img)

        stack = ImageStack(image_list)
        return stack

    def insert_object(self, trj):
        """Insert a fake object given the trajectory.

        Parameters
        ----------
        trj : `Trajectory`
            The trajectory of the fake object to insert.
        """
        t0 = self.times[0]

        for i in range(self.num_times):
            dt = self.times[i] - t0
            px = trj.x + dt * trj.vx + 0.5
            py = trj.y + dt * trj.vy + 0.5

            # Get the image for the timestep, add the object, and
            # re-set the image. This last step needs to be done
            # explicitly because of how pybind handles references.
            current = self.stack.get_single_image(i)
            add_fake_object(current, px, py, trj.flux, current.get_psf())

        # Save the trajectory into the internal list.
        self.trajectories.append(trj)

    def insert_random_object(self, flux):
        """Create a fake object and insert it into the image.

        Parameters
        ----------
        flux : `float`
            The flux of the object.

        Returns
        -------
        t : `Trajectory`
            The trajectory of the inserted object.
        """
        dt = self.times[-1] - self.times[0]

        # Create the random trajectory.
        t = Trajectory()
        t.x = int(random.random() * self.width)
        xe = int(random.random() * self.width)
        t.vx = (xe - t.x) / dt
        t.y = int(random.random() * self.height)
        ye = int(random.random() * self.height)
        t.vy = (ye - t.y) / dt
        t.flux = flux

        # Insert the object.
        self.insert_object(t)

        return t

    def save_fake_data_to_dir(self, data_dir):
        """Create the fake data in a given directory.

        Parameters
        ----------
        data_dir : `str`
            The path of the directory for the fake data.
        """
        # Make the subdirectory if needed.
        dir_path = Path(data_dir)
        if not dir_path.is_dir():
            print("Directory '%s' does not exist. Creating." % data_dir)
            os.mkdir(data_dir)

        # Save each of the image files.
        for i in range(self.stack.img_count()):
            img = self.stack.get_single_image(i)
            filename = os.path.join(data_dir, ("%06i.fits" % i))

            # If the file already exists, delete it.
            if Path(filename).exists():
                os.remove(filename)

            # Save the file.
            img.save_layers(filename)

            # Open the file and insert fake WCS data.
            wcs_dict = make_fake_wcs_info(200.6145, -7.7888, 2000, 4000)
            hdul = fits.open(filename)
            append_wcs_to_hdu_header(wcs_dict, hdul[1].header)
            hdul.writeto(filename, overwrite=True)
            hdul.close()

    def save_time_file(self, file_name):
        """Save the mapping of visit ID -> timestamp to a file.

        Parameters
        ----------
        file_name : `str`
            The file name for the timestamp file.
        """
        mapping = {}
        for i in range(self.num_times):
            id_str = "%06i" % i
            mapping[id_str] = self.times[i]
        FileUtils.save_time_dictionary(file_name, mapping)

    def delete_fake_data_dir(self, data_dir):
        """Remove the fake data in a given directory.

        Parameters
        ----------
        data_dir : `str`
            The path of the directory for the fake data.
        """
        for i in range(self.stack.img_count()):
            img = self.stack.get_single_image(i)
            filename = os.path.join(data_dir, ("%06i.fits" % i))
            if Path(filename).exists():
                os.remove(filename)

    def save_fake_data_to_work_unit(self, filename, config=None):
        """Create the fake data in a WorkUnit file.

        Parameters
        ----------
        filename : `str`
            The name of the resulting WorkUnit file.
        config : `SearchConfiguration`, optional
            The configuration parameters to use. If None then uses
            default values.
        """
        if config is None:
            config = SearchConfiguration()
        work = WorkUnit(im_stack=self.stack, config=config)
        work.to_fits(filename)
