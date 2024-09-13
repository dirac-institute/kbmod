"""A helper function for creating data used in the demo notebook
and some of the tests.
"""

import os

from kbmod.configuration import SearchConfiguration
from kbmod.fake_data.fake_data_creator import *
from kbmod.search import *

def make_demo_data(filename=None):
    """Make the fake demo data.

    Parameters
    ----------
    filename : `str`
        The path and file anem to store the demo data. If ``None`` then
        does not save the demo data.

    Returns
    -------
    work : `WorkUnit`
        A WorkUnit with the fake data.
    """
    # Set the characteristics of the fake data.
    img_width = 256
    img_height = 256
    num_times = 20

    # Create the fake images
    fake_times = create_fake_times(num_times, t0=57130.2)
    ds = FakeDataSet(img_width, img_height, fake_times)

    # Insert a fake object with only horizontal velocity.
    trj = Trajectory(x=50, y=40, vx=10, vy=0, flux=500)
    ds.insert_object(trj)

    # Create configuraiton settings that match the object inserted.
    settings = {
        # Override the search data to match the known object.
        "generator_config": {
            "name": "EclipticSearch",
            "vel_steps": 21,
            "min_vel": 0.0,
            "max_vel": 20.0,
            "ang_steps": 11,
            "min_ang_offset": -0.5,
            "max_ang_offset": 0.5,
            "angle_units": "radians",
            "force_ecliptic": 0.0,
        },
        # Loosen the other filtering parameters.
        "clip_negative": True,
        "sigmaG_lims": [15, 60],
        "mom_lims": [37.5, 37.5, 1.5, 1.0, 1.0],
        "peak_offset": [3.0, 3.0],
    }
    config = SearchConfiguration.from_dict(settings)

    # Create a WorkUnit and save it if needed.
    work = WorkUnit(im_stack=ds.stack, config=config, wcs=ds.fake_wcs)
    if filename is not None:
        work.to_fits(filename)
    return work