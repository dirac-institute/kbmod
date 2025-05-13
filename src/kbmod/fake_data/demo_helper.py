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
            "name": "EclipticCenteredSearch",
            "velocities": [0, 20.0, 21],
            "angles": [-0.5, 0.5, 11],
            "angle_units": "radian",
            "given_ecliptic": 0.0,
        },
        # Loosen the other filtering parameters.
        "clip_negative": True,
        "sigmaG_lims": [15, 60],
    }
    config = SearchConfiguration.from_dict(settings)

    # Create a WorkUnit and save it if needed.
    work = WorkUnit(im_stack=ds.stack_py, config=config, wcs=ds.fake_wcs)
    if filename is not None:
        work.to_fits(filename)
    return work