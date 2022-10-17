from astropy.coordinates import SkyCoord, ICRS, solar_system_ephemeris, get_body_barycentric, EarthLocation

from astropy import units as u
from tqdm import tqdm
import numba
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from scipy.ndimage import shift
import scipy
from scipy.optimize import minimize
import sys
import pickle

# These functions run the orbit fitting

# these four functions transform a coordinate c at date dates[ii]


class JointFit:
    def __init__(self, stamps, variances, dates, stamp_center_radec, stamp_center_pixel, psfs, wcs_list):

        self.earth_pos_list = []
        self.obs_pos_list = []
        self.dates = dates
        self.j = len(stamps)
        self.stamps = stamps
        self.variances = variances
        self.stamp_center_radec = stamp_center_radec
        self.stamp_center_pixel = stamp_center_pixel
        self.wcs_list = wcs_list
        self.psfs = psfs
        self.stamp_pos = np.array(stamp_center_pixel)
        self.weights = 1 / np.array(self.variances)

        for i in range(self.j):
            with solar_system_ephemeris.set(
                "de432s"
            ):  # https://docs.astropy.org/en/stable/coordinates/solarsystem.html
                earth_pos = get_body_barycentric("earth", Time(dates[i], format="mjd"))
                self.earth_pos_list.append(earth_pos)

                obs_pos = EarthLocation.of_site("ctio").get_gcrs(Time(dates[i], format="mjd"))
                obs_pos.representation_type = "cartesian"

                obs_pos = SkyCoord(
                    earth_pos.x + obs_pos.x,
                    earth_pos.y + obs_pos.y,
                    earth_pos.z + obs_pos.z,
                    representation_type="cartesian",
                )
                self.obs_pos_list.append(obs_pos)

    def geo_to_bary_fast(self, c, i):
        c.representation_type = "cartesian"
        earth_pos = self.earth_pos_list[i]
        c = SkyCoord(c.x + earth_pos.x, c.y + earth_pos.y, c.z + earth_pos.z, representation_type="cartesian")
        c.representation_type = "spherical"
        return c

    def bary_to_geo_fast(self, c, i):
        c.representation_type = "cartesian"
        earth_pos = self.earth_pos_list[i]
        c = SkyCoord(c.x - earth_pos.x, c.y - earth_pos.y, c.z - earth_pos.z, representation_type="cartesian")
        c.representation_type = "spherical"
        return c

    def obs_to_bary_fast(self, c, i):
        c.representation_type = "cartesian"
        obs_pos = self.obs_pos_list[i]
        c = SkyCoord(c.x + obs_pos.x, c.y + obs_pos.y, c.z + obs_pos.z, representation_type="cartesian")
        c.representation_type = "spherical"
        return c

    def bary_to_obs_fast(self, c, i):
        c.representation_type = "cartesian"
        obs_pos = self.obs_pos_list[i]
        c = SkyCoord(c.x - obs_pos.x, c.y - obs_pos.y, c.z - obs_pos.z, representation_type="cartesian")
        c.representation_type = "spherical"
        return c

    # output is the barycentric coordinate that, when observed from Earth on dates[i]
    # gives geocentric coordinate c while having barycentric distance dist
    def geo_to_bary_wdist(self, c, i, dist):
        c.representation_type = "cartesian"
        earth_pos = self.earth_pos_list[i]
        r2_earth = earth_pos.x * earth_pos.x + earth_pos.y * earth_pos.y + earth_pos.z * earth_pos.z

        dot = earth_pos.x * c.x + earth_pos.y * c.y + earth_pos.z * c.z
        bary_dist = dist * u.au
        r = -dot + np.sqrt(bary_dist * bary_dist - r2_earth + dot * dot)

        bary_c = SkyCoord(
            earth_pos.x + r * c.x,
            earth_pos.y + r * c.y,
            earth_pos.z + r * c.z,
            representation_type="cartesian",
        )
        bary_c.representation_type = "spherical"
        return bary_c

    def obs_to_bary_wdist(self, c, i, dist):
        c.representation_type = "cartesian"
        obs_pos = self.obs_pos_list[i]
        r2_earth = obs_pos.x * obs_pos.x + obs_pos.y * obs_pos.y + obs_pos.z * obs_pos.z

        dot = obs_pos.x * c.x + obs_pos.y * c.y + obs_pos.z * c.z
        bary_dist = dist * u.au
        r = -dot + np.sqrt(bary_dist * bary_dist - r2_earth + dot * dot)

        bary_c = SkyCoord(
            obs_pos.x + r * c.x, obs_pos.y + r * c.y, obs_pos.z + r * c.z, representation_type="cartesian"
        )
        bary_c.representation_type = "spherical"
        return bary_c

    def model_traj_from_geo(self, x1, y1, x2, y2, bary_dist):
        c1 = self.wcs_list[0].pixel_to_world(x1, y1)
        c1b = self.geo_to_bary_wdist(c1, 0, bary_dist)
        c2 = self.wcs_list[-1].pixel_to_world(x2, y2)
        c2b = self.geo_to_bary_wdist(c2, -1, bary_dist)

        m_ra = (c2b.ra - c1b.ra) / (self.dates[-1] - self.dates[0])
        m_dec = (c2b.dec - c1b.dec) / (self.dates[-1] - self.dates[0])

        pix = np.zeros((self.j, 2))
        for i in range(self.j):
            c = SkyCoord(
                c1b.ra + m_ra * (self.dates[i] - self.dates[0]),
                c1b.dec + m_dec * (self.dates[i] - self.dates[0]),
                distance=bary_dist * u.au,
            )
            c = self.bary_to_geo_fast(c, i)
            w = self.wcs_list[i]

            pix[i] = w.world_to_pixel(c)
        return pix

    def model_traj_from_obs(self, x1, y1, x2, y2, bary_dist):
        c1 = self.wcs_list[0].pixel_to_world(x1, y1)
        c1b = self.obs_to_bary_wdist(c1, 0, bary_dist)
        c2 = self.wcs_list[-1].pixel_to_world(x2, y2)
        c2b = self.obs_to_bary_wdist(c2, -1, bary_dist)

        m_ra = (c2b.ra - c1b.ra) / (self.dates[-1] - self.dates[0])
        m_dec = (c2b.dec - c1b.dec) / (self.dates[-1] - self.dates[0])

        pix = np.zeros((self.j, 2))
        for i in range(self.j):
            c = SkyCoord(
                c1b.ra + m_ra * (self.dates[i] - self.dates[0]),
                c1b.dec + m_dec * (self.dates[i] - self.dates[0]),
                distance=bary_dist * u.au,
            )
            c = self.bary_to_obs_fast(c, i)
            w = self.wcs_list[i]

            pix[i] = w.world_to_pixel(c)
        return pix

    def model_images(self, traj):
        mdl = np.zeros_like(self.psfs)
        for i in range(self.j):
            mdl[i] = shift(
                self.psfs[i], [traj[i, 1] - self.stamp_pos[i, 1], traj[i, 0] - self.stamp_pos[i, 0]]
            )
        return mdl

    #    def model_images_streaked(self,traj):
    #        """
    #        This only works for linear trajectories
    #        """
    #        mdl = np.zeros_like(self.psfs)
    #        mean_time = np.mean(self.dates)
    #        vx = (traj[-1,0]-traj[0,0])/(self.dates[-1]-self.dates[0])
    #        vy = (traj[-1,1]-traj[0,1])/(self.dates[-1]-self.dates[0])
    #        streaked_psf = np.zeros_like(self.psfs[0])
    #        for i in range(self.j):
    #            current_time_offset = mean_time-self.dates[i]
    #            streaked_psf += shift(self.psfs[i],(current_time_offset*vy,current_time_offset*vx))
    #        streaked_psf/=self.j
    #
    #        for i in range(self.j):
    #            mdl[i] = shift(self.psfs[i], [traj[i,1]-self.stamp_pos[i,1], traj[i,0]-self.stamp_pos[i,0]])-streaked_psf
    #        return mdl

    def model_images_streaked(self, traj):
        """
        This will work for any trajectory
        """
        mdl = np.zeros_like(self.psfs)
        num_img = self.j
        for i in range(num_img):
            streaked_psf = np.zeros_like(self.psfs[0])
            for ii in range(num_img):
                streaked_psf += scipy.ndimage.shift(
                    self.psfs[ii], (traj[ii, 1] - traj[i, 1], traj[ii, 0] - traj[i, 0])
                )
            streaked_psf /= num_img

            mdl[i] = scipy.ndimage.shift(
                self.psfs[i] - streaked_psf,
                [traj[i, 1] - self.stamp_pos[i, 1], traj[i, 0] - self.stamp_pos[i, 0]],
            )
        return mdl

    def model_images_streaked_2(self, traj):
        """
        This models streaked i mages without nested for loops
        TODO: Confirm that this reproduces model_images_streaked()
        """
        x = traj[:, 0]
        y = traj[:, 1]
        minx = int(np.floor(np.min(np.concatenate([x, self.stamp_pos[:, 0]]))))
        maxx = int(np.ceil(np.max(np.concatenate([x, self.stamp_pos[:, 0]]))))
        miny = int(np.floor(np.min(np.concatenate([y, self.stamp_pos[:, 1]]))))
        maxy = int(np.ceil(np.max(np.concatenate([y, self.stamp_pos[:, 1]]))))
        nimg = self.j
        stampsize = np.shape(self.psfs[0])[0]

        streakedPSF = np.zeros(shape=(maxy - miny + stampsize, maxx - minx + stampsize))
        streakstamps = np.zeros_like(self.psfs)

        # place PSFs on subimage that contains trajectory
        for i in range(nimg):
            rely = int(np.floor(y[i] - miny))
            relx = int(np.floor(x[i] - minx))
            streakedPSF[rely : rely + stampsize, relx : relx + stampsize] += (
                shift(self.psfs[i], (y[i] % 1, x[i] % 1)) / nimg
            )
        # extract streak portions centred on trajectory points
        for i in range(nimg):
            rely = int(np.floor(self.stamp_pos[i, 1] - miny))
            relx = int(np.floor(self.stamp_pos[i, 0] - minx))
            streakstamps[i] = streakedPSF[rely : rely + stampsize, relx : relx + stampsize]

        mdl = np.zeros_like(self.psfs)
        for i in range(nimg):
            mdl[i] = shift(
                self.psfs[i], [traj[i, 1] - self.stamp_pos[i, 1], traj[i, 0] - self.stamp_pos[i, 0]]
            )

        return mdl - streakstamps

    def bestfluxes(self, traj):
        freg = 25000

        mdl = self.model_images(traj)
        a = np.sum(mdl * mdl * self.weights, axis=(1, 2)) + freg**-2
        c = np.sum(mdl * self.stamps * self.weights, axis=(1, 2))
        return c / a

    def kbmodFluxes(self, traj):

        mdl = self.model_images(traj)
        a = np.sum(mdl * mdl * self.weights, axis=(1, 2))
        c = np.sum(mdl * self.stamps * self.weights, axis=(1, 2))
        return c / a

    def kbmodFluxes_streaked(self, traj):

        mdl = self.model_images_streaked(traj)
        a = np.sum(mdl * mdl * self.weights, axis=(1, 2))
        c = np.sum(mdl * self.stamps * self.weights, axis=(1, 2))
        return c / a

    def kbmodLH(self, traj):

        mdl = self.model_images(traj)
        a = np.sum(mdl * mdl * self.weights, axis=(1, 2))
        c = np.sum(mdl * self.stamps * self.weights, axis=(1, 2))
        return c / np.sqrt(a)

    def kbmodSumLH(self, traj):

        mdl = self.model_images(traj)
        a = np.sum(mdl * mdl * self.weights)
        c = np.sum(mdl * self.stamps * self.weights)
        return c / np.sqrt(a)

    # obtain flux and error from maximum likelihood fit for f
    def kbmodSumFluxes_streakedML(self, traj):

        mdl = self.model_images_streaked(traj)
        a = np.sum(mdl * mdl * self.weights)
        c = np.sum(mdl * self.stamps * self.weights)
        # flux, std dev
        return c / a, np.sqrt(1 / a)

    def kbmodPhiPsi_streaked(self, traj):
        mdl = self.model_images_streaked(traj)
        a = np.sum(mdl * mdl * self.weights, axis=(1, 2))
        c = np.sum(mdl * self.stamps * self.weights, axis=(1, 2))
        return c, a

    def kbmodPhiPsi(self, traj):
        mdl = self.model_images(traj)
        a = np.sum(mdl * mdl * self.weights, axis=(1, 2))
        c = np.sum(mdl * self.stamps * self.weights, axis=(1, 2))
        return c, a

    def uncertainties(self, traj):
        mdl = self.model_images(traj)
        a = np.sum(mdl * mdl * self.weights, axis=(1, 2))
        return 1 / np.sqrt(a)

    def negloglike_from_geo(self, x):
        x1, y1, x2, y2, bary_dist = x

        traj = self.model_traj_from_geo(x1, y1, x2, y2, bary_dist)
        mdl = self.model_images(traj)
        f = self.bestfluxes(traj)
        f[f < 0] = 0

        bestmdl = f[:, None, None] * mdl
        logL = 0.5 * np.sum(self.weights * (bestmdl - self.stamps) ** 2)
        return logL

    def negloglike_from_geo_fixdist(self, x, bary_dist):
        x = np.append(x, bary_dist)
        return self.negloglike_from_geo_2(x)

    def negloglike_from_obs(self, x):
        x1, y1, x2, y2, bary_dist = x

        traj = self.model_traj_from_obs(x1, y1, x2, y2, bary_dist)
        mdl = self.model_images(traj)
        f = self.bestfluxes(traj)
        f[f < 0] = 0

        bestmdl = f[:, None, None] * mdl
        logL = 0.5 * np.sum(self.weights * (bestmdl - self.stamps) ** 2)
        return logL

    def array_deltaLH_from_obs(self, x):
        x1, y1, x2, y2, bary_dist = x

        traj = self.model_traj_from_obs(x1, y1, x2, y2, bary_dist)
        mdl = self.model_images(traj)
        f = self.bestfluxes(traj)
        f[f < 0] = 0

        bestmdl = f[:, None, None] * mdl
        logL = 0.5 * np.sum(self.weights * (bestmdl - self.stamps) ** 2, axis=(1, 2))
        null_H = 0.5 * np.sum(self.weights * np.array(self.stamps) ** 2, axis=(1, 2))
        return null_H - logL

    def negloglike_from_obs_fixdist(self, x, bary_dist):
        x = np.append(x, bary_dist)
        return self.negloglike_from_obs(x)

    def model_traj_topo_pv(self, ra_m, dec_m, v_ra, v_dec):
        """
        Fit straight line in topocentric coordinates - appropriate for short arcs
        Position and velocity at mean time
        """

        mean_date = np.mean(self.dates)

        pix = np.zeros((self.j, 2))
        for i in range(self.j):
            c = SkyCoord(
                ra_m + v_ra * (self.dates[i] - mean_date),
                dec_m + v_dec * (self.dates[i] - mean_date),
                unit="deg",
            )
            w = self.wcs_list[i]

            pix[i] = w.world_to_pixel(c)
        return pix

    def model_traj_topo_pp(self, ra_a, dec_a, ra_b, dec_b):
        """
        Fit straight line in topocentric coordinates - appropriate for short arcs
        Positions at mean time +/- std dev
        """
        mean_date = np.mean(self.dates)
        ra_m = (ra_a + ra_b) / 2
        dec_m = (dec_a + dec_b) / 2

        std_date = np.std(self.dates)
        v_ra = (ra_b - ra_a) / (2 * std_date)
        v_dec = (dec_b - dec_a) / (2 * std_date)

        pix = np.zeros((self.j, 2))
        for i in range(self.j):
            c = SkyCoord(
                ra_m + v_ra * (self.dates[i] - mean_date),
                dec_m + v_dec * (self.dates[i] - mean_date),
                unit="deg",
            )
            w = self.wcs_list[i]

            pix[i] = w.world_to_pixel(c)
        return pix

    def model_traj_topo_start_end(self, xi, yi, xf, yf):
        """
        Fit straightl ine in topocentric coordinates - appropriate for short arcs
        Positions at starting and ending time
        """
        v_x = (xf - xi) / (self.dates[-1] - self.dates[0])
        v_y = (yf - yi) / (self.dates[-1] - self.dates[0])
        pix = np.zeros((self.j, 2))

        for i in range(self.j):
            elapsed_time = self.dates[i] - self.dates[0]
            pix[i] = [xi + v_x * elapsed_time, yi + v_y * elapsed_time]
        return pix

    def negloglike_topo_pv(self, x):
        ra_m, dec_m, v_ra, v_dec = x

        traj = self.model_traj_topo_pv(ra_m, dec_m, v_ra, v_dec)
        mdl = self.model_images(traj)
        f = self.bestfluxes(traj)
        f[f < 0] = 0

        bestmdl = f[:, None, None] * mdl
        logL = 0.5 * np.sum(self.weights * (bestmdl - self.stamps) ** 2)
        return logL

    def negloglike_topo_pp(self, x):
        ra_a, dec_a, ra_b, dec_b = x

        traj = self.model_traj_topo_pp(ra_a, dec_a, ra_b, dec_b)
        mdl = self.model_images(traj)
        f = self.bestfluxes(traj)
        f[f < 0] = 0

        bestmdl = f[:, None, None] * mdl
        logL = 0.5 * np.sum(self.weights * (bestmdl - self.stamps) ** 2)
        return logL

    def negloglike_topo_start_end(self, x):
        xi, yi, xf, yf = x

        traj = self.model_traj_topo_start_end(xi, yi, xf, yf)
        mdl = self.model_images(traj)
        f = self.bestfluxes(traj)
        f[f < 0] = 0

        bestmdl = f[:, None, None] * mdl
        logL = 0.5 * np.sum(self.weights * (bestmdl - self.stamps) ** 2)
        return logL

    def negloglike_topo_start_end_streaked(self, x):
        """
        This only works for linear trajectories
        This previously used model_images_streaked_2()
        """
        xi, yi, xf, yf = x

        traj = self.model_traj_topo_start_end(xi, yi, xf, yf)
        mdl = self.model_images_streaked(traj)
        f = self.bestfluxes(traj)
        f[f < 0] = 0

        bestmdl = f[:, None, None] * mdl
        logL = 0.5 * np.sum(self.weights * (bestmdl - self.stamps) ** 2)
        return logL

    def compare_traj(self, traj):
        plt.figure(figsize=(16, 16))
        n = np.ceil(np.sqrt(self.j))
        for i in range(self.j):
            plt.subplot(n, n, i + 1)
            plt.imshow(self.stamps[i])
            plt.scatter(
                [self.window + traj[i, 0] - self.stamp_pos[i, 0]],
                [self.window + traj[i, 1] - self.stamp_pos[i, 1]],
                marker="x",
                c="r",
                s=200,
            )
        plt.show()


def get_mpc_times(filename):

    """
    Read in a file with observations in MPC format and return the coordinates.

    Inputs
    ------
    filename: str
        The name of the file with the MPC-formatted observations.

    Returns
    -------
    c: astropy SkyCoord object
        A SkyCoord object with the ra, dec of the observations.
    """
    iso_times = []
    time_frac = []
    ra = []
    dec = []

    with open(filename, "r") as f:
        for line in f:
            year = str(line[15:19])
            month = str(line[20:22])
            day = str(line[23:31])
            iso_times.append(str("%s-%s-%s" % (year, month, day)))
            time_frac.append(str(line[25:31]))
            ra.append(str(line[32:44]))
            dec.append(str(line[44:56]))

    return iso_times


def load_pg_names(file_path):
    """
    Load the pointing group numbers, ccd numbers,
    and indexes based on the file names of the
    detected objects
    """

    found_object_filenames = os.listdir(file_path)
    pg_names = []
    ccd_nums = []
    indexes = []
    for file in found_object_filenames:
        if file[-4:] == ".png":
            split_name = file.split("_")
            pg_names.append("_".join([split_name[0], split_name[1]]))
            ccd_nums.append(split_name[2])
            indexes.append(int(split_name[3]))
    return (pg_names, ccd_nums, indexes)


def load_pg_names_from_df(file_path, good_list_path, pg_name=None, suffix="FAKE_DEEP_hyak"):
    """
    Load the pointing group numbers, ccd numbers, and indexes from the
    dataframe generated for the vetting of the images with the fakes injected.

    Inputs
    ------
    file_path: str
        The path to the dataframe containing all the metadata for the results
    good_list_path: str
        The path to the text file containing the obj_ids for the candidates
        labelled as "good"
    pg_name: str
        Optional parameter to slice the dataframe based on a single pointing
        group
    suffix: str
        Slice the dataframe based on the kbmod suffix
    """
    metadata_df = pd.read_csv(file_path)
    with open(good_list_path, "r") as f:
        lines = f.readlines()
    found_obj_id = [int(line[:6]) for line in lines]
    good_df = metadata_df.iloc[np.intersect1d(metadata_df["obj_id"], found_obj_id, return_indices=True)[1]]

    if pg_name is not None:
        good_df = good_df[good_df["pg_name"] == pg_name]

    good_df = good_df[good_df["suffix"] == suffix]

    pg_names = good_df["pg_name"].values
    ccd_names = good_df["ccd_name"].values
    indexes = good_df["index"].values
    return (pg_names, ccd_names, indexes)


def load_pg_names_from_df_w_suffix(file_path, good_list_path=None, pg_name=None):
    """
    Load the pointing group numbers, ccd numbers, and indexes from the
    dataframe generated for the vetting of the images with the fakes injected.

    Inputs
    ------
    file_path: str
        The path to the dataframe containing all the metadata for the results
    good_list_path: str
        The path to the text file containing the obj_ids for the candidates
        labelled as "good"
    pg_name: str
        Optional parameter to slice the dataframe based on a single pointing
        group
    """
    metadata_df = pd.read_csv(file_path)
    if good_list_path is not None:
        with open(good_list_path, "r") as f:
            lines = f.readlines()
        found_obj_id = [int(line[:6]) for line in lines]
        good_df = metadata_df.iloc[
            np.intersect1d(metadata_df["obj_id"], found_obj_id, return_indices=True)[1]
        ]
    else:
        good_df = metadata_df

    if pg_name is not None:
        good_df = good_df[good_df["pg_name"] == pg_name]

    pg_names = good_df["pg_name"].values
    ccd_names = good_df["ccd_name"].values
    indexes = good_df["index"].values
    suffixes = good_df["suffix"].values
    return (pg_names, ccd_names, indexes, suffixes)


def shift_images(traj, stamps, stamp_pos):
    shifted_stamps = []
    for i in range(len(stamps)):
        shifted_stamps.append(
            shift(stamps[i], [-traj[i, 1] + stamp_pos[i, 1], -traj[i, 0] + stamp_pos[i, 0]])
        )
    return shifted_stamps
