import numpy as np
from lsst.sims.utils import CoordinateTransformations as ct
import matplotlib.pyplot as plt

def inclined_vec(a,i,t,theta_0=0.,omega0=2*np.pi):
    omega = a**(-3/2.)*omega0
    theta_0_rad = np.deg2rad(theta_0)
    x = a*np.cos(omega*t + theta_0_rad)
    y = a*np.cos(i)*np.sin(omega*t + theta_0_rad)
    z = a*np.sin(i)*np.sin(omega*t + theta_0_rad)
    return x,y,z

def earth_vec(t,omega0=2*np.pi):
    x = np.cos(omega0*t)
    y = np.sin(omega0*t)
    z = 0
    return x,y,z

def diff_vec(a,i,t,theta_0=0.,omega0=2*np.pi):
    x_o, y_o, z_o = inclined_vec(a,i,t,theta_0=theta_0,omega0=omega0)
    x_e, y_e, z_e = earth_vec(t,omega0=omega0)
    x_new = x_o-x_e
    y_new = y_o-y_e
    z_new = z_o-z_e
    return np.array((x_new, y_new, z_new))

def plot_trajectory(radius, incl, maxTime, dt, theta_0):
    """radius, in AU
       incl, inclination in degrees
       maxTime, max time of plot in years
       dt, time step in years
       theta_0, time 0 angle along orbit in relation to sun-Earth line.
                Note: probably would be more useful to have angle from
                opposition as viewed from Earth. Will make this change
                in future update
    """
    time_step = np.arange(0,maxTime,dt)
    lon = []
    lat = []
    for time in time_step:
        lon_now,lat_now = ct.sphericalFromCartesian(diff_vec(radius,np.deg2rad(incl),time,theta_0))
        lon.append(ct.arcsecFromRadians(lon_now))
        lat.append(ct.arcsecFromRadians(lat_now))
    lon = np.array(lon)
    lat = np.array(lat)
    #fig = plt.figure(figsize=(12,12))
    plt.scatter(lon, lat, c=time_step, lw=0)
    plt.xlabel('Long (arcsec)')
    plt.ylabel('Lat (arcsec)')
    plt.title('Trajectory of object over %.2f years' % maxTime)
    plt.colorbar()

def plot_ang_vel(radius, incl, maxTime, dt, theta_0):
    """radius, in AU
       incl, inclination in degrees
       maxTime, max time of plot in years
       dt, time step in years
       theta_0, time 0 angle along orbit in relation to sun-Earth line.
                Note: probably would be more useful to have angle from
                opposition as viewed from Earth. Will make this change
                in future update
    """
    time_step = np.arange(0,maxTime,dt)
    lon = []
    lat = []
    for time in time_step:
        lon_now,lat_now = ct.sphericalFromCartesian(diff_vec(radius,np.deg2rad(incl),time,theta_0))
        lon.append(lon_now)
        lat.append(lat_now)
    lon = np.array(lon)
    lat = np.array(lat)
    ang_vel = []
    for array_val in xrange(0, len(lon)-1):
        ang_vel.append(ct.arcsecFromRadians(ct.haversine(lon[array_val], lat[array_val],
                                                         lon[array_val+1], lat[array_val+1]))/(dt*365*24))
    #fig = plt.figure(figsize=(12,12))
    plt.plot(time_step[:-1], ang_vel)
    plt.ylabel('Arcsec/hr')
    plt.xlabel('Time (yrs)')
    plt.title('Max Angular Velocity = %.4e arcsec/hr.' % np.max(ang_vel))
