# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 19:00:07 2023

Limit and re-print the extents of the 3D fields based on areas of interest

@author: manga005
"""

# import libaries
import pandas as pd
import numpy as np
import xarray as xr

def read_df(x):
    # time limits
    # times = x.time / (60 * 60) + 6.
    # time_lim = np.where((times >= 8) & (times <=17))
    # time_lim = np.where((times >= 7) & (times <=8))
    z = x.z
    z_lim = np.where(z <= 3500)

    # times = times[time_lim] * (60 * 60) + 6

    # Limits of the regional domain. In indexs!
    xmin = 309
    xmax = 1142
    ymin = 232
    ymax = 1064

    # space limits
    df = x.isel(z = slice(0, np.shape(z_lim)[1]), x = slice(xmin, xmax), y = slice(ymin, ymax))
    
    return(df)

def read_df_u(x):  
    # time limits
    # times = x.time / (60 * 60) + 6.
    # time_lim = np.where((times >= 8) & (times <=17))
    # time_lim = np.where((times >= 7) & (times <=8))
    z = x.z
    z_lim = np.where(z <= 3500) 
    
    # times = times[time_lim] * (60 * 60) + 6
    
    # Limits of the regional domain. In indexs!
    xmin = 309
    xmax = 1142
    ymin = 232
    ymax = 1064

    # space limits
    df = x.isel(z = slice(0, np.shape(z_lim)[1]), xh = slice(xmin, xmax), y = slice(ymin, ymax))

    return(df)

def read_df_v(x):
    # time limits
    # times = x.time / (60 * 60) + 6.
    # time_lim = np.where((times >= 8) & (times <=17))
    # time_lim = np.where((times >= 7) & (times <=8))
    z = x.z
    z_lim = np.where(z <= 3500)

    # times = times[time_lim] * (60 * 60) + 6

    # Limits of the regional domain. In indexs!
    xmin = 309
    xmax = 1142
    ymin = 232
    ymax = 1064

    # space limits
    df = x.isel(z = slice(0, np.shape(z_lim)[1]), x = slice(xmin, xmax), yh = slice(ymin, ymax))

    return(df)
def read_df_w(x):
    # time limits
    # times = x.time / (60 * 60) + 6.
    # time_lim = np.where((times >= 8) & (times <=17))
    # time_lim = np.where((times >= 7) & (times <=8))
    z = x.zh
    z_lim = np.where(z <= 3500)

    # times = times[time_lim] * (60 * 60) + 6

    # Limits of the regional domain. In indexs!
    xmin = 309
    xmax = 1142
    ymin = 232
    ymax = 1064

    # space limits
    df = x.isel(zh = slice(0, np.shape(z_lim)[1]), x = slice(xmin, xmax), y = slice(ymin, ymax))

    return(df)

# read the data
u = xr.open_dataset('u.nc')
u_out = read_df_u(u)
u_out.to_netcdf("u_out.nc", format = 'NETCDF4', engine= 'netcdf4')
u.close()
del(u_out)

v = xr.open_dataset('v.nc')
v_out = read_df_v(v)
v_out.to_netcdf("v_out.nc", format = 'NETCDF4', engine= 'netcdf4')
v.close()
del(v_out)

w = xr.open_dataset('w.nc')
w_out = read_df_w(w)
w_out.to_netcdf("w_out.nc", format = 'NETCDF4', engine= 'netcdf4')
w.close()
del(w_out)

thl = xr.open_dataset('thl.nc')
thl_out = read_df(thl)
thl_out.to_netcdf("thl_out.nc", format = 'NETCDF4', engine= 'netcdf4')
thl.close()
del(thl_out)

qt = xr.open_dataset('qt.nc')
qt_out = read_df(qt)
qt_out.to_netcdf("qt_out.nc", format = 'NETCDF4', engine= 'netcdf4')
qt.close()
del(qt_out)

c = xr.open_dataset('c.nc')
c_out = read_df(c)
c_out.to_netcdf("c_out.nc", format = 'NETCDF4', engine= 'netcdf4')
c.close()
del(c_out)

d = xr.open_dataset('d.nc')
d_out = read_df(d)
d_out.to_netcdf("d_out.nc", format = 'NETCDF4', engine= 'netcdf4')
d.close()
del(d_out)


