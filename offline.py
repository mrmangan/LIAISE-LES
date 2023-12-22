# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 16:44:33 2022

Create the land surface input file for the LIAISE regional MicroHH case.

The size of the land surface map is ~32 m resloution, so keep the resloution some coordinate of that. 
import operator as op

@author: manga005
Last updated: 11 April 2023
"""


#import sources
import numpy as np
# import netCDF4 as nc4

#import custom scripts
import microhh_tools as mht
import haversine as hs
import xarray as xr
import microhh_ls2d_tools as mlt
import socket 
from datetime import datetime
import pandas as pd
from scipy.interpolate import griddata
from bresenham import bresenham
import operator as op

import sys
sys.path.append('/gpfs/home3/mrmangan/models/LS2D/')
import ls2d

float_type = np.float32


# path = 'C:\\Users\\manga005\\OneDrive - Wageningen University & Research\\Research\\LES\\Offline'
# os.chdir(path)

# Experiment name
exp_name = 'offline'


# Read MicroHH namelist & grid properties:
nl = mlt.read_namelist(exp_name + '.ini.base')

# Constants
Rd = 287.04
Rv = 461.5
ep = Rd/Rv

"""
SET PATH SETTINGS
"""
#
# Switch between different systems:
#
env_snellius = {
        'system': 'snellius',
        'ls2d_path': '/home/mrmangan/models/LS2D',
        'ls2d_cases_path': '/home/mrmangan/models/LS2D_cases',
        'era5_path': '/home/mrmangan/data/ERA5/',
        'work_path': '/scratch-shared/mrmangan/liaise',
        'microhh_path': '/home/mrmangan/models/develop_old/microhh',
        'land_use_source': 'corine',
        'land_use_tif': '/home/bstratum/data/Corine/u2018_clc2018_v2020_20u1_raster100m/DATA/U2018_CLC2018_V2020_20u1.tif'}
env_eddy = {
        'system': 'eddy',
        'ls2d_path': '/home/maryrose/models/LS2D',
        'era5_path': '/home/scratch1/maryrose/data/',
        'microhh_path': '/home/maryrose/models/microhh',
        'land_use_source': 'corine',
        'land_use_tif': '/home/scratch1/bart/Corine/u2018_clc2018_v2020_20u1_raster100m/DATA/U2018_CLC2018_V2020_20u1.tif',
        'work_path': 'test_case'}

env_arch = {
        'system': 'arch',
        'ls2d_path': '/home/bart/meteo/models/LS2D',
        'ls2d_cases_path': '/home/bart/meteo/models/LS2D_cases',
        'era5_path': '/home/scratch1/meteo_data/LS2D/',
        'work_path': '/home/bart/meteo/models/LS2D_cases/microhh/cabauw_realistic_sfc/test_case',
        'microhh_path': '/home/bart/meteo/models/microhh',
        'land_use_source': 'corine',
        'land_use_tif': '/home/scratch1/meteo_data/Corine/u2018_clc2018_v2020_20u1_raster100m/DATA/U2018_CLC2018_V2020_20u1.tif'}

env_ecmwf = {
        'system': 'ecmwf',
        'ls2d_path': '/home/ms/nl/nkbs/models/LS2D',
        'era5_path': '/scratch/ms/nl/nkbs/LS2D'}

host = socket.gethostname()
print('Host = {}'.format(host))

if host == 'manjaro':
    env = env_arch
elif 'int' in host or 'tcn' in host or 'gcn' in host:
    env = env_snellius
elif 'ecgb' in host:
    env = env_ecmwf
elif 'eddy' in host:
    env = env_eddy
elif 'MBP' in host:
    env = env_mbp
else:
    raise Exception('Unknown host in settings.py')


"""
BOTTOM BOUNDARY CONDITIONS 
"""

#read in heat flux files
df = xr.open_dataset('LIAISE_GoldenDay_Sfc_30m_ext_8Dec.nc', decode_times = False)  #want to regrid this to match with the data!
df = df.reindex(lat=df.lat[::-1])  #reverse the lat index so that we can transpose the data

#limit data frame to have correct # of grid cells
# df = df.isel(lon = slice(0, 416), lat = slice(0, 416))

# pick itot and jtot from the land surface 
# make sure this matches ini files
factor = 1  #this the factor increase in points from the default map (e.g. factor = 2, res is 1/2 of map)

#set names for grid characteristics: 
itot = nl['grid']['itot']
jtot = nl['grid']['jtot']
ktot = nl['grid']['ktot']
xsize = nl['grid']['xsize']
ysize = nl['grid']['ysize']
zsize = nl['grid']['zsize']

#time variables
endtime = nl['time']['endtime']
sampletime = nl['stats']['sampletime']
dt = endtime/sampletime		#time step (s)

dz = zsize / ktot

#make the grid with diferent fluxes
#fill with zeros
thlfluxbot = np.zeros((jtot, itot), dtype=float_type)
qtfluxbot  = np.zeros((jtot, itot), dtype=float_type)


# make ini files for the time varying surface variables
outputtimes = np.arange(start  = 0, stop = 45000, step = 1800) 
outputtimes = list(map(str,outputtimes))
for j in range(0, len(outputtimes)):
    outputtimes[j] = str(outputtimes[j]).zfill(7)
    
# fix the size of the arrays to match itot by jtot
buff_y = len(df.lat) - jtot
buff_x = len(df.lon) - itot

#select the ind - because there is an odd number as a buffer, remove one extra col from the eastern side. Change if input changes!!
df  = df.isel(lat = slice(int(buff_y/2), len(df.lat) - int(buff_y/2) - 1),
              lon = slice(int(buff_x/2), len(df.lon) - int(buff_x/2) - 1))

j = 0
for i in range(12, 37):
    thlfluxbot = np.array(df.h[i, :, :].T)
    qtfluxbot = np.array(df.le[i, :, :].T)
    
    #change unit into kinematic units instead of flux units
    thlfluxbot = thlfluxbot / 1.2/1004.
    qtfluxbot = qtfluxbot / 1.2/2.5e6

    #increase resloution (if necessary)
    # qfluxbot = np.repeat(qfluxbot, factor, axis = 0)
    # qfluxbot = np.repeat(qfluxbot, factor, axis = 1)
     
    # thlfluxbot = np.repeat(thlfluxbot, factor, axis = 0)
    # thlfluxbot = np.repeat(thlfluxbot, factor, axis = 1)
    
    thlfluxbot.astype(float_type).tofile('thl_bot_in.'+ outputtimes[j])
    qtfluxbot.astype(float_type).tofile('qt_bot_in.'+ outputtimes[j])
    
    j = j + 1  #update j counter
    
    
# Make static time files for the scalar fluxes
# in the netcdf file, somehow the indicies are flipped for some are flipped. Changed orietation manuanally.
cbot = np.array(df.lanscape.T)
dbot = np.array(df.local.T)
epbot = np.array(df.ep.T)
reg = np.array(df.regional.T)
dry = np.array(df.dry.T)

# make output files for the domains
cbot.astype(float_type).tofile('wet.0000000')
dbot.astype(float_type).tofile('alfalfa.0000000')
reg.astype(float_type).tofile('regional.0000000')
dry.astype(float_type).tofile('dry.0000000')
epbot.astype(float_type).tofile('fallow.0000000') 

# print scalar files
cbot.astype(float_type).tofile('c_bot_in.0000000')
dbot.astype(float_type).tofile('d_bot_in.0000000')


# Make the roughness files
z0m = np.array(df.z0m.T)
z0h = np.array(df.z0h.T)

#print to the output file 
z0m.astype(float_type).tofile('z0m.0000000')
z0h.astype(float_type).tofile('z0h.0000000')


j = 0
for i in range(12, 37):
#    z0m.tofile('z0m_bot_in.'+ outputtimes[j])
#    z0h.tofile('z0h_bot_in.'+ outputtimes[j])
    cbot.astype(float_type).tofile('c_bot_in.' + outputtimes[j])
    dbot.astype(float_type).tofile('d_bot_in.' + outputtimes[j])

    j = j + 1  #update j counter


# Check the input fluxes!

j = 0
df_out = pd.DataFrame(columns=['i', 'thlflux_dom', 'thlflux_reg', 'thlflux_wet', 'thlflux_dry', 'thlflux_alf', 'thlflux_fal', 'qflux_dom', 'qflux_reg', 'qflux_wet', 'qflux_dry', 'qflux_alf', 'qflux_fal'])
for i in range(12, 36):
    thlfluxbot = np.array(df.h[i, :, :]).T
    qfluxbot = np.array(df.le[i, :, :]).T
    
    #change unit into kinematic units instead of flux units
#    thlfluxbot = thlfluxbot / 1.2/1004.
#    qfluxbot = qfluxbot / 1.2/2.5e6
    
    df_out.loc[j] = [i] + [np.mean(thlfluxbot)] + [np.mean(thlfluxbot[np.where(reg == 1.0)])] + [np.mean(thlfluxbot[np.where(cbot == 1.0)])] + [np.mean(thlfluxbot[np.where(dry == 1.0)])] + [np.mean(thlfluxbot[np.where(dbot == 1.0)])] + [np.mean(thlfluxbot[np.where(epbot == 1.0)])] + [np.mean(qfluxbot)] + [np.mean(qfluxbot[np.where(reg == 1.0)])]+ [np.mean(qfluxbot[np.where(cbot == 1.0)])] + [np.mean(qfluxbot[np.where(dry == 1.0)])] + [np.mean(qfluxbot[np.where(dbot == 1.0)])] + [np.mean(qfluxbot[np.where(epbot == 1.0)])]
    
    j = j + 1
    
df_out.to_csv('ini_fluxes.csv')

"""
LATERAL BOUNDARY CONDITIONS 
"""
# Make the profiles using the large-scale forcings from ERA5

settings = {
    'start_date'  : datetime(year=2021, month=7, day=21, hour=6),
    'end_date'    : datetime(year=2021, month=7, day=21, hour=18),
    'central_lat' : 41.7,
    'central_lon' : 1.0,
    'area_size'   : 1,
    'case_name'   : 'liaise',
    'era5_path'   : env['era5_path'],
    'era5_expver' : 1,
    'write_log'   : False,
    'data_source' : 'CDS'
    }
# Download ERA5.
# ls2d.download_era5(settings)

# Read ERA5 data, and calculate derived properties (thl, etc.):
era = ls2d.Read_era5(settings)

# Calculate initial conditions and large-scale forcings for LES:
era.calculate_forcings(n_av=0, method='4th')

# Define vertical grid LES:
# grid = ls2d.grid.Grid_equidist(kmax=196, dz0 = 25.)

#manually streched grid
heights = [0, 500, 1000, 3000, 10000 ]
alpha = [1.0, 1.01, 1.015, 1.03] 
grid = ls2d.grid.Grid_stretched_manual(kmax=196, dz0=10, heights=heights, factors=alpha)

#grid.plot()


# Interpolate ERA5 variables and forcings onto LES grid.
# In addition, `get_les_input` returns additional variables needed to init LES.
les_input = era.get_les_input(grid.z)

# Remove top ERA5 level, to ensure that pressure stays
# above the minimum reference pressure in RRTMGP
les_input = les_input.sel(lay=slice(0,135), lev=slice(0,136))


# Import sonding for ini
df = pd.read_csv('LIAISE_ELS-PLANS_RS_20210721_060007.csv')

df.theta = df.theta + 273.15
df['q'] = df.mixingRatio/1000.

df['u'] = -1 * df.windSpeed * np.sin(df.windDirection * np.pi/180)
df['v'] = -1 * df.windSpeed * np.cos(df.windDirection * np.pi/180)

# interpolate onto LES grid
q_les = griddata(df.altitude, df.q, grid.z)
thl_les = griddata(df.altitude, df.theta, grid.z)
u_les = griddata(df.altitude, df.u, grid.z)
v_les = griddata(df.altitude, df.v, grid.z)

h2o_les = q_les / (ep - ep*q_les)


#
# MicroHH specific initialisation
#
# Settings:
# Nudging time scale atmosphere
tau_nudge = 10800
# ------------------


# Nudge factor
nudge_fac = np.ones(grid.kmax) / tau_nudge

#
# Write NetCDF input file for MicroHH
#
init_profiles = {
        'z': grid.z,
        'thl': thl_les,
        'qt': q_les,
        'u': u_les,
        'v': v_les,
        'h2o': h2o_les}

timedep_ls = {
        'time_ls': les_input['time_sec'],
        'u_geo': les_input['ug'],
        'v_geo': les_input['vg'],
        'w_ls': les_input['wls'],
        'thl_ls': les_input['dtthl_advec'],
        'qt_ls': les_input['dtqt_advec'],
        'u_ls': les_input['dtu_advec'],
        'v_ls': les_input['dtv_advec']}
        
timedep_surface = {
        'time_surface': les_input['time_sec'],
        'p_sbot': les_input['ps'] }

mlt.write_netcdf_input(
        exp_name, 'f8', init_profiles,
        timedep_surface, timedep_ls)        


# Update namelist
nl['grid']['ktot'] = grid.kmax
nl['grid']['zsize'] = grid.zsize
nl['time']['endtime'] = float(les_input['time_sec'][-1])
# nl['time']['endtime'] = float(7200)
nl['force']['fc'] = les_input.attrs['fc']
nl['radiation']['lon'] = settings['central_lon']
nl['radiation']['lat'] = settings['central_lat']

start = settings['start_date']
datetime_utc = '{0:04d}-{1:02d}-{2:02d} {3:02d}:{4:02d}:{5:02d}'.format(
        start.year, start.month, start.day, start.hour, start.minute, start.second)
nl['time']['datetime_utc'] = datetime_utc

# Add column locations - for locations: LC, EP, other locations? 
locs = pd.read_csv('tower_450m.csv')
locs['y'] = locs['y'].astype(int)

#convert from i,j to m
locs['y'] = locs['y']*30.
locs['x'] = locs['x']*30.


nl['column']['coordinates[x]'] = list(locs.x)
nl['column']['coordinates[y]'] = list(locs.y)

mlt.write_namelist(exp_name + '.ini', nl)

