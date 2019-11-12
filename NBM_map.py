# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:57:27 2019

@author: thomas.turnage
"""
import re
import os
import sys
import numpy as np
import pandas as pd
import requests
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from gis_layers import make_shapes_mi
shape_mini = make_shapes_mi()


from reference_data import nbm_station_dict
station_master = nbm_station_dict()


download = False

try:
    os.listdir('/var/www')
    base_gis_dir = '/data/GIS'
except:
    base_gis_dir = 'C:/data/GIS'


try:
    os.listdir('/usr')
    windows = False
    base_dir = '/data'
    sys.path.append('/data/scripts/resources')
    image_dir = os.path.join('/var/www/html/radar','images')
    image_dir = os.path.join('/data','images')

    raw_nbm_file = os.path.join(base_dir,'nbm_raw.txt')
    trimmed_nbm_file =  os.path.join(base_dir,'nbm_trimmed.txt')

except:
    windows = True
    base_dir = 'C:/data'    
    image_dir = os.path.join(base_dir,'images','NBM')
    raw_nbm_file = os.path.join(base_dir,'nbm_raw.txt')
    trimmed_nbm_file =  os.path.join(base_dir,'nbm_trimmed.txt')
    sys.path.append('C:/data/scripts/resources')

from my_functions import dtList_nbm, categorize

download = False
bulletin_type = 'nbhtx'

def download_nbm_bulletin(url,fname,path_check):
    dst = os.path.join(base_dir,fname)
    if path_check != 'just_path':
        r = requests.get(url)
        print('downloading ... ' + str(url))
        open(dst, 'wb').write(r.content)
    return dst

now = datetime.utcnow()
now2 = now - timedelta(hours=3)
ymd = now2.strftime('%Y%m%d')
hour = now2.strftime('%H')
#url = 'https://para.nomads.ncep.noaa.gov/pub/data/nccf/com/blend/para/blend.20191107/15/text/blend_nbhtx.t15z'
url = 'https://para.nomads.ncep.noaa.gov/pub/data/nccf/com/blend/para/blend.' + ymd + '/' + hour + '/text/blend_' + bulletin_type + '.t' + hour + 'z'


map_plot_stations = {}

mi_stations = []
for key in station_master:
    if station_master[key]['state'] == 'MI':
        mi_stations.append(key)

fname = 'nbm_raw_hourly.txt'

if download:
    raw_file_path = download_nbm_bulletin(url,fname,'hi')
    download = False
else:
    raw_file_path = download_nbm_bulletin(url,fname,'just_path')

for key in mi_stations:
    #if s in ['KAZO','KGRR','KMKG','KMOP','KMKG','KBIV']:
    station_id = key
    station_description = station_master[key]['name']
    lat = station_master[key]['lat']
    lon = station_master[key]['lon']
    column_list = []
    station_found = False
    utc_shift = station_master[key]['time_shift']
    p = re.compile(key)
    s = re.compile('SOL')
    dt = re.compile('DT')
    ymdh = re.compile('[0-9]+/[0-9]+/[0-9]+\s+[0-9]+')
        
    dst = open(trimmed_nbm_file, 'w')
    with open(raw_file_path) as fp:  
        for line in fp:
            m = p.search(line)
            sol = s.search(line)
            dt_match = dt.search(line)
            if m is not None:
                station_found = True
                dt_line = line
                ymdh_match = ymdh.search(dt_line)
                run_dt = datetime.strptime(ymdh_match[0], '%m/%d/%Y  %H%M')
                idx,model_run_local = dtList_nbm(run_dt,bulletin_type,utc_shift) ######################################################3333333
                start_time = idx[1]
                end_time = idx[-1]
                data_list = idx[1:-1]

            elif station_found and sol is None:
                if dt_match is not None:
                    pass
                else:
                    start = str(line[1:4])
                    column_list.append(start)
                    dst.write(line)             
            elif sol is not None and station_found:
                dst.close()
                break 

    nbm_old = None
    nbm = None
    
    nbm_old = pd.read_fwf(trimmed_nbm_file, widths=(5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3))
    elements = column_list[1:]

    # flip table so times are rows to align with pandas
    nbm = nbm_old.transpose()

    # after the flip, column names are useless. Use the created column_list before the flip
    # to make a dictionary that replaces bad column names with the original, pre-flip column names
    old_column_names = nbm.columns.tolist()
    col_rename_dict = {i:j for i,j in zip(old_column_names,elements)}
    nbm.rename(columns=col_rename_dict, inplace=True)
    
    # Now that columns names have been defined there is an extraneous line to remove
    # -- With nbhtx (hourly guidance), remove the UTC line.
    # -- With nbstx (short term guidance), remove the FHR line.
    # Then set the index with the pandas time series
    try:
        nbm.drop(['UTC'], inplace=True)

    except:
        pass

    nbm.set_index(data_list, inplace=True)

    sn_plot = []
    ts_list = []
    sn_list = nbm.S01.tolist()
    sn_cat_list = categorize(sn_list,'sn')
    for t in (np.arange(0,12,2)):
        ts = nbm.index[t]
        t_str = ts.strftime('%d %h %Y %H')
        t_str = ts.strftime('%B %d, %Y - %I %p')
        sn_plot.append(sn_cat_list[t])
        ts_list.append(t_str)
    map_plot_stations[key] = {'snow':sn_plot, 'time_string':ts_list, 'lon':lon,'lat':lat}



cat_color_dict = {'0':(0.2,0.2,0.2),
                  '1':(0.3,0.3,0.4),
                  '2':(0.4,0.4,0.6),
                  '3':(0.5,0.5,0.8),
                  '4':(0.6,0.6,0.2),
                  '5':(0.6,0.0,0.0),
                  '6':(1,0,0),
                  '7':(0.9,0.0,0.0),
                  }

cat_color_dict = {'0':(0.2,0.4,0.4),
                  '1':(0.6,0.2,0.2),
                  '2':(0.8,0.1,0.1),
                  '3':(1,0,0),
                  '4':(0.6,0.6,0.2),
                  '5':(0.6,0.0,0.0),
                  '6':(1,0,0),
                  '7':(0.9,0.0,0.0),
                  }



extent = [-86.7,-84.3,41.5,44.5]
fig, axes = plt.subplots(2,3,figsize=(15,12),subplot_kw={'projection': ccrs.PlateCarree()})

for a,n in zip(axes.ravel(),(np.arange(0,6,1))):
    #this_title = plts[y]['title']
    a.set_extent(extent, crs=ccrs.PlateCarree())
    a.tick_params(axis='both', labelsize=8)
    for sh in shape_mini:
        a.add_feature(shape_mini[sh], facecolor='none', edgecolor='gray', linewidth=0.5)

    for key in map_plot_stations:
        #print(lon,lat,dat,c)
        this_lon = map_plot_stations[key]['lon']
        this_lat = map_plot_stations[key]['lat']
        
        this_dat = map_plot_stations[key]['snow'][n]
        this_c = cat_color_dict[str(this_dat)]
        
        a.scatter(this_lon,this_lat,s=((this_dat+2)*10),c=[this_c])

        a.set_title(map_plot_stations[key]['time_string'][n])

        #plt.text(lon+.03, lat+.03, key, fontsizs=e=10)
#plt.yticks(np.linspace(0,250,6,endpoint=True))
#plt.xticks(np.linspace(0,6,7,endpoint=True))
## End Test Plotting
