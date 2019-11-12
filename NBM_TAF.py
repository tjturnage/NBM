# -*- coding: utf-8 -*-
"""
Grabs NBM V3.2 houlry bulletins from:
#https://sats.nws.noaa.gov/~downloads/nbm/bulk-textv32/current/
#https://para.nomads.ncep.noaa.gov/pub/data/nccf/noaaport/blend/blend_nbstx.t15z.tran
https://para.nomads.ncep.noaa.gov/pub/data/nccf/com/blend/para/blend.20191105/07/text/blend_nbhtx.t07z
    
    Bulletin Key:
    https://www.weather.gov/mdl/nbm_textcard_v32#nbh
Elements list here for reference:
TMP DPT SKY WDR WSP GST P01 P06 Q01 DUR T01 PZR PSN PPL PRA S01 SLV I01

CIG VIS LCB
VIS = visibility, 1/10th miles (rounded to nearest mile for values >= 1 mile)


Ignoring these fire weather elements
MHT TWD TWS HID
"""

#https://www.weather.gov/mdl/nbm_text?ele=NBH&cyc=00&sta=KGRR&download=yes

import os
import sys

def dtList_nbm(run_dt,bulletin_type):
    """
      Create pandas date range of forecast valid times based on bulletin
      issuance time and bulletin type.
    
      Parameters
      ----------
        run_dt : python datetime object
                 Contains yr,mon,date,hour associated with bulletin issue time

  bulletin_type: string
                 'nbstx' -- short term guidance (3 hourly)
                 'nbhtx' -- hourly guidance ( hourly)

                 This is required to define model forecast hour start and 
                 end times as a well as forecast hour interval.
                  
      Returns
      -------
           pandas date/time range to be used as index as well as start/end times
    """

    fcst_hour_zero_utc = run_dt + timedelta(hours=1)
    fcst_hour_zero_local = fcst_hour_zero_utc - timedelta(hours=5)
    #pTime = pd.Timestamp(fcst_hour_zero_utc)
    pTime = pd.Timestamp(fcst_hour_zero_local)
    idx2 = pd.date_range(pTime, periods=26, freq='H')
    idx = idx2[1:-1]

    return idx, idx[0], idx[-1]


def round_values(x,places,direction):
    amount = 10**places
    if direction == 'up':
        return int(math.ceil(x / float(amount))) * int(amount)
    if direction == 'down':
        return int(math.floor(x / float(amount))) * int(amount)
       

def download_nbm_bulletin(url,fname,path_check):
    dst = os.path.join(base_dir,fname)
    if path_check != 'just_path':
        r = requests.get(url)
        print('downloading ... ' + str(url))
        open(dst, 'wb').write(r.content)
    return dst

def add_zero(value,places):
    check1 = 10**places
    check2 = 10**(places-1)
    val = str(int(value))
    if value < check2:
        val = '0' + val
    if value < check1:
        val = '0' + val    
    return val


def cig_round(hgt):
    if hgt > 200:
        hgt = 250
    elif hgt > 120:
        hgt = 150
    elif hgt > 100:
        hgt = 120

        

def calc_cig(cig,sky,lowest_base):
    if sky > 85:
        desc = 'OVC'
    elif sky > 40:
        desc = 'BKN'
    elif sky > 10:
        desc = 'SCT'
    elif sky > 1:
        desc = 'FEW'
    else:
        return 'SKC'

    if cig > 800:
        cig_desc = ''
        cig_str = ''
    elif sky > 85:
        cig_desc = 'OVC'
        cig_str = add_zero(cig,2)
    elif sky > 40 and sky <= 85:
        cig_desc = 'BKN'
        cig_str = add_zero(cig,2)

    cig_final = cig_desc + cig_str 
            

    if lowest_base < cig:
        if sky > 45:
            lb_desc = 'BKN'
        else:
            lb_desc = 'SCT'
        
        lb_str = add_zero(lowest_base,2)

    else:
        lb_str = ''

    lb_final = lb_desc + lb_str

    full_cig_str = lb_final + ' ' + cig_final
    return full_cig_str

def calc_vis(vis):
    if vis > 6:
        vis_str = 'P6SM'
    elif vis >= 3:
        vis_str = str(int(vis)) + 'SM'
    elif vis >= 2.5:
        vis_str = '2 1/2SM'
    elif vis >= 2:
        vis_str = '2SM'
    elif vis >= 1.75:
        vis_str = '1 3/4SM'
    elif vis >= 1.5:
        vis_str = '1 1/2SM'
    elif vis >= 1.25:
        vis_str = '1 1/4SM'
    elif vis >= 1:
        vis_str = '1SM'
    elif vis >= 0.75:
        vis_str = '3/4SM'
    elif vis >= 0.5:
        vis_str = '1/2SM'
    elif vis >= 0.25:
        vis_str = '1/4SM'
    else:
        vis_str = 'M1/4SM'
    
    return vis_str
      
def u_v_components(wdir, wspd):
    # since the convention is "direction from"
    # we have to multiply by -1
    # If an arrow is drawn, it needs a dx of 2/(number of arrows) to fit in the row of arrows
    u = (math.sin(math.radians(wdir)) * wspd) * -1.0
    v = (math.cos(math.radians(wdir)) * wspd) * -1.0
    dx = math.sin(math.radians(wdir)) * -0.4 / (12)
    dy = math.cos(math.radians(wdir)) * -0.4
    return u,v, dx, dy

try:
    os.listdir('/usr')
    windows = False
    base_dir = '/data'
    sys.path.append('/data/scripts/resources')


    raw_nbm_file = os.path.join(base_dir,'nbm_raw.txt')
    trimmed_nbm_file =  os.path.join(base_dir,'nbm_trimmed.txt')

except:
    windows = True
    base_dir = 'C:/data'    
    image_dir = os.path.join(base_dir,'images','NBM')
    raw_nbm_file = os.path.join(base_dir,'nbm_raw.txt')
    trimmed_nbm_file =  os.path.join(base_dir,'nbm_trimmed.txt')
    sys.path.append('C:/data/scripts/resources')

# ensure image directory is created


import re
import math
import pandas as pd
import numpy as np


from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import requests
from datetime import datetime, timedelta


download = False
bulletin_type = 'nbhtx'
fname = 'nbm_raw_hourly.txt'

if download:
    now = datetime.utcnow()
    now2 = now - timedelta(hours=3)
    ymd = now2.strftime('%Y%m%d')
    hour = now2.strftime('%H')
    url = 'https://para.nomads.ncep.noaa.gov/pub/data/nccf/com/blend/para/blend.20191107/12/text/blend_nbhtx.t12z'
    #url = 'https://para.nomads.ncep.noaa.gov/pub/data/nccf/com/blend/para/blend.' + ymd + '/' + hour + '/text/blend_' + bulletin_type + '.t' + hour + 'z'

fin = 'C:/data/scripts/NBM/NBM_stations.txt'
fout = 'C:/data/scripts/NBM/NBM_MI_stations.txt'
station_info = []
with open(fin,'r') as src:
    with open(fout,'w') as dst:
        for line in src:
            elements = line.split(',')
            if elements[2] == 'MI' and float(elements[3]) < 44.5 and float(elements[4]) > -86.8:
                stn = [elements[0],float(elements[3]),float(elements[4])]
                station_info.append(elements[0])
                station_info.append(float(elements[3]))
                station_info.append(float(elements[4]))     
stations = station_info[0::3]

#for s in ['KAMN','BDWM4','KBELD','KRQB','KCAD']:
for st in ['KCAD']:
#for s in ['KMQT']:
    column_list = []
    station_found = False
    station_name = st
    p = re.compile(station_name)
    s = re.compile('SOL')
    dt = re.compile('DT')
    ymdh = re.compile('[0-9]+/[0-9]+/[0-9]+\s+[0-9]+')

    if download:
        raw_file_path = download_nbm_bulletin(url,fname,'hi')
        download = False
        
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
                pd_series,start_time,end_time = dtList_nbm(run_dt,bulletin_type)
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
    

    nbm_old = pd.read_fwf(trimmed_nbm_file, widths=(5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3))
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
        nbm.drop(['FHR'], inplace=True)
    finally:
        pass

    nbm.set_index(pd_series, inplace=True)


    
    wdir = nbm.loc[:, ['WDR']]
    wdir_list = nbm.WDR.tolist()

    for w in range(0,len(wdir_list)):
        wdir_list[w] = round(float(wdir_list[w]))
        if wdir_list[w] < 10:
            wdir_list[w] = '0' + str(wdir_list[w])
        else:
            wdir_list[w] = str(wdir_list[w])
        wdir_list[w] = wdir_list[w] + '0'
        #print(wdir_list[w])
    
    wspd = nbm.loc[:, ['WSP']]
    wspd_list = nbm.WSP.tolist()
    for s in range(0,len(wdir_list)):
        wspd_list[s] = round(float(wspd_list[s]))
        if wspd_list[s] < 10:
            wspd_list[s] = '0' + str(wspd_list[s])
        else:
            wspd_list[s] = str(wspd_list[s])          

    
    wgst = nbm.loc[:, ['GST']]
    wgst_list = nbm.GST.tolist()
    wgst_list = [round(x) for x in wgst_list]
    
    sky = nbm.loc[:, ['SKY']]
    sky_list = nbm.SKY.tolist()
    
    # sometimes we have to convert units because they're in tenths or hundredths
    nbm.VIS = nbm.VIS.multiply(0.1)
    vis = nbm.loc[:, ['VIS']]
    vis.clip(upper=7.0,inplace=True)
    vis_list = nbm.VIS.tolist()
    
    # conditional probabilities for rain (PRA), snow (PSN), freezing rain (PZR), sleet (PPL)
    # define y axis range and yticks/ylabels for any element that's probabilistic 

    #p_ra = nbm.loc[:, ['PRA']]
    p01_list = nbm.P01.tolist()
    p_ra_list = nbm.PRA.tolist()
    p_sn_list = nbm.PSN.tolist()
    p_zr_list = nbm.PZR.tolist()
    p_pl_list = nbm.PPL.tolist()

    ra_01_list = np.multiply(p01_list,p_ra_list)/100
    sn_01_list = np.multiply(p01_list,p_sn_list)/100

    cig_list = nbm.CIG.tolist()
    lcb_list = nbm.LCB.tolist()


    nbm.S01 = nbm.S01.multiply(0.1)
    s01_list = nbm.S01.tolist()
    nbm.Q01 = nbm.Q01.multiply(0.01)
    q01_list = nbm.Q01.tolist()

 
    
    hours = mdates.HourLocator()
    myFmt = DateFormatter("%d%h")
    myFmt = DateFormatter("%d%b\n%HZ")
    myFmt = DateFormatter("%I\n%p")

    for h in range(0,len(cig_list)):
        wg = wgst_list[h]
        ws = wspd_list[h]
        vis = vis_list[h]
        vis_str = calc_vis(vis)
        sky = sky_list[h]
        lowest_base = lcb_list[h]
        cig = cig_list[h]
        full_cig_str = calc_cig(cig,sky,lowest_base)
        s01 = s01_list[h]
        ra_01 = ra_01_list[h]      
        p_ra = p_ra_list[h]
        q01 = q01_list[h]
        

        if wg > 10 and ((int(wg) - int(ws)) > 7):
        #if wg > 12:
            wg_str = 'G' + str(wg)

        else:
            wg_str = ''

        if s01 > 0:
           sn  = 'SN'
        else:
            sn = ''
        
        if q01 > 0 and p_ra > 30:
            ra = 'RA'
        else:
            ra = ''
        
        wx_str = ra + sn
        wind_str = str(int(wdir_list[h])) + str(wspd_list[h]) + wg_str + 'KT'
        full_str = wind_str + ' ' + vis_str + ' ' + wx_str + ' ' + full_cig_str
        print(full_str)



"""
KGRR 091130Z 0912/1012 23012G20KT P6SM BKN150
     FM091600 23012G21KT P6SM BKN200
     FM092100 23012KT P6SM BKN120
     FM100600 23006KT P6SM OVC025=
"""
    

