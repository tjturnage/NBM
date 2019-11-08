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
    # returns both a pandas time series and a string for plotting title
    hr = run_dt.hour
    if bulletin_type == 'nbstx':
        if (hr)%3 == 0:
            hours_ahead = 6
        elif (hr)%3 == 1:
            hours_ahead = 5
        elif (hr)%3 == 2:
            hours_ahead = 4
        fcst_hour_zero_utc = run_dt + timedelta(hours=hours_ahead)
        fcst_hour_zero_local = fcst_hour_zero_utc - timedelta(hours=5)
        pTime = pd.Timestamp(fcst_hour_zero_local)
        idx = pd.date_range(pTime, periods=23, freq='3H')

    elif bulletin_type == 'nbhtx':
        fcst_hour_zero_utc = run_dt + timedelta(hours=1)
        fcst_hour_zero_local = fcst_hour_zero_utc - timedelta(hours=5)
        #pTime = pd.Timestamp(fcst_hour_zero_utc)
        pTime = pd.Timestamp(fcst_hour_zero_local)
        idx2 = pd.date_range(pTime, periods=25, freq='H')
        idx = idx2[1:-1]
    else:
        pass
   
    return idx, idx[0], idx[-1]


def round_values(x,places,direction):
    amount = 10**places
    if direction == 'up':
        return int(math.ceil(x / float(amount))) * int(amount)
    if direction == 'down':
        return int(math.floor(x / float(amount))) * int(amount)
       
def download_directory_list():
    time_list= []
    now = datetime.utcnow()
    if int(now.hour) < 12:
        base_time = now.replace(hour=0)
    else:
        base_time = now.replace(hour=12)
    for t in range(1,4):
        next_time = base_time - timedelta(hours=(t*12))
        dir_segment = next_time.strftime("%d/%H")
        time_list.append(dir_segment)

    return time_list



def download_nbm_bulletin(url,fname,path_check):
    dst = os.path.join(base_dir,fname)
    if path_check != 'just_path':
        r = requests.get(url)
        print('downloading ... ' + str(url))
        open(dst, 'wb').write(r.content)
    return dst
        
    
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

# ensure image directory is created
try:
    os.makedirs(image_dir)
except:
    pass

import re
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from my_functions import GridShader
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
#from bs4 import BeautifulSoup
import requests
from datetime import datetime, timedelta


download = False
bulletin_type = 'nbhtx'
bulletin_type = 'nbstx'

now = datetime.utcnow()
now2 = now - timedelta(hours=3)
ymd = now2.strftime('%Y%m%d')
hour = now2.strftime('%H')
# sample url = https://para.nomads.ncep.noaa.gov/pub/data/nccf/com/blend/para/blend.20191105/07/text/blend_nbhtx.t07z
url = 'https://para.nomads.ncep.noaa.gov/pub/data/nccf/com/blend/para/blend.' + ymd + '/' + hour + '/text/blend_' + bulletin_type + '.t' + hour + 'z'




if bulletin_type == 'nbstx':
    products = ['t','wind','vis','sky_bar','ra_06','q06_bar', 's06_bar']
    fname = 'nbm_raw_short.txt'

elif bulletin_type == 'nbhtx':
    products = ['t','wind','vis','ra_01','sky_bar']
    fname = 'nbm_raw_hourly.txt'

# sample url = https://para.nomads.ncep.noaa.gov/pub/data/nccf/com/blend/para/blend.20191105/07/text/blend_nbhtx.t07z
url = 'https://para.nomads.ncep.noaa.gov/pub/data/nccf/com/blend/para/blend.' + ymd + '/' + hour + '/text/blend_' + bulletin_type + '.t' + hour + 'z'
#for s in ['KAMN','BDWM4','KBELD','KRQB','KCAD']:
for s in ['KAZO','KGRR','KMKG','KMOP','KCAD']:
#for s in ['KMQT']:
    column_list = []
    station_found = False
    station_name = s
    p = re.compile(station_name)
    s = re.compile('SOL')
    dt = re.compile('DT')
    ymdh = re.compile('[0-9]+/[0-9]+/[0-9]+\s+[0-9]+')

    # passing 'just_path' to this function only returns raw_file_path
    # without downloading anything

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
                #ymdh_match
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
    
    if bulletin_type == 'nbhtx': 
        nbm_old = pd.read_fwf(trimmed_nbm_file, widths=(5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3))
        elements = column_list[1:]

    elif bulletin_type == 'nbstx':
        nbm_old = pd.read_fwf(trimmed_nbm_file, skiprows=[0],widths=(5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3))
        elements = column_list[2:]

    # flip table so times are rows to align with pandas
    nbm = nbm_old.transpose()

    # after the flip, column names are useless. Use the created column_list before the flip
    # to make a dictionary that replaces bad column names with the original, pre-flip column names
    old_column_names = nbm.columns.tolist()
    col_rename_dict = {i:j for i,j in zip(old_column_names,elements)}
    nbm.rename(columns=col_rename_dict, inplace=True)
    #nbm.drop(nbm.index[1], inplace=True)
    
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

    # Now will start building a dictionary of products to plot
    # some products are only in a certain bulletin
    #   e.g., "nbm.X01" is hourly only, nbm.X06 is 6hourly and not in hourly product
    prods = {}
    
    # To plot time series lines  -- slice the dataframe with 'loc'
    # To plot a bar graph        -- convert the slice to a list.
    # Either can be done independently, but I usually do both to have the later 
    # option of plotting either way.
    
    # TMP and DPT will go on same panel, so using min(DPT) and max(TMP) to define bounds
    t = nbm.loc[:, ['TMP']]
    t_list = nbm.TMP.tolist()
    t_max = round_values(max(t_list),1,'up')
    
    dp = nbm.loc[:, ['DPT']]
    dp_list = nbm.DPT.tolist()
    dp_min = round_values(min(dp_list),1,'down')
    
    #bumping out the values a little to provide a buffer for yticks and ylabels
    tdp_min = dp_min - 5
    tdp_max = t_max + 5
    
    t_dp_yticks = []
    t_dp_ytick_labels = []
    #for r in (-20,-15,-10,-5,0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100):
    for r in (-20,-10,0,10,20,30,40,50,60,70,80,90,100):
        if r > tdp_min and r < tdp_max:
            t_dp_yticks.append(r)
            t_dp_ytick_labels.append(str(r))
    t_dp_min = t_dp_yticks[0] - 5
    t_dp_max = t_dp_yticks[-1] + 5
    
    wdir = nbm.loc[:, ['WDR']]
    wdir_list = nbm.WDR.tolist()
    
    wspd = nbm.loc[:, ['WSP']]
    wspd_list = nbm.WSP.tolist()
    wspd_list = [round(x) for x in wspd_list]
    
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
    prob_yticks = [0, 20, 40, 60, 80, 100]
    prob_ytick_labels = ["0","20", "40","60","80","100"]
    p_min = -5
    p_max = 105

    p_ra = nbm.loc[:, ['PRA']]
    p_ra_list = nbm.PRA.tolist()
    prods['p_ra'] = {'data': p_ra, 'color':(0.2, 0.8, 0.2, 0.6), 'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Prob Rain\n(%)'}
    prods['p_ra_bar'] = {'data': p_ra_list, 'color':(0.2, 0.8, 0.2, 0.6), 'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Prob Rain\n(%)'}


    p_sn = nbm.loc[:, ['PSN']]
    p_sn_list = nbm.PSN.tolist()
    prods['p_sn'] = {'data': p_sn, 'color':(0.2, 0.2, 0.8, 0.6), 'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Prob Rain\n(%)'}
    prods['p_sn_bar'] = {'data': p_sn_list, 'color':(0.2, 0.2, 0.8, 0.6), 'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Prob Snow\n(%)'}


    p_zr = nbm.loc[:, ['PZR']]
    p_zr_list = nbm.PZR.tolist()
    prods['p_zr'] = {'data': p_zr, 'color':(0.6, 0.4, 0.2, 0.6), 'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Prob ZR\n(%)'}
    prods['p_zr_bar'] = {'data': p_zr_list, 'color':(0.6, 0.4, 0.2, 0.6), 'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Prob ZR\n(%)'}

    p_pl = nbm.loc[:, ['PPL']]
    p_pl_list = nbm.PPL.tolist()
    prods['p_pl'] = {'data': p_pl, 'color':(0.2, 0.8, 0.2, 0.6), 'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Prob PL\n(%)'}
    prods['p_pl_bar'] = {'data': p_pl_list, 'color':(0.2, 0.8, 0.2, 0.6), 'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Prob PL\n(%)'}

   
    qpf_color = (0.1, 0.9, 0.1, 0.8)
      
    try:
        # hourly snow amount, convert integers to tenths of an inch
        nbm.S01 = nbm.S01.multiply(0.1)
        s01_list = nbm.S01.tolist()
        prods['s01_bar'] = {'data': s01_list, 'color':(0.1, 0.1, 0.7, 0.7), 'ymin':0.0,'ymax':1.01,'yticks':[0,0.25,0.5,0.75,1], 'ytick_labels':['0','1/4','1/2','3/4','1'], 'title':'Hourly snow\n(inches)' }

        # hourly precip amount, convert integers to hundredths of an inch
        nbm.Q01 = nbm.Q01.multiply(0.01)
        q01_list = nbm.Q01.tolist()
        prods['q01_bar'] = {'data': q01_list, 'color':qpf_color, 'ymin':0.0,'ymax':0.50,'yticks':[0.0,0.1,0.2,0.3], 'ytick_labels':['0','0.1','0.2','0.3'],'title':'Precip\nAmount\n(inches)'}
 
        #  P01 is one hour Probability for any type of precip, aka PPI
        p01_list = nbm.P01.tolist()
        prods['p01_bar'] = {'data': p01_list, 'color':(0.5, 0.5, 0.5, 0.5), 'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Precip\nChances\n(%)'}

        #multiply P01 by ProbRain to get absolute rain probability
        #multiply P01 to ProbSnow to get absolute snow probability
        nbm['RA_01'] = nbm['P01'] * nbm['PRA']/100
        nbm['SN_01'] = nbm['P01'] * nbm['PSN']/100

        ra_01 = nbm.loc[:, ['RA_01']]
        sn_01 = nbm.loc[:, ['SN_01']]

        ra_01_list = np.multiply(p01_list,p_ra_list)/100
        sn_01_list = np.multiply(p01_list,p_sn_list)/100
        
        prods['ra_01_bar'] = {'data': ra_01_list, 'color':(0.4, 0.7, 0.4, 0.8), 'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Abs Prob\nRain (%)'}
        prods['sn_01_bar'] = {'data': sn_01_list, 'color':(0.3, 0.3, 0.8, 0.8), 'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Abs Prob\nSnow(%)'}
        prods['ra_01'] = {'data': ra_01, 'color':(0.4, 0.7, 0.4, 0.8), 'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Abs Prob\nRain (%)'}
        prods['sn_01'] = {'data': sn_01, 'color':(0.3, 0.3, 0.8, 0.8), 'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Abs Prob\nSnow(%)'}


    except:
        pass
    
    try:
        p06_list = nbm.P06.tolist()
        prods['p06_bar'] = {'data': p06_list, 'color':(0.2, 0.8, 0.2, 0.6), 'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels,'title':'Precip\nChances\n(%)'}
        p06 = nbm.loc[:, ['P06']]
        prods['p06'] = {'data': p06, 'color':(0.2, 0.8, 0.2, 0.6), 'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels,'title':'Precip\nChances\n(%)'}

        nbm.S06 = nbm.S06.multiply(0.1)
        s06 = nbm.loc[:, ['S06']]
        s06_list = nbm.S06.tolist()        

        nbm.Q06 = nbm.Q06.multiply(0.01)
        q06 = nbm.loc[:, ['Q06']]
        q06_list = nbm.Q06.tolist()

        #multiply P06 by ProbRain to get absolute rain probability
        #multiply P06 to ProbSnow to get absolute snow probability
        nbm['RA_06'] = nbm['P06'] * nbm['PRA']/100
        nbm['SN_06'] = nbm['P06'] * nbm['PSN']/100

        ra_06 = nbm.loc[:, ['RA_06']]
        ra_06.dropna(inplace=True)
        sn_06 = nbm.loc[:, ['SN_06']]
        sn_06.dropna(inplace=True)

        ra_06_list = np.multiply(p06_list,p_ra_list)/100
        sn_06_list = np.multiply(p06_list,p_sn_list)/100
    except:
        print('no luck with absolute p06')

    try:       
        prods['ra_06_bar'] = {'data': ra_06_list, 'color':(0.4, 0.7, 0.4, 0.8), 'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Abs Prob\nRain (%)'}
        prods['sn_06_bar'] = {'data': sn_06_list, 'color':(0.3, 0.3, 0.8, 0.8), 'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Abs Prob\nSnow(%)'}
        prods['ra_06'] = {'data': ra_06, 'color':(0.4, 0.7, 0.4, 0.8), 'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Abs Prob\nRain (%)'}
        prods['sn_06'] = {'data': sn_06, 'color':(0.3, 0.3, 0.8, 0.8), 'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Abs Prob\nSnow(%)'}

        prods['q06'] = {'data': q06, 'color':qpf_color, 'ymin':0.0,'ymax':0.50,'yticks':[0.0,0.1,0.2,0.3], 'ytick_labels':['0','0.1','0.2','0.3'],'title':'Precip\nAmount\n(inches)'}
        prods['s06'] = {'data': s06, 'color':(0.1, 0.1, 0.7, 0.7), 'ymin':0.0,'ymax':3.01, 'yticks':[0,1,2,3], 'ytick_labels':['0','1','2','3','4','5','6'],'title':'6hr Snow\n(inches)' }
        prods['q06_bar'] = {'data': q06_list, 'color':qpf_color, 'ymin':0.0,'ymax':0.50,'yticks':[0.0,0.1,0.2,0.3], 'ytick_labels':['0','0.1','0.2','0.3'],'title':'Precip\nAmount\n(inches)'}
        prods['s06_bar'] = {'data': s06_list, 'color':(0.1, 0.1, 0.7, 0.7), 'ymin':0.0,'ymax':prods['s06']['ymax'], 'yticks':prods['s06']['yticks'], 'ytick_labels':prods['s06']['yticks'],'title':'6hr Snow\n(inches)' }
    except:
        print('no luck with prods')

    finally:
        pass
    
    
    prods['wind'] = {'data': wspd, 'color':(0.5, 0.5, 0.5, 0.8),'title':'Wind\nSpeed\n& Gust'}
    prods['t'] = {'data': t, 'color':(0.8, 0.0, 0.0, 0.8), 'ymin':t_dp_min,'ymax':t_dp_max,'yticks':t_dp_yticks,'ytick_labels':t_dp_ytick_labels, 'title':'Temerature\nDewpoint' }
    prods['dp'] = {'data': dp, 'color':(0.0, 0.9, 0.1, 0.6), 'ymin':t_dp_min,'ymax':t_dp_max,'yticks':t_dp_yticks,'ytick_labels':t_dp_ytick_labels, 'title':'Temerature\nDewpoint'  }
    prods['sky'] = {'data': sky, 'color':(0.6, 0.6, 0.6, 0.6), 'ymin':-5,'ymax':105,'yticks':[0,25,50,75,100],'ytick_labels':['0','25','50','75','100'], 'title':'Sky cover\n(%)'}
    prods['vis'] = {'data': vis, 'color':(0.7, 0.7, 0.3, 1), 'ymin':-0.5,'ymax':8,'yticks':[1, 3, 5, 7],'ytick_labels':['1','3','5', '>6'], 'title':'Visibility\n(miles)'}


    prods['t_bar'] = {'data': t_list, 'color':(0.8, 0.0, 0.0, 0.8), 'ymin':t_dp_min,'ymax':t_dp_max,'yticks':t_dp_yticks,'ytick_labels':t_dp_ytick_labels, 'title':'Temerature\nDewpoint' }
    prods['dp_bar'] = {'data': dp_list, 'color':(0.0, 0.9, 0.1, 0.6), 'ymin':t_dp_min,'ymax':t_dp_max,'yticks':t_dp_yticks,'ytick_labels':t_dp_ytick_labels, 'title':'Temerature\nDewpoint'  }
    prods['sky_bar'] = {'data': sky_list, 'color':(0.6, 0.6, 0.6, 0.6), 'ymin':-5,'ymax':105,'yticks':[0,25,50,75,100],'ytick_labels':['0','25','50','75','100'], 'title':'Sky cover\n(%)'}
    prods['vis_bar'] = {'data': vis_list, 'color':(0.7, 0.7, 0.3, 1), 'ymin':-0.5,'ymax':8,'yticks':[1, 3, 5, 7],'ytick_labels':['1','3','5', '>6'], 'title':'Visibility\n(miles)'}
    
    hours = mdates.HourLocator()
    myFmt = DateFormatter("%d%h")
    myFmt = DateFormatter("%d%b\n%HZ")
    myFmt = DateFormatter("%I\n%p")
    fig, axes = plt.subplots(len(products),1,figsize=(16,10),sharex=True,subplot_kw={'xlim': (start_time,end_time)})
    #fig, axes = plt.subplots(len(products),1,figsize=(16,8),sharex=True,subplot_kw={'xlim': (start_time,end_time)})
    plt.subplots_adjust(bottom=0.1, left=0.25, top=0.9)
    plt.suptitle('NBM hourly Guidance - ' + station_name + '\n' + start_time.strftime('%B %d, %Y -- %I %p EST'))
    
    
    for y,a in zip(products,axes.ravel()):
        if bulletin_type == 'nbstx':
            #if offset == 
            a.xaxis.set_major_locator(mdates.HourLocator(byhour=[7,19]))
            a.xaxis.set_minor_locator(mdates.HourLocator(byhour=[1,4,10,13,16,22]))
            a.xaxis.set_major_formatter(DateFormatter("%I\n%p\n%m/%d"))
            a.xaxis.set_minor_formatter(DateFormatter("%I\n%p"))
        else:
            a.xaxis.set_major_locator(hours)
            a.xaxis.set_major_formatter(myFmt)

        a.yaxis.set_label_coords(-0.08,0.25)
        if start_time.hour > 16:
            first_gray = True
        else:
            first_gray = False            
        if y == 't':
            a.grid()
            a.get_xaxis().set_visible(False)
            gs = GridShader(a, facecolor="lightgrey", first=first_gray, alpha=0.5)            
            a.set_ylim(prods[y]['ymin'],prods[y]['ymax'])
            a.set(yticks = prods[y]['yticks'], yticklabels = prods[y]['ytick_labels'])
            a.set_ylabel(prods[y]['title'], rotation=0)
            a.plot(prods['dp']['data'],linewidth=2,color=prods['dp']['color'])
            a.plot(prods['t']['data'],linewidth=2,color=prods['t']['color'])

        if y == 'ra_01':
            a.grid()
            a.get_xaxis().set_visible(False)
            gs = GridShader(a, facecolor="lightgrey", first=first_gray, alpha=0.5) 
            a.set_ylim(prods[y]['ymin'],prods[y]['ymax'])
            a.set(yticks = prods[y]['yticks'], yticklabels = prods[y]['ytick_labels'])
            a.set_ylabel(prods[y]['title'], rotation=0)
            a.plot(prods['ra_01']['data'],linewidth=2,color=prods['ra_01']['color'])
            a.plot(prods['sn_01']['data'],linewidth=2,color=prods['sn_01']['color'])

        if y == 'ra_06':
            a.grid()
            a.get_xaxis().set_visible(False)
            gs = GridShader(a, facecolor="lightgrey", first=first_gray, alpha=0.5) 
            a.set_ylim(prods[y]['ymin'],prods[y]['ymax'])
            a.set(yticks = prods[y]['yticks'], yticklabels = prods[y]['ytick_labels'])
            a.set_ylabel(prods[y]['title'], rotation=0)
            a.plot(prods['ra_06']['data'],linewidth=2,color=prods['ra_06']['color'])
            a.plot(prods['sn_06']['data'],linewidth=2,color=prods['sn_06']['color'])
    
        if y == 'wind':
            plt.rc('font', size=12) 
            a.set_ylim(0,1)
            a.set_ylabel(prods[y]['title'], rotation=0)
            a.get_xaxis().set_visible(False)
            gs = GridShader(a, facecolor="lightgrey", first=first_gray, alpha=0.5)
    
            for s,d,g,p in zip(wspd_list,wdir_list,wgst_list,pd_series):
                u,v,dx,dy = u_v_components(d,s)
                a.barbs(p, 0.7, u, v, length=7, color=[0,0,1,0.9], pivot='middle')
                a.text(p, 0.28, f'{s}',horizontalalignment='center',color=[0,0,1,0.9])
                a.text(p, 0.08, f'{g}',horizontalalignment='center',color=[0.6,0,1,0.6])
                
        # these are dataframe slices that use matplotlib plot to create time series
        if y in ['p01','p06','p_zr','p_pl','sky','vis']:
            gs = GridShader(a, facecolor="lightgrey", first=first_gray, alpha=0.5) 
            a.set_ylim(prods[y]['ymin'],prods[y]['ymax'])
            a.set(yticks = prods[y]['yticks'], yticklabels = prods[y]['ytick_labels'])
            a.plot(prods[y]['data'],color=prods[y]['color'])
            a.set_ylabel(prods[y]['title'], rotation=0)
            a.set(yticks = prods[y]['yticks'], yticklabels = prods[y]['ytick_labels'])
    
        # these are lists that use matplotlib bar to create bar graphs
        if y in ['p01_bar','q01_bar','s01_bar','p06_bar','q06_bar','s06_bar', 'sky_bar','p_zr_bar','p_sn_bar','p_pl_bar','p_ra_bar','sn_01_bar','ra_01_bar','vis_bar']:
            gs = GridShader(a, facecolor="lightgrey", first=first_gray, alpha=0.5) 
            a.set_ylim(prods[y]['ymin'],prods[y]['ymax'])
            if bulletin_type == 'nbstx':
                if y == 'sky_bar':
                    a.bar(pd_series,prods[y]['data'],width=1/(50/3), align="center",color=prods[y]['color'])
                else:
                    a.bar(pd_series,prods[y]['data'],width=1/(50/3), align="center",color=prods[y]['color'])            
            else:
                a.bar(pd_series,prods[y]['data'],width=1/25, align="edge",color=prods[y]['color'])

            a.set_ylabel(prods[y]['title'], rotation=0)
            a.set(yticks = prods[y]['yticks'], yticklabels = prods[y]['ytick_labels'])
    
    image_file = station_name + '_NBM_' + bulletin_type + '.png'
    image_dst_path = os.path.join(image_dir,image_file)
    plt.savefig(image_dst_path,format='png')

