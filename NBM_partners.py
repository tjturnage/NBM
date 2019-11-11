# -*- coding: utf-8 -*-
"""
Grabs NBM V3.2 houlry bulletins from:
#https://sats.nws.noaa.gov/~downloads/nbm/bulk-textv32/current/
#https://para.nomads.ncep.noaa.gov/pub/data/nccf/noaaport/blend/blend_nbstx.t15z.tran
https://para.nomads.ncep.noaa.gov/pub/data/nccf/com/blend/para/blend.20191105/07/text/blend_nbhtx.t07z
    
    Bulletin Key:
    https://www.weather.gov/mdl/nbm_textcard_v32#nbh

"""

#https://www.weather.gov/mdl/nbm_text?ele=NBH&cyc=00&sta=KGRR&download=yes

import os
import sys

def dtList_nbm(run_dt,bulletin_type,tz_shift):
    """
      Create pandas date range of forecast valid times based on bulletin
      issuance time and bulletin type.
    
      Parameters
      ----------
        run_dt : python datetime object
                 Contains yr,mon,date,hour associated with bulletin issue time

  bulletin_type: string

                 'nbhtx' -- hourly guidance ( hourly)

                 This is required to define model forecast hour start and 
                 end times as a well as forecast hour interval.
                  
      Returns
      -------
           pandas date/time range to be used as index as well as start/end times
    """

    fcst_hour_zero_utc = run_dt + timedelta(hours=0)
    fcst_hour_zero_local = fcst_hour_zero_utc - timedelta(hours=tz_shift)
    #pTime = pd.Timestamp(fcst_hour_zero_utc)
    pTime = pd.Timestamp(fcst_hour_zero_local)
    idx = pd.date_range(pTime, periods=27, freq='H')

    return idx, fcst_hour_zero_local


def round_values(x,places,direction):
    amount = 10**places
    if direction == 'up':
        return int(math.ceil(x / float(amount))) * int(amount)
    if direction == 'down':
        return int(math.floor(x / float(amount))) * int(amount)

def wind_chill(t,s):
    """
    Returns wind chill in degress F
    Inputs:
        t   : temperatures in degrees F
        s   : wspd in MPH
    """    
    wc = 0.6215*t + 35.74 - 35.75*(s**0.16) + (0.4275*t)*(s**0.16)


    if wc >= -15:
        fbt = 4

    if wc < -15:
        fbt = 3

    if wc < -35:
        if wspd > 20:
            fbt = 2
        else:
            fbt = 3

    if wc < -45:
        if wspd > 10:
            fbt = 2
        else:
            fbt = 3
                
    return round(wc), fbt


def categorize(data_list,element):
    """dat
 [0,1,2,3,4,5,6],'major_yticks_labels':['0.00','0.01','0.05','0.10','0.2','0.3','0.5'], 
        zr_list   : one hourly snow in hundredths
    
    """

    vis_dict = {'0.0':0,'0.3':1,'0.6':2,'1':3,'2':4,'4':3,'6':6}
    zr_dict = {'-0.1':0,'1':1,'8':2,'18':3,'28':4,'45':5}
    sn_dict = {'-0.1':0,'0.05':1,'0.15':2,'0.45':3,'0.75':4,'0.95':5,'1.35':6,'1.85':7}  
    #sn_dict = {'-0.1':0,'0.05':1,'0.15':2,'0.33':3,'0.75':4,'1.5':5,'2.5':6}    

    if element == 'sn':
        data_dict = sn_dict
    if element == 'zr':
        data_dict = zr_dict    
    if element == 'vis':
        data_dict = vis_dict    

    category_list = []  
    for x in range(0,len(data_list)):
        val = data_list[x]
        for key in data_dict:
            if val > float(key):
                x_cat = data_dict[key]
            
        category_list.append(x_cat)

    return category_list

def temperature_bounds(t_list,wc_list):
    max_val = np.max(t_shifted_list)
    min_val = np.min(wc_list)
    high_list = np.arange(100,0,-10)
    low_list = np.arange(0,100,10)
    for h in range(0,(len(high_list))):
        if high_list[h] > max_val:
            upper_limit = high_list[h]
        else:
            break
    upper_limit = upper_limit + 10     

    for l in range(0,(len(low_list))):
        if low_list[l] < min_val:
            lower_limit = low_list[h]
        else:
            break

    lower_limit = lower_limit - 20  
    
    tick_list = np.arange(lower_limit,upper_limit,10)
    tick_labels = []
    for t in range(0,len(tick_list)):
        tick_label = str(int(tick_list[t] - 30))
        tick_labels.append(tick_label)
    print(tick_list,tick_labels)
    
    return tick_list,tick_labels    

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
import requests
from datetime import datetime, timedelta
import matplotlib.transforms

download = True
bulletin_type = 'nbhtx'

now = datetime.utcnow()
now2 = now - timedelta(hours=3)
ymd = now2.strftime('%Y%m%d')
hour = now2.strftime('%H')
#url = 'https://para.nomads.ncep.noaa.gov/pub/data/nccf/com/blend/para/blend.20191107/15/text/blend_nbhtx.t15z'
url = 'https://para.nomads.ncep.noaa.gov/pub/data/nccf/com/blend/para/blend.' + ymd + '/' + hour + '/text/blend_' + bulletin_type + '.t' + hour + 'z'


state2timezone = { 'AK': 'US/Alaska', 'AL': 'US/Central', 'AR': 'US/Central', 'AS': 'US/Samoa', 'AZ': 'US/Mountain', 'CA': 'US/Pacific', 'CO': 'US/Mountain', 'CT': 'US/Eastern', 'DC': 'US/Eastern', 'DE': 'US/Eastern', 'FL': 'US/Eastern', 'GA': 'US/Eastern', 'GU': 'Pacific/Guam', 'HI': 'US/Hawaii', 'IA': 'US/Central', 'ID': 'US/Mountain', 'IL': 'US/Central', 'IN': 'US/Eastern', 'KS': 'US/Central', 'KY': 'US/Eastern', 'LA': 'US/Central', 'MA': 'US/Eastern', 'MD': 'US/Eastern', 'ME': 'US/Eastern', 'MI': 'US/Eastern', 'MN': 'US/Central', 'MO': 'US/Central', 'MP': 'Pacific/Guam', 'MS': 'US/Central', 'MT': 'US/Mountain', 'NC': 'US/Eastern', 'ND': 'US/Central', 'NE': 'US/Central', 'NH': 'US/Eastern', 'NJ': 'US/Eastern', 'NM': 'US/Mountain', 'NV': 'US/Pacific', 'NY': 'US/Eastern', 'OH': 'US/Eastern', 'OK': 'US/Central', 'OR': 'US/Pacific', 'PA': 'US/Eastern', 'PR': 'America/Puerto_Rico', 'RI': 'US/Eastern', 'SC': 'US/Eastern', 'SD': 'US/Central', 'TN': 'US/Central', 'TX': 'US/Central', 'UT': 'US/Mountain', 'VA': 'US/Eastern', 'VI': 'America/Virgin', 'VT': 'US/Eastern', 'WA': 'US/Pacific', 'WI': 'US/Central', 'WV': 'US/Eastern', 'WY': 'US/Mountain', '' : 'US/Pacific', '--': 'US/Pacific' }
time_shift_dict = {'US/Eastern':4,'US/Central':5,'US/Mountain':6,'US/Pacific':7,'US/Hawaii':7,'US/Alaska':8,'Pacific/Guam':9,'America/Puerto_Rico':3,'America/Virgin':3}

    
fin = 'C:/data/scripts/NBM/NBM_stations.txt'
fout = 'C:/data/scripts/NBM/NBM_MI_stations.txt'

station_master = {}
with open(fin,'r') as src:
    with open(fout,'w') as dst:
        for line in src:
            elements = line.split(',')
            station_id = str(elements[0])
            station_name = str(elements[1])
            state = str(elements[2])
            if state in state2timezone.keys():
                lat = float(elements[3])
                lon = float(elements[4])
                utc_shift = time_shift_dict[state2timezone[state]]
                #print(station,state,lat,lon,time_shift_dict[utc_shift])
                #station_info[station] = ([('state', state) , ('utc_shift', time_shift_dict[utc_shift]) ,('lat', lat) , ('lon' , 20)] )
                #station_master[station] = ([('state',state),('time_shift',utc_shift),('lat',lat),('lon',lon)])
                station_master[station_id] = ({'name':station_name,'state':state,'time_shift':utc_shift,'lat':lat,'lon':lon})

            else:
                utc_shift = 0

station_mini = {}
lats = []
lons = []
names = []
plt.figure(figsize=(16, 16))
for key in station_master:
    station_id = key
    station_description = station_master[key]['name']
    lat = station_master[key]['lat']
    lon = station_master[key]['lon']
    if station_master[key]['state'] == 'MI':
        #if station_master[key]['lat'] > 44.5 and station_master[key]['lon'] > -88:

        lats.append(lat)
        lons.append(lon)
        names.append(key)        
        station_mini[key] = station_master[key]
        lats_tuple = tuple(lats)    
        lons_tuple = tuple(lons)
        if lat < 44.5 and lon > -86.7 and lon < -84.3:
            plt.plot(station_master[key]['lon'],station_master[key]['lat'],'ro')
            plt.text(lon+.03, lat+.03, key, fontsize=10)
        else:
            pass
plt.xlim(-86.7,-84.3)
plt.ylim(41.5,44.5)
#plt.yticks(np.linspace(0,250,6,endpoint=True))
#plt.xticks(np.linspace(0,6,7,endpoint=True))

#products = ['t_bar','wc_bar','time_fb_bar','wind','vis_bar','s01_bar']
products = ['t_bar','time_fb_bar','wind','vis_test','sn_test','zr_test']
fname = 'nbm_raw_hourly.txt'
#fname = 'C:/data/scripts/NBM/20191110/blend_nbhtx.t18z'
#C:\data\scripts\NBM\20191109
for key in ('KMBL','KCAD','KHTL','KLDM','BDWM4','EVAM4',
                     'KMEAR','KRQB','KMOP','KMKG','KAMN','KBIV','KGRR',
                     'KY70','KLAN','KFPK','KTEW','KLWA','KPAWP','KAZO','KBTL','KRMY',
                     'KJXN','KBEH'):
    #if s in ['KAZO','KGRR','KMKG','KMOP','KMKG','KBIV']:
    station_id = key
    station_description = station_master[key]['name']
    column_list = []
    station_found = False
    utc_shift = station_master[key]['time_shift']
    p = re.compile(key)
    s = re.compile('SOL')
    dt = re.compile('DT')
    ymdh = re.compile('[0-9]+/[0-9]+/[0-9]+\s+[0-9]+')

    if download:
        raw_file_path = download_nbm_bulletin(url,fname,'hi')
        download = False
    else:
        raw_file_path = download_nbm_bulletin(url,fname,'just_path')
        
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

    # Now will start building a dictionary of products to plot
    # some products are only in a certain bulletin
    #   e.g., "nbm.X01" is hourly only, nbm.X06 is 6hourly and not in hourly product
    prods = {}
    
    # To plot time series lines  -- slice the dataframe with 'loc'
    # To plot a bar graph        -- convert the slice to a list.
    # Either can be done independently, but I usually do both to have the later 
    # option of plotting either way.
    
    # TMP and DPT will go on same panel, so using min(DPT) and max(TMP) to define bounds

    t_list = nbm.TMP.tolist()
    t_list = np.asarray(t_list, dtype=np.float32)
    t_shifted_list = t_list + 30
    dp_list = nbm.DPT.tolist()
    wdir_list = nbm.WDR.tolist()

    wdir = nbm.loc[:, ['WDR']]
    wspd = nbm.loc[:, ['WSP']]
    wspd_list_kt = nbm.WSP.tolist()
    wspd_list = np.multiply(wspd_list_kt,1.151)
    wspd_list = [round(x) for x in wspd_list]
    wspd_arr = np.asarray(wspd_list)
    wspd_list = wspd_arr.astype(int)
    wgst_list = nbm.GST.tolist()
    wgst_list = [round(x) for x in wgst_list]
    sky_list = nbm.SKY.tolist()

    wc_list = []
    time_to_fb_list = []
    for chill in range(0,len(wspd_list)):
        wc,time_to_fb = wind_chill(t_list[chill],wspd_list[chill])
        wc_list.append(wc)
        time_to_fb_list.append(time_to_fb)
    wc_list = np.asarray(wc_list, dtype=np.float32)
    wc_list = wc_list + 30

    twc_tick_list,twc_tick_labels = temperature_bounds(t_shifted_list,wc_list)
    
    # sometimes we have to convert units because they're in tenths or hundredths
    nbm.VIS = nbm.VIS.multiply(0.1)
    vis = nbm.loc[:, ['VIS']]
    vis.clip(upper=7.0,inplace=True)
    vis_list = nbm.VIS.tolist()
    

    p_ra_list = nbm.PRA.tolist()
    p_sn_list = nbm.PSN.tolist()
    p_zr_list = nbm.PZR.tolist()
    p_pl_list = nbm.PPL.tolist()


    # hourly snow amount, convert integers to tenths of an inch
    nbm.S01 = nbm.S01.multiply(0.1)
    s01_list = nbm.S01.tolist()
    sn_list = nbm.S01.tolist()
    sn_cat_list = categorize(sn_list,'sn')

    i01_list = nbm.I01.tolist()
    zr_list = i01_list

    # hourly precip amount, convert integers to hundredths of an inch
    nbm.Q01 = nbm.Q01.multiply(0.01)
    q01_list = nbm.Q01.tolist()
 
    #  P01 is one hour Probability for any type of precip, aka PPI
    p01_list = nbm.P01.tolist()

    #multiply P01 by ProbRain to get absolute rain probability
    #multiply P01 to ProbSnow to get absolute snow probability
    nbm['RA_01'] = nbm['P01'] * nbm['PRA']/100
    nbm['SN_01'] = nbm['P01'] * nbm['PSN']/100

    ra_01 = nbm.loc[:, ['RA_01']]
    sn_01 = nbm.loc[:, ['SN_01']]

    ra_01_list = np.multiply(p01_list,p_ra_list)/100
    sn_01_list = np.multiply(p01_list,p_sn_list)/100

    qpf_color = (0.1, 0.9, 0.1, 0.8)

    """
    Uncomment to make synthetic data
    
    t_list = np.arange(20,-5,-1)
    t_shifted_list = t_list + 20
    wspd_list_temp = np.arange(12,24.5,0.5)
    wspd_list = np.rint(wspd_list_temp)
    wdir_list = np.arange(140,43,-4)
    vis_test = np.arange(10,0.1,-0.4)
    vis_list = categorize(vis_test,'vis')
    sn_test = np.arange(0,2.5,0.1)
    sn_list = categorize(sn_test,'sn')
    zr_test = np.arange(1,26,1)
    zr_list = categorize(zr_test,'zr')

    wind_chill_list = []
    time_to_fb_list = []
    for chill in range(0,len(wspd_list)):
        wc,time_to_fb = wind_chill(t_list[chill],wspd_list[chill])
        wc = wc + 70
        wind_chill_list.append(wc)
        time_to_fb_list.append(time_to_fb)
    """
    ### -----------------------------  Begin Plotting ---------------------------

    # conditional probabilities for rain (PRA), snow (PSN), freezing rain (PZR), sleet (PPL)
    # define y axis range and yticks/ylabels for any element that's probabilistic 
    prob_yticks = [0, 20, 40, 60, 80, 100]
    prob_ytick_labels = ["0","20", "40","60","80","100"]
    p_min = -5
    p_max = 105


    prods['p_sn_bar'] = {'data': p_sn_list, 'color':(0.2, 0.2, 0.8, 0.6), 'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Prob Snow\n(%)'}
    prods['p_zr_bar'] = {'data': p_zr_list, 'color':(0.6, 0.4, 0.2, 0.6), 'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Prob ZR\n(%)'}
    prods['p_pl_bar'] = {'data': p_pl_list, 'color':(0.2, 0.8, 0.2, 0.6), 'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Prob PL\n(%)'}
    prods['ra_01_bar'] = {'data': ra_01_list, 'color':(0, 153/255, 0, 0.8), 'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Abs Prob\nRain (%)'}

    prods['p01_bar'] = {'data': p01_list, 'color':(0.5, 0.5, 0.5, 0.5), 'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Precip\nChances\n(%)'}


    prods['s01_bar'] = {'data': s01_list, 'color':(0.1, 0.1, 0.7, 0.7),
         'ymin':0.0,'ymax':1.01,'yticks':[0,0.25,0.5,0.75,1],
         'ytick_labels':['0','1/4','1/2','3/4','1'], 'title':'Snow\nAccum' }



    prods['wind'] = {'data': wspd, 'color':(0.5, 0.5, 0.5, 0.8),'title':'Wind\nSpeed\n& Gust'}


    prods['t_bar'] = {'data': t_shifted_list, 'color':(0.9, 0.1, 0.1, 0.7),
         'ymin':30,'ymax':80,'bottom':0,
         'major_yticks':[30,45,62,75],'major_yticks_labels':['0','15','32','45'],
         'minor_yticks':[60],'minor_yticks_labels':['30'],
         'title':'Temperature\nWind Chill\n(F)' }

    prods['wc_bar'] = {'data': wc_list, 'color':(0, 0, 255/255, 0.3),
         'ymin':20,'ymax':75,'bottom':0,
         'major_yticks':[30,45,62,75],'major_yticks_labels':['0','15','32','45'],
         'minor_yticks':[60],'minor_yticks_labels':['30'],
         'title':'Wind\nChill'}

    prods['time_fb_bar'] = {'data': time_to_fb_list, 'color':(0.9, 0.9, 0.2, 0.8),
         'ymin':1,'ymax':4.5,'yticks':[1,2,3,4],'ytick_labels':['5','10','30','60+'],
         'title':'Minutes to\nFrostbite'}

    prods['sky_bar'] = {'data': sky_list, 'color':(0.6, 0.6, 0.6, 0.6),
         'ymin':-5,'ymax':105,'yticks':[0,25,50,75,100],
         'ytick_labels':['0','25','50','75','100'], 'title':'Sky cover\n(%)'}

#    prods['vis_bar'] = {'data': vis_list, 'color':(0.7, 0.7, 0.3, 1),
#         'ymin':-0.5,'ymax':8,'yticks':[1, 3, 5, 7],
#         'ytick_labels':['1','3','5', '>6'], 'title':'Visibility\n(miles)'}

    prods['sn_01_bar'] = {'data': sn_01_list, 'color':(0.3, 0.3, 0.8, 0.8),
         'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels,
         'title':'Abs Prob\nSnow(%)'}

    prods['sn_test'] = {'data': sn_cat_list, 'color':(0,153/255,204/255, 0.7),
         'ymin':0,'ymax':7,'bottom':0,
         'major_yticks':[0,1,2,3,4,5,6,7],'major_yticks_labels':['0.0','0.1','0.2','0.5','0.8','1.0','1.5','2.0'],    
         'title':'Snow\nAccum\n(in)' }
    sn_dict = {'-0.1':0,'0.05':1,'0.15':2,'0.45':3,'0.75':4,'0.95':5,'1.45':6}        

    prods['zr_test'] = {'data': zr_list, 'color':(204/255,153/255,204/255, 0.7),
         'ymin':0,'ymax':5,'bottom':0,
         'major_yticks':[0,1,2,3,4,5],'major_yticks_labels':['0.01','0.03','0.05','0.10','0.25','0.5'],    
         'title':'Ice\nAccum\n(in)' }

    prods['vis_test'] = {'data': vis_list, 'color':(150/255,150/255,245/255, 0.6),
         'ymin':0,'ymax':6,'bottom':0,
         'major_yticks':[0,1,2,3,4,5,6],'major_yticks_labels':['0.00','0.25','0.50','1.00','2.00','3.00',' > 6'],    
         'title':'Visibility\n(miles)' }

    prods['p_ra_bar'] = {'data': p_ra_list, 'color':(0.2, 0.8, 0.2, 0.6),
         'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Prob Rain\n(%)'}

    prods['q01_bar'] = {'data': q01_list, 'color':qpf_color,
         'ymin':0.0,'ymax':0.50,
         'yticks':[0.0,0.1,0.2,0.3], 'ytick_labels':['0','0.1','0.2','0.3'],
         'title':'Precip\nAmount\n(inches)'}    

    hours = mdates.HourLocator()
    myFmt = DateFormatter("%d%h")
    myFmt = DateFormatter("%d%b\n%HZ")
    myFmt = DateFormatter("%I\n%p")
    fig, axes = plt.subplots(len(products),1,figsize=(15,10),sharex=True,subplot_kw={'xlim': (start_time,end_time)})

    #fig, axes = plt.subplots(len(products),1,figsize=(16,8),sharex=True,subplot_kw={'xlim': (start_time,end_time)})
    plt.subplots_adjust(bottom=0.1, left=0.17, top=0.9)
    
    plt.suptitle('Hourly Forecast -- ' + model_run_local.strftime('%B %d, %Y') + ' ... Updated ' + model_run_local.strftime('%I %p EST') + '\n' + station_description  )
    
    first_gray = True
    for y,a in zip(products,axes.ravel()):

        #plt.setp( ax.xaxis.get_majorticklabels(), rotation=-45) 

        # Create offset transform by 5 points in x direction
        dx = 7/72.; dy = 0/72. 
        offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

        # apply offset transform to all x ticklabels.

        for label in a.xaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + offset)


        a.xaxis.set_major_locator(hours)
        a.xaxis.set_major_formatter(myFmt)


        plt.setp( a.xaxis.get_majorticklabels(), ha="left",rotation=0 )
        a.yaxis.set_label_coords(-0.08,0.25)
          
        if y == 't':
            a.grid()
            a.get_xaxis().set_visible(False)
            gs = GridShader(a, facecolor="lightgrey", first=first_gray, alpha=0.5)            
            a.set_ylim(prods[y]['ymin'],prods[y]['ymax'])
            a.set(yticks = prods[y]['yticks'], yticklabels = prods[y]['ytick_labels'])
            a.set_ylabel(prods[y]['title'], rotation=0)
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


        if y == 'wind':
            plt.rc('font', size=12) 
            a.set_ylim(0,1)

            dx = 7/72.; dy = 0/72. 
            offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

            # apply offset transform to all x ticklabels.

            a.set_ylabel(prods[y]['title'], rotation=0)
            a.get_xaxis().set_visible(False)
            a.get_yaxis().set_visible(False)
            #gs = GridShader(a, facecolor="lightgrey", first=first_gray, alpha=0.5)
            a.set_xticks(data_list)    
            for s,d,g,p in zip(wspd_list,wdir_list,wgst_list,data_list):
                u,v,dx,dy = u_v_components(d,s)
                a.barbs(p, 0.7, u, v, length=7, color=[0,0,1,0.9], pivot='middle')
                a.text(p, 0.28, f'{s}',horizontalalignment='center',color=[0,0,1,0.9])
                a.text(p, 0.08, f'{g}',horizontalalignment='center',color=[0.6,0,1,0.6])



        # these are dataframe slices that use matplotlib plot to create time series
        if y in ['p01','p06','p_zr','p_pl','sky','vis']:
            gs = GridShader(a, facecolor="lightgrey", first=first_gray, alpha=0.5) 
            a.set_ylim(prods[y]['ymin'],prods[y]['ymax'])
            a.set(yticks = prods[y]['yticks'])
            a.plot(prods[y]['data'],color=prods[y]['color'])
            a.set_ylabel(prods[y]['title'], rotation=0)
            a.set(yticks = prods[y]['yticks'], yticklabels = prods[y]['ytick_labels'])

        # specialized treatment for ranges and gridlines
        if y in ['sn_test','zr_test','vis_test']:
            gs = GridShader(a, facecolor="lightgrey", first=first_gray, alpha=0.1) 
            a.set_yticks(prods[y]['major_yticks'], minor=False)
            a.set_yticklabels(prods[y]['major_yticks_labels'],minor=False)
            a.grid(which='major', axis='y')
            a.set_xticks(data_list)
            a.set_ylim(prods[y]['ymin'],prods[y]['ymax'])
            a.bar(data_list,prods[y]['data'],width=1/25, align="edge",bottom=prods[y]['bottom'],color=prods[y]['color'])

            a.set_ylabel(prods[y]['title'], rotation=0)
            a.get_xaxis().set_visible(True)

        if y in ['t_bar','wc_bar']:
            gs = GridShader(a, facecolor="lightgrey", first=first_gray, alpha=0.2) 
            a.set_yticks(twc_tick_list, minor=False)
            a.set_yticklabels(twc_tick_labels,minor=False)
            a.grid(which='major', axis='y')
            a.set_xticks(data_list)
            a.set_ylim(twc_tick_list[0],twc_tick_list[-1])
            a.bar(data_list,prods['t_bar']['data'],width=1/25, align="edge",bottom=prods[y]['bottom'],color=prods['t_bar']['color'])
            a.bar(data_list,prods['wc_bar']['data'],width=1/25, align="edge",bottom=prods[y]['bottom'],color=prods['wc_bar']['color'])
            a.set_ylabel(prods[y]['title'], rotation=0)
            a.get_xaxis().set_visible(True)
    
        # these are lists that use matplotlib bar to create bar graphs
        if y in ['p01_bar','q01_bar','s01_bar','sky_bar','p_zr_bar','p_sn_bar','p_pl_bar','p_ra_bar','sn_01_bar','ra_01_bar','vis_bar']:
            gs = GridShader(a, facecolor="lightgrey", first=first_gray, alpha=0.3) 
            a.set_xticks(data_list)
            a.set_ylim(prods[y]['ymin'],prods[y]['ymax'])
            a.bar(data_list,prods[y]['data'],width=1/25, align="edge",color=prods[y]['color'])

            a.set_ylabel(prods[y]['title'], rotation=0)
            a.get_xaxis().set_visible(True)
            a.set(yticks = prods[y]['yticks'], yticklabels = prods[y]['ytick_labels'])
            
        if y in ['time_fb_bar']:
            gs = GridShader(a, facecolor="lightgrey", first=first_gray, alpha=0.0) 
            a.grid(which='major', axis='y')
            a.set_xticks(data_list)
            a.set_ylim(prods[y]['ymin'],prods[y]['ymax'])
            a.bar(data_list,prods[y]['data'],width=1/25, align="edge",color=prods[y]['color'])

            a.set_ylabel(prods[y]['title'], rotation=0)
            a.get_xaxis().set_visible(True)
            a.set(yticks = prods[y]['yticks'], yticklabels = prods[y]['ytick_labels'])
    
    image_file = key + '_NBM_' + bulletin_type + '.png'
    image_dst_path = os.path.join(image_dir,image_file)
    #plt.show()
    plt.savefig(image_dst_path,format='png')
    plt.close()



