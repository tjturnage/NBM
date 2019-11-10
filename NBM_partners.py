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

                 'nbhtx' -- hourly guidance ( hourly)

                 This is required to define model forecast hour start and 
                 end times as a well as forecast hour interval.
                  
      Returns
      -------
           pandas date/time range to be used as index as well as start/end times
    """

    fcst_hour_zero_utc = run_dt + timedelta(hours=0)
    fcst_hour_zero_local = fcst_hour_zero_utc - timedelta(hours=4)
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
    #print(round(wc))
    # time to frostbite 4=60, 3=30, 2=10, 1=5
    print(wc)
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

download = False
bulletin_type = 'nbhtx'

now = datetime.utcnow()
now2 = now - timedelta(hours=3)
ymd = now2.strftime('%Y%m%d')
hour = now2.strftime('%H')
url = 'https://para.nomads.ncep.noaa.gov/pub/data/nccf/com/blend/para/blend.20191107/15/text/blend_nbhtx.t15z'
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
lats = station_info[1::3]
lons = station_info[2::3]
products = ['t_bar','wc_bar','time_fb_bar','wind','vis_bar','s01_bar']
fname = 'nbm_raw_hourly.txt'


#for s in ['KAMN','BDWM4','KBELD','KRQB','KCAD']:
station_location = {}
for sta in range(0,len(stations)):
    station_location.update({stations[sta]:{'lat':lats[sta],'lon':lons[sta] }})

station_data = {}

for st in stations[14:15]:
    #if s in ['KAZO','KGRR','KMKG','KMOP','KMKG','KBIV']:
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
                idx,model_run_local = dtList_nbm(run_dt,bulletin_type) ######################################################3333333
                print(idx)
                start_time = idx[1]
                print(start_time)
                end_time = idx[-1]
                print(end_time)
                data_list = idx[1:-1]
                print(data_list)
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
    dp_list = nbm.DPT.tolist()
    wdir_list = nbm.WDR.tolist()

    wdir = nbm.loc[:, ['WDR']]
    wspd = nbm.loc[:, ['WSP']]
    wspd_list = nbm.WSP.tolist()
    wspd_list = [round(x) for x in wspd_list]
    wgst_list = nbm.GST.tolist()
    wgst_list = [round(x) for x in wgst_list]
    sky_list = nbm.SKY.tolist()

    wind_chill_list = []
    time_to_fb_list = []
    for chill in range(0,len(wspd_list)):
        wc,time_to_fb = wind_chill(t_list[chill],wspd_list[chill])
        wind_chill_list.append(wc)
        time_to_fb_list.append(time_to_fb)
    
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

    t_list = np.arange(20,-5,-1)
    wspd_list = np.arange(5,30,1)
    vis_list = np.arange(10,0.1,-0.4)

    wind_chill_list = []
    time_to_fb_list = []
    for chill in range(0,len(wspd_list)):
        wc,time_to_fb = wind_chill(t_list[chill],wspd_list[chill])
        wind_chill_list.append(wc)
        time_to_fb_list.append(time_to_fb)
    
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
    prods['ra_01_bar'] = {'data': ra_01_list, 'color':(0.4, 0.7, 0.4, 0.8), 'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Abs Prob\nRain (%)'}
    prods['sn_01_bar'] = {'data': sn_01_list, 'color':(0.3, 0.3, 0.8, 0.8), 'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Abs Prob\nSnow(%)'}
    prods['p01_bar'] = {'data': p01_list, 'color':(0.5, 0.5, 0.5, 0.5), 'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Precip\nChances\n(%)'}
    prods['s01_bar'] = {'data': s01_list, 'color':(0.1, 0.1, 0.7, 0.7), 'ymin':0.0,'ymax':1.01,'yticks':[0,0.25,0.5,0.75,1], 'ytick_labels':['0','1/4','1/2','3/4','1'], 'title':'Snow Accum' }



    prods['wind'] = {'data': wspd, 'color':(0.5, 0.5, 0.5, 0.8),'title':'Wind\nSpeed\n& Gust'}


    prods['t_bar'] = {'data': t_list, 'color':(0.8, 0.0, 0.0, 0.8),
         'ymin':-20,'ymax':40,'bottom':-20,
         'major_yticks':[-15,0,15,32],'major_yticks_labels':['-15','0','15','32'],
         'minor_yticks':[-20,-10,10],'minor_yticks_labels':['-20','-10','10'],
         'title':'Temp (F)' }

    prods['wc_bar'] = {'data': wind_chill_list, 'color':(0.2, 0.8, 0.2, 0.6),
         'ymin':-35,'ymax':25,'bottom':-35,
         'major_yticks':[-30,-15,0,15],'major_yticks_labels':['-30','-15','0','15'],
         'minor_yticks':[-20,20],'minor_yticks_labels':['-20','20'],
         'title':'Wind\nChill'}

    prods['time_fb_bar'] = {'data': time_to_fb_list, 'color':(0.8, 0.8, 0.2, 0.6),
         'ymin':0,'ymax':4,'yticks':[1,2,3,4],'ytick_labels':['5','10','30','60'],
         'title':'Minutes to\nFrostbite'}

    prods['sky_bar'] = {'data': sky_list, 'color':(0.6, 0.6, 0.6, 0.6),
         'ymin':-5,'ymax':105,'yticks':[0,25,50,75,100],
         'ytick_labels':['0','25','50','75','100'], 'title':'Sky cover\n(%)'}

    prods['vis_bar'] = {'data': vis_list, 'color':(0.7, 0.7, 0.3, 1),
         'ymin':-0.5,'ymax':8,'yticks':[1, 3, 5, 7],
         'ytick_labels':['1','3','5', '>6'], 'title':'Visibility\n(miles)'}

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
    fig, axes = plt.subplots(len(products),1,figsize=(16,10),sharex=True,constrained_layout=True,subplot_kw={'xlim': (start_time,end_time)})

    #fig, axes = plt.subplots(len(products),1,figsize=(16,8),sharex=True,subplot_kw={'xlim': (start_time,end_time)})
    plt.subplots_adjust(bottom=0.1, left=0.25, top=0.9)
    
    plt.suptitle('NBM hourly Guidance - ' + station_name + '\n' + model_run_local.strftime('%B %d, %Y -- %I %p EST'))
    
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
        if y in ['t_bar','wc_bar']:
            gs = GridShader(a, facecolor="lightgrey", first=first_gray, alpha=0.3) 
            a.set_yticks(prods[y]['major_yticks'], minor=False)
            a.grid(which='major', axis='y')
            a.set_xticks(data_list)
            a.set_ylim(prods[y]['ymin'],prods[y]['ymax'])
            a.bar(data_list,prods[y]['data'],width=1/25, bottom=prods[y]['bottom'],  align="edge",color=prods[y]['color'])

            a.set_ylabel(prods[y]['title'], rotation=0)
            a.get_xaxis().set_visible(True)

    
        # these are lists that use matplotlib bar to create bar graphs
        if y in ['time_fb_bar','p01_bar','q01_bar','s01_bar','sky_bar','p_zr_bar','p_sn_bar','p_pl_bar','p_ra_bar','sn_01_bar','ra_01_bar','vis_bar']:
            gs = GridShader(a, facecolor="lightgrey", first=first_gray, alpha=0.3) 
            a.set_xticks(data_list)
            a.set_ylim(prods[y]['ymin'],prods[y]['ymax'])
            a.bar(data_list,prods[y]['data'],width=1/25, align="edge",color=prods[y]['color'])

            a.set_ylabel(prods[y]['title'], rotation=0)
            a.get_xaxis().set_visible(True)
            a.set(yticks = prods[y]['yticks'], yticklabels = prods[y]['ytick_labels'])
    
    image_file = station_name + '_NBM_' + bulletin_type + '.png'
    image_dst_path = os.path.join(image_dir,image_file)
    plt.show()
    plt.savefig(image_dst_path,format='png')
    plt.close()



