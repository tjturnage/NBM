# -*- coding: utf-8 -*-
"""
Grabs NBM V3.2 houlry bulletins from:
https://sats.nws.noaa.gov/~downloads/nbm/bulk-textv32/current/

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

def dtList(dtLine):
    # returns both a pandas time series and a string for plotting title
    mdyList = dtLine.split()
    hrStr = mdyList[-2][0:2]
    hr = int(hrStr)
    mdy_group = mdyList[-3]
    mm_dd_yr = mdy_group.split('/')
    mm = int(mm_dd_yr[0])
    dd = int(mm_dd_yr[1])
    yyyy = int(mm_dd_yr[-1])
    pTime = pd.Timestamp(yyyy,mm,dd,hr)
    idx = pd.date_range(pTime, periods=25, freq='H')
    return idx[1:]

def bounds(dtLine):
    mdyList = dtLine.split()
    hrStr = mdyList[-2][0:2]
    hr = int(hrStr)
    mdy_group = mdyList[-3]
    mm_dd_yr = mdy_group.split('/')
    mm = int(mm_dd_yr[0])
    dd = int(mm_dd_yr[1])
    yyyy = int(mm_dd_yr[-1])
    pTime = pd.Timestamp(yyyy,mm,dd,hr)
    idx = pd.date_range(pTime, periods=26, freq='H')
    return idx[0],idx[-1]

def round_values(x,places,direction):
    amount = 10**places
    if direction == 'up':
        return int(math.ceil(x / float(amount))) * int(amount)
    if direction == 'down':
        return int(math.floor(x / float(amount))) * int(amount)
       

def download_nbm_bulletin(bulletin_type):
    url = "https://sats.nws.noaa.gov/~downloads/nbm/bulk-textv32/current/"
    if bulletin_type == 'hourly':
        searchStr = 'nbh'
    elif bulletin_type == 'short':
        searchStr = 'nbs'        
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')

    for link in soup.find_all('a'):
        fName = str(link.get('href'))
        if searchStr in fName:
            src = os.path.join(url,fName)
            break

    r = requests.get(src)
    print('downloading ... ' + str(src))
    fname = 'nbm_raw.txt'
    dst = os.path.join(base_dir,fname)
    open(dst, 'wb').write(r.content)
    return

def u_v_components(wdir, wspd):
    # since the convention is "direction from"
    # we have to multiply by -1
    u = (math.sin(math.radians(wdir)) * wspd) * -1.0
    v = (math.cos(math.radians(wdir)) * wspd) * -1.0
    return u,v

try:
    os.listdir('/usr')
    windows = False
    base_dir = '/data/scripts'
    sys.path.append('/data/scripts/resources')
    image_dir = os.path.join('/var/www/html/radar','images')
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
from my_functions import GridShader, plot_settings
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from bs4 import BeautifulSoup
import requests

column_list = []
station_found = False
station_name = 'KDVN'
p = re.compile(station_name)
s = re.compile('SOL')

download_nbm_bulletin('hourly')


dst = open(trimmed_nbm_file, 'w')
with open(raw_nbm_file) as fp:  
    for line in fp:
        #print(line)
        m = p.search(line)
        sol = s.search(line)
    #print(m)
        if m is not None:
            station_found = True
            pd_series = dtList(line)
            start_time,end_time = bounds(line)
            #print(pd_series)
        elif station_found and sol is None:
            start = str(line[1:4])
            column_list.append(start)
            dst.write(line)             
        elif sol is not None and station_found:
            dst.close()
            break
            

elements = column_list[1:]
nbm_old = None
nbm_old = pd.read_fwf(trimmed_nbm_file, widths=(5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3))

# flip table so times are rows to align with pandas
nbm = nbm_old.transpose()

# after the flip column names are useless. Use the created column_list before the flip
# to make a dictionary that replaces bad column names with the original, pre-flip column names
old_column_names = nbm.columns.tolist()
col_rename_dict = {i:j for i,j in zip(old_column_names,elements)}
nbm.rename(columns=col_rename_dict, inplace=True)

# Now that columns are created, there's now a redundant UTC line. Remove it.
# Then replace the index with the pandas time series
nbm.drop(['UTC'], inplace=True)
nbm.set_index(pd_series, inplace=True)

# slice out columns for the elements you want to plot as lines
# if you want a bar plat, you just need to send a slice directly "tolist"
# We will plot TMP and DPT on same pane, so use range min(DPT) to max(TMP)
t = nbm.loc[:, ['TMP']]
t_list = nbm.TMP.tolist()
t_max = round_values(max(t_list),1,'up')

dp = nbm.loc[:, ['DPT']]
dp_list = nbm.DPT.tolist()
t_min = round_values(min(dp_list),1,'down')

wdir = nbm.loc[:, ['WDR']]
wdir_list = nbm.WDR.tolist()

wspd = nbm.loc[:, ['WSP']]
wspd_list = nbm.WSP.tolist()

wgst = nbm.loc[:, ['GST']]
wgst_list = nbm.GST.tolist()
wgst_list = [round(x) for x in wgst_list]

sky = nbm.loc[:, ['SKY']]
sky_list = nbm.SKY.tolist()

nbm.VIS = nbm.VIS.multiply(0.1)
vis = nbm.loc[:, ['VIS']]
vis.clip(upper=7.0,inplace=True)
vis_list = nbm.VIS.tolist()
#vis_clipped = np.clip(vis_list,0,7.0)

#bulletin lists hourly snow in tenths of an inch
nbm.S01 = nbm.Q01.multiply(0.1)
sn01_list = nbm.S01.tolist()

#bulletin lists hourly QPF in hundredths of an inch
nbm.Q01 = nbm.Q01.multiply(0.01)
qp_list = nbm.Q01.tolist()

qp_max = max(qp_list)
qp_yticks = [0.0,0.1,0.2,0.3]
qp_ylabels = []
for z in range(0,len(qp_yticks)):
    qp_ylabels.append(str(qp_yticks[z]))


ppi = nbm.P01.tolist()
p_ra = nbm.PRA.tolist()
p_zr = nbm.PZR.tolist()
p_pl = nbm.PPL.tolist()
p_sn = nbm.PSN.tolist()


prods = {}
prods['snow'] = {'data': sn01_list, 'color':(0.1, 0.1, 0.7, 0.7), 'ymin':0.0,'ymax':1.5, 'ylabel':'Snow\n(inches)' }
prods['pop01'] = {'data': ppi, 'color':(0.2, 0.8, 0.2, 0.6), 'ymin':-5,'ymax':105, 'ylabel':'Precip\nChances\n(%)'}
prods['qpf01'] = {'data': qp_list, 'color':(0.2, 0.8, 0.2, 0.7), 'ymin':0.0,'ymax':0.50,'ylabel':'Precip\nAmount\n(inches)'}
prods['t'] = {'data': t, 'color':(0.8, 0.0, 0.0, 0.8), 'ymin':t_min,'ymax':t_max, 'ylabel':'Temerature\nDewpoint' }
prods['dp'] = {'data': dp, 'color':(0.0, 0.9, 0.1, 0.6), 'ymin':t_min,'ymax':t_max, 'ylabel':'Temerature\nDewpoint'  }
prods['wspd'] = {'data': wspd, 'color':(0.5, 0.5, 0.5, 0.8),'ylabel':'Wind\nSpeed'}
prods['wgst'] = {'data': wgst, 'color':(0.7, 0.7, 0.7, 0.7),'ylabel':'Wind\nSpeed'}
prods['wdir'] = {'data': wdir, 'color':(0.7, 0.7, 0.7, 0.7),'ylabel':'Wind\nSpeed'}
prods['sky'] = {'data': sky_list, 'color':(0.4, 0.4, 0.4, 0.8), 'ymin':-5,'ymax':105, 'yticks':[0,25,50,75,100], 'ylabel':'Sky cover\n(%)'}
prods['vis'] = {'data': vis, 'color':(0.7, 0.7, 0.3, 1), 'ymin':-0.5,'ymax':8, 'ylabel':'Visibility\n(miles)'}

products = ['t','wdir','vis','sky','pop01','qpf01','snow']

hours = mdates.HourLocator()
myFmt = DateFormatter("%d%h")
myFmt = DateFormatter("%d%b\n%HZ")
myFmt = DateFormatter("%d\n%HZ")
fig, axes = plt.subplots(len(products),1,figsize=(16,10),sharex=True,subplot_kw={'xlim': (start_time,end_time)})
#fig, axes = plt.subplots(len(products),1,figsize=(16,8),sharex=True,subplot_kw={'xlim': (start_time,end_time)})
plt.subplots_adjust(bottom=0.1, left=0.25, top=0.9)
plt.suptitle('NBM hourly Guidance - ' + station_name + '\n' + start_time.strftime('%B %d, %Y -- %HZ'))
#plt.suptitle('NBM hourly Guidance - KGRR')
for y,a in zip(products,axes.ravel()):
    a.xaxis.set_major_locator(hours)
    a.xaxis.set_major_formatter(myFmt)
    a.yaxis.set_label_coords(-0.08,0.30)

    if y == 't':
        a.grid()
        a.get_xaxis().set_visible(False)
        gs = GridShader(a, facecolor="lightgrey", first=False, alpha=0.5)
        a.set_ylim(prods[y]['ymin'],prods[y]['ymax'])
        a.set_ylabel(prods['t']['ylabel'], rotation=0)
        a.plot(prods['dp']['data'],linewidth=3,color=prods['dp']['color'])
        a.plot(prods['t']['data'],linewidth=3,color=prods['t']['color'])

    if y == 'vis':
        a.grid()
        a.get_xaxis().set_visible(True)
        #a.set_ylabel(prods[y]['ylabel'], rotation=0)
        a.set_ylim(prods[y]['ymin'],prods[y]['ymax'])
        a.set(yticks = [1, 3, 5, 7], yticklabels = ["1","3","5", ">6"])

        gs = GridShader(a, facecolor="lightgrey", first=False, alpha=0.6)

        a.plot(prods[y]['data'],linewidth=4,color=prods[y]['color'])

    if y == 'wdir':
        a.set_ylim(0,1)
        a.set_ylabel(prods[y]['ylabel'], rotation=0)
        a.get_xaxis().set_visible(False)
        gs = GridShader(a, facecolor="lightgrey", first=False, alpha=0.25)

        for s,d,g,p in zip(wspd_list,wdir_list,wgst_list,pd_series):
            u,v = u_v_components(d,s)
            a.barbs(p, 0.6, u, v, length=8, color=[0,0,1,0.9], pivot='middle')
            a.text(p, 0.2, f'{g}',horizontalalignment='center',color=[0,0,1,0.9])
         
    if y == 'pop01':
        gs = GridShader(a, facecolor="lightgrey", first=False, alpha=0.3)
        a.set_ylim(prods[y]['ymin'],prods[y]['ymax'])
        a.plot(prods[y]['data'],color=prods[y]['color'])
        a.bar(pd_series,prods[y]['data'],width=1/25, align="edge",color=prods[y]['color'])
        a.set_ylabel(prods[y]['ylabel'], rotation=0)
        a.set(yticks = [0, 20, 40, 60, 80, 100], yticklabels = ["0","20", "40","60","80","100"])

    if y == 'snow':
        gs = GridShader(a, facecolor="lightgrey", first=False, alpha=0.3)
        a.set_ylim(prods[y]['ymin'],prods[y]['ymax'])
        a.plot(prods[y]['data'],color=prods[y]['color'])
        a.bar(pd_series,prods[y]['data'],width=1/25, align="edge",color=prods[y]['color'])
        a.set_ylabel(prods[y]['ylabel'], rotation=0)
        a.set(yticks = [0, 0.25, 0.5, 0.75, 1], yticklabels = ["0","0.25", "0.5","0.75","1"])


    if y == 'sky':
        gs = GridShader(a, facecolor="lightgrey", first=False, alpha=0.3)
        a.set_ylim(prods[y]['ymin'],prods[y]['ymax'])
        a.plot(prods[y]['data'],color=prods[y]['color'])
        a.bar(pd_series,prods[y]['data'],width=1/26, align="edge",color=prods[y]['color'])
        a.set_ylabel(prods[y]['ylabel'], rotation=0)
        a.set(yticks = [0, 20, 40, 60, 80, 100], yticklabels = ["0","20", "40","60","80","100"])

    if y == 'qpf01':
        gs = GridShader(a, facecolor="lightgrey", first=False, alpha=0.3)
        a.set_ylim(prods[y]['ymin'],prods[y]['ymax'])
        a.plot(prods[y]['data'],color=prods[y]['color'])
        a.bar(pd_series,prods[y]['data'],width=1/25, align="edge",color=prods[y]['color'])
        a.set_ylabel(prods[y]['ylabel'], rotation=0)
        a.set(yticks = qp_yticks, yticklabels = qp_ylabels)

image_dst_path = os.path.join(image_dir,'GRR_NBM.png')
plt.savefig(image_dst_path,format='png')
