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
import re
import os
import sys
import math
import pandas as pd
import matplotlib.pyplot as plt
#from matplotlib.ticker import FormatStrFormatter
from matplotlib.dates import DateFormatter
from bs4 import BeautifulSoup
import requests

#https://www.weather.gov/mdl/nbm_text?ele=NBH&cyc=00&sta=KGRR&download=yes

def dtList(dtLine):
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
       

def download_nbm_bulletin():
    url = "https://sats.nws.noaa.gov/~downloads/nbm/bulk-textv32/current/"
    searchStr = 'nbh'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')

    for link in soup.find_all('a'):
        fName = str(link.get('href'))
        if searchStr in fName:
            src = os.path.join(url,fName)
            break

    r = requests.get(src)
    print('downloading ... ' + str(src))
    dst = os.path.join(base_dir,'nbm_raw.txt')
    open(dst, 'wb').write(r.content)
    return

def u_v_components(wdir, wspd):
    u = (math.sin(math.radians(wdir)) * wspd) * -1.0
    v = (math.cos(math.radians(wdir)) * wspd) * -1.0
    #loc = str(int(x)) + ',' + str(int(y)) + ',1,'
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
    src = os.path.join(base_dir,'nbm_raw.txt')
    dst_file = os.path.join(base_dir,'nbm_out.txt')
    sys.path.append('C:/data/scripts/resources')

# ensure image directory is created
try:
    os.makedirs(image_dir)
except:
    pass


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
BIGGEST_SIZE = 14

plt.rc('font', size=BIGGEST_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=20)  # fontsize of the figure title
column_list = []
grr = False

p = re.compile('KGRR')
s = re.compile('SOL')
#download_nbm_bulletin()
#src = 'C:/data/nbm_raw.txt'
dst = open(dst_file, 'w')
with open(src) as fp:  
    for line in fp:
        #print(line)
        m = p.search(line)
        sol = s.search(line)
        #print(m)
        if m is not None:
            grr = True
            pd_series = dtList(line)
            start_time,end_time = bounds(line)
            #print(pd_series)
        elif grr and sol is None:
            start = str(line[1:4])
            column_list.append(start)
            #print(start)
            dst.write(line)             
        elif sol is not None and grr:
            break

elements = column_list[1:]
nbm_old = pd.read_fwf('C:/data/nbmout.txt', widths=(5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3))

# flip table so times are rows to align with pandas
nbm = nbm_old.transpose()

# after the flip column names are useless. Use the created column_list before the flip
# to make a dictionary that replaces bad column names with the original, pre-flip column names
old_column_names = nbm.columns.tolist()
col_rename_dict = {i:j for i,j in zip(old_column_names,elements)}
nbm.rename(columns=col_rename_dict, inplace=True)

# Now that columns are created, get rid of redundant UTC line
nbm.drop(['UTC'], inplace=True)

# replace the index with the pandas time series
nbm.set_index(pd_series, inplace=True)
#print(nbm)

# qpf is in hundredths and snow is in tenths of an inch, so have to convert
nbm.Q01 = nbm.Q01.multiply(0.01)
nbm.S01 = nbm.Q01.multiply(0.1)

# slice out columns for the elements you want to plot
t = nbm.loc[:, ['TMP']]
t_list = nbm.TMP.tolist()
t_max = round_values(max(t_list),1,'up')
t_min = round_values(min(t_list),1,'down')

dp = nbm.loc[:, ['DPT']]
dp_list = nbm.DPT.tolist()
dp_max = round_values(max(dp_list),1,'up')
dp_min = round_values(min(dp_list),1,'down')

wdir = nbm.loc[:, ['WDR']]
wspd = nbm.loc[:, ['WSP']]
wgst = nbm.loc[:, ['GST']]

wdir_list = nbm.WDR.tolist()
wspd_list = nbm.WSP.tolist()
wgst_list = nbm.GST.tolist()
wgst_max = round_values(max(wgst_list),1,'up')
wgst_list = [round(x) for x in wgst_list]

sn01 = nbm.S01.tolist()
qp_list = nbm.Q01.tolist()
qp_max = max(qp_list)


ppi = nbm.P01.tolist()
p_ra = nbm.loc[:, ['PRA']]
p_zr = nbm.PZR.tolist()
p_pl = nbm.PPL.tolist()
p_sn = nbm.loc[:, ['PSN']]

myFmt = DateFormatter("%d%h")

prods = {}
prods['snow1'] = {'data': sn01, 'color':(0.2, 0.4, 0.6, 0.6), 'ymin':0.0,'ymax':2.0 }
prods['pop01'] = {'data': ppi, 'color':(0.2, 0.8, 0.2, 0.6), 'ymin':0,'ymax':110}
prods['qpf01'] = {'data': qp_list, 'color':(0.2, 0.2, 0.8, 0.6), 'ymin':0.0,'ymax':0.50}
prods['t'] = {'data': t, 'color':(0.8, 0.0, 0.0, 0.8), 'ymin':t_min,'ymax':t_max }
prods['dp'] = {'data': dp, 'color':(0.0, 0.9, 0.1, 0.6), 'ymin':dp_min,'ymax':dp_max  }
prods['wspd'] = {'data': wspd, 'color':(0.5, 0.5, 0.5, 0.8), 'ymin':0,'ymax':wgst_max}
prods['wgst'] = {'data': wgst, 'color':(0.7, 0.7, 0.7, 0.7), 'ymin':0,'ymax':wgst_max}
prods['wdir'] = {'data': wdir, 'color':(0.7, 0.7, 0.7, 0.7), 'ymin':0,'ymax':wgst_max}

products = ['pop01','qpf01','t','wdir']
myFmt = DateFormatter("%d%b\n%HZ")
fig, axes = plt.subplots(len(products),1,figsize=(13,8),sharex=True,subplot_kw={'xlim': (start_time,end_time)})
plt.suptitle('NBM hourly Guidance - KGRR')
for y,a in zip(products,axes.ravel()):
    a.xaxis.set_major_formatter(myFmt)
    if y == 't':
        a.set_ylim(prods[y]['ymin'],prods[y]['ymax'])
        a.plot(prods[y]['data'],color=prods[y]['color'])
        a.plot(prods['dp']['data'],color=prods['dp']['color'])
    if y == 'wdir':
        a.set_ylim(0,1)
        a.get_yaxis().set_visible(False)
        for s,d,g,p in zip(wspd_list,wdir_list,wgst_list,pd_series):
            u,v = u_v_components(d,s)
            #print(u,v)
            #x_position = (w/len(wspd_list)) * 10 
            a.barbs(p, 0.6, u, v, length=8, color=[0,0,1,0.9], pivot='middle')
            a.text(p, 0.2, f'{g}',horizontalalignment='center',color=[0,0,1,0.9])
            #a.plot(prods['wspd']['data'],color=prods['wspd']['color'])            
    if y == 'pop01' or y == 'qpf01':
        a.set_ylim(prods[y]['ymin'],prods[y]['ymax'])
        a.plot(prods[y]['data'],color=prods[y]['color'])
        a.bar(pd_series,prods[y]['data'],width=1/26, color=prods[y]['color'])

image_dst_path = os.path.join(image_dir,'GRR_NBM.png')
plt.savefig(image_dst_path,format='png')
