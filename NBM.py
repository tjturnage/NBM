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
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
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

def round_values(x,places,direction):
    amount = 10**places
    if direction == 'up':
        return int(math.ceil(x / float(amount))) * int(amount)
    if direction == 'down':
        return int(math.floor(x / float(amount))) * int(amount)
       

def download_nbm_bulletin():
    base_dir = 'C:/data'
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


column_list = []
grr = False

p = re.compile('KGRR')
s = re.compile('SOL')
download_nbm_bulletin()
src = 'C:/data/nbm_raw.txt'
dst = open('C:/data/nbmout.txt', 'w')
with open(src) as fp:  
    for line in fp:
        #print(line)
        m = p.search(line)
        sol = s.search(line)
        #print(m)
        if m is not None:
            grr = True
            pd_series = dtList(line)
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

wspd = nbm.loc[:, ['WSP']]
wgst = nbm.loc[:, ['GST']]
wg_list = nbm.GST.tolist()
wg_max = round_values(max(wg_list),1,'up')
sn01 = nbm.S01.tolist()
qp_list = nbm.Q01.tolist()
qp_max = max(qp_list)


ppi = nbm.P01.tolist()
#ppi = nbm.loc[:, ['P01']]
p_ra = nbm.loc[:, ['PRA']]
p_zr = nbm.PZR.tolist()
p_pl = nbm.PPL.tolist()
p_sn = nbm.loc[:, ['PSN']]


start_time = pd_series[0]
end_time = pd_series[-1]
myFmt = DateFormatter("%d%h")

prods = {}
prods['snow1'] = {'data': sn01, 'color':(0.2, 0.4, 0.6, 0.6), 'ymin':0.0,'ymax':2.0 }
prods['pop01'] = {'data': ppi, 'color':(0.2, 0.8, 0.2, 0.6), 'ymin':0,'ymax':110}
prods['qpf01'] = {'data': qp_list, 'color':(0.2, 0.2, 0.8, 0.6), 'ymin':0.0,'ymax':0.50}
prods['t'] = {'data': t, 'color':(0.8, 0.0, 0.0, 0.8), 'ymin':t_min,'ymax':t_max }
prods['dp'] = {'data': dp, 'color':(0.0, 0.9, 0.1, 0.6), 'ymin':dp_min,'ymax':dp_max  }
prods['wspd'] = {'data': wspd, 'color':(0.7, 0.7, 0.7, 0.8), 'ymin':0,'ymax':wg_max}
prods['wgst'] = {'data': wgst, 'color':(0.7, 0.7, 0.7, 0.6), 'ymin':0,'ymax':wg_max}


products = ['pop01','qpf01','t','dp','wspd','wgst']

myFmt = DateFormatter("%d%b\n%HZ")
fig, axes = plt.subplots(len(products),1,figsize=(12,8),sharex=True,subplot_kw={'xlim': (start_time,end_time)})
font = {'weight' : 'normal',
        'size'   : 24}
plt.suptitle('NBM hourly Guidance - KGRR')
font = {'weight' : 'normal', 'size'   : 12}
for y,a in zip(products,axes.ravel()):
    a.xaxis.set_major_formatter(myFmt)
    a.set_ylim(prods[y]['ymin'],prods[y]['ymax'])
    if y != 'pop01' and y != 'qpf01':
        a.plot(prods[y]['data'],color=prods[y]['color'])
    else:
        a.bar(pd_series,prods[y]['data'],width=1/26, color=prods[y]['color'])

"""
ax1.plot(dp,'g--')
ax1.plot(t,'r--')
ax2.plot(wspd,'b')
ax2.plot(wgst,'b--')
ax3.bar(pd_series,ppi,width=1/26)
ax3.plot(p_sn,'b--')
ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax4.bar(pd_series,qp01,width=1/26)
ax5.bar(pd_series,sn01,width=1/26)
"""