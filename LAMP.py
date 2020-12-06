# -*- coding: utf-8 -*-
"""
Created on Sat 12/5/2020

@author: thomas.turnage

virtual card - https://www.weather.gov/mdl/lamp_desc

UTC Hour of the day in UTC time. This is the hour at which the forecast is valid, or if the forecast is valid for a period, the

WDR wind direction in tens of degrees.
WSP wind speed in knots.
WGS wind gust in knots.  "NG" = no gust
PPO probability of precipitation (even non-measurable)
P06 probability of measurable precipitation (PoP) during a 6-h period ending at that time.
PCO categorical forecast of yes (Y) or no (N) of any precipitation
LP1 probability of lightning (at least one total lightning flash) previous hour
LC1 categorical forecast of no (N), low (L), medium (M), or high (H) prob lightning past hour
CP1 probability of lightning flash and/or Z >= 40dBZ occurring past hour.
CC1 categorical forecast of no (N), low (L), medium (M), or high (H) for above
POZ* conditional probability of freezing pcp occurring at the hour. 
POS* conditional probability of snow occurring at the hour. 
TYP* conditional precipitation type at the hour.
CLD forecast categories of total sky cover valid at that hour.
CIG ceiling height categorical forecasts at the hour.
CCG conditional ceiling height categorical forecasts if precipitation occurring.
VIS visibility categorical forecasts at the hour.
CVS conditional visibility categorical forecasts if pcpn occurring at the hour
OBV obstruction to vision categorical forecasts at the hour.

"""




import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.pyplot as plt
import matplotlib.transforms
import math


try:
    os.listdir('/usr')
    scripts_dir = '/data/scripts'
except:
    scripts_dir = 'C:/data/scripts'
    sys.path.append(os.path.join(scripts_dir,'resources'))

NBM_dir = os.path.join(scripts_dir,'NBM')
ymdh = re.compile('[0-9]+/[0-9]+/[0-9]+\s+[0-9]+')
mtr_ob = re.compile('(\S*KT.*(?=\S{2,4}/))')

class LAMP:

    from my_nbm_functions import basic_elements, hourly_elements, categorize
    from my_nbm_functions import prods, GridShader, nbm_station_dict
    station_master = nbm_station_dict(scripts_dir)
    def __init__(self, station, download=True, plot_flag=True):
        self.station = station   # single station
        self.station_description = self.station_master[self.station]['name']
        self.bulletin_type = 'glamp'
        self.download = download

        self.raw_file = f'nbm_raw_{self.bulletin_type}.txt'
        self.trimmed_file = f'nbm_trimmed_{self.bulletin_type}.txt'
        self.name = 'lamp'
        #self.idx = None

        self.now = datetime.utcnow()
        self.current_hour = self.now.replace(minute=0, second=0, microsecond=0)
        self.model_download_time = self.current_hour - timedelta(hours=1)

        self.raw_path = os.path.join(NBM_dir, self.raw_file)
        self.trimmed_path = os.path.join(NBM_dir, self.trimmed_file)
        
        self.column_list = []
        self.data = []
        
        self.taf = None
        self.taf_old = None
        self.master()
        

    def master(self):
        self.get_lamp()
        #self.create_trimmed_file()
        self.make_idx()
        self.create_df()
        self.expand_df()
        # self.get_observation()
        # self.taf()
        # if self.plot_flag:
        #     self.plot()



    def get_lamp(self):
        if self.download:
            self.url = 'https://www.nws.noaa.gov/cgi-bin/lamp/getlav.pl?sta=' + self.station
            self.r = requests.get(self.url)
            self.soup = BeautifulSoup(self.r.text, 'html.parser')
            self.tag = self.soup.findAll('pre')[0]
            self.bull_text = self.tag.string

            #print('downloading ... ' + str(self.url))
            #GFS LAMP 2030 UTC  12/05/2020
            #ymdh = re.compile('[0-9]+/[0-9]+/[0-9]+\s+[0-9]+')
            ymd = re.compile('[0-9]+/[0-9]+/[0-9]+')
            h = re.compile('(\d.*(?=\d\d\sUTC))')
            dt = ymd.search(self.bull_text.string)
            hr = h.search(self.bull_text.string)
            if dt is not None and hr is not None:
                self.ymd = dt[0]
                self.hr = hr[0]
                self.ymdh_str = self.ymd + self.hr
                self.ymdh_str = '{} {}'.format(self.ymd, self.hr)
                print(self.ymdh_str)
            else:
                print('missing match!')
            self.run_ymdh = datetime.strptime(self.ymdh_str,'%m/%d/%Y %H')
            self.fh0 = self.run_ymdh + timedelta(hours=1)
            blines = self.bull_text.splitlines()
            self.fixed = '\n'.join(blines[2:-1])
            lines = self.fixed.splitlines()
            for l in lines:
                self.column_list.append(l[1:4])
            with open(self.trimmed_path, 'w') as self.trimmed:
                self.trimmed.write(self.fixed)
            return
        
        
    def get_observation(self):
            #https://w1.weather.gov/data/METAR/KGRR.1.txt
            mtr_ob = re.compile('(\S*KT.*(?=\S{2,4}/))')
            self.url = 'https://w1.weather.gov/data/METAR/' + self.station + '.1.txt'
            self.mtr_raw_path = os.path.join(NBM_dir, self.station + '.txt')
            self.r = requests.get(self.url)
            self.mtr_text = str(self.r.content)
            self.mtr_match = mtr_ob.search(str(self.mtr_text))
            if (self.mtr_match is not None):
                print(self.mtr_match[0])
                self.ob_str = self.mtr_match[0]
            else:
                self.ob_str = 'No match!'
            return


    # def file_path(self):
    #     self.raw_file = f'nbm_raw_{self.bulletin_type}.txt'
    #     self.trimmed_file = f'nbm_trimmed_{self.bulletin_type}.txt'
    #     self.raw_path = os.path.join(self.NBM_dir, self.raw_file)
    #     self.trimmed_path = os.path.join(self.NBM_dir, self.trimmed_file)


    def make_idx(self):
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



        self.fh0_utc = self.run_ymdh + timedelta(hours=1)
        self.idx0_utc = self.fh0_utc - timedelta(hours=1)
        self.idx = pd.date_range(self.idx0_utc, periods=27, freq='H')

        
        self.data_times = self.idx[1:-1]
        return self.data_times


    def create_df(self):

        self.nbm_old = pd.read_fwf(self.trimmed_path, header=[0], skip_blank_lines=True, widths=(5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3))
        self.nbm = self.nbm_old.transpose()
        try:
            self.nbm.drop(['UTC'], inplace=True)
        except:
            pass
        self.old_column_names = self.column_list

        #self.nbm.set_index(self.data_times, inplace=True)
        self.elements = self.column_list
        els = [k for k in range(0,len(self.elements))]
        
        self.col_rename_dict = {i:j for i,j in zip(els[:-1],self.elements[1:])}
        self.nbm.rename(columns=self.col_rename_dict, inplace=True)
        self.nbm.set_index(self.data_times, inplace=True)
        return


    # To plot time series lines  -- slice the dataframe with 'loc'
    # To plot a bar graph        -- convert the slice to a list.
    # Either can be done independently, but I usually do both to have the later 
    # option of plotting either way.
    def expand_df(self):

        #self.pop_bar = self.nbm.PoP.tolist()

        #self.pop_ts = self.nbm['PoP'] 
        self.nbm['PoPF'] = self.nbm.PPO.ffill()
        self.popf_ts =self.nbm.loc[:, ['PoPF']]
        #self.pop_fill_ts[0] = 0.0
        self.prods['popf_ts']['data'] = self.popf_ts
        #self.prods['pop_ts']['data'] = self.pop_ts
        self.popf_bar = self.nbm.PoPF.to_list()
        self.prods['popf_bar']['data'] = self.popf_bar
        

        # self.nbm['APTSF'] = self.nbm.TST.ffill()
        # # this in not conditional probability like the others
        # self.aptsf_bar = self.nbm.APTSF.to_list()
        # self.aptsf_ts = self.nbm.loc[:, ['APTSF']]
        # self.prods['aptsf_bar']['data'] = self.aptsf_bar
        # self.prods['aptsf_ts']['data'] = self.aptsf_ts



        self.nbm['PSNF'] = self.nbm.POS.ffill()
        self.nbm['APSNF'] = self.nbm['PoPF'] * self.nbm['PSNF']/100
        self.apsnf_bar = self.nbm.APSNF.to_list()
        self.apsnf_ts = self.nbm.loc[:, ['APSNF']]
        self.prods['apsnf_bar']['data'] = self.apsnf_bar
        self.prods['apsnf_ts']['data'] = self.apsnf_ts

        self.nbm['PZRF'] = self.nbm.POZ.ffill()
        self.nbm['APZRF'] = self.nbm['PoPF'] * self.nbm['PZRF']/100
        self.apzrf_bar = self.nbm.APZRF.to_list()
        self.apzrf_ts = self.nbm.loc[:, ['APZRF']]
        self.prods['apzrf_bar']['data'] = self.apzrf_bar
        self.prods['apzrf_ts']['data'] = self.apzrf_ts


        self.nbm['VIS'] = self.nbm['VIS']*0.1



        self.t_bar = self.nbm.TMP.tolist()
        self.t_bar = np.asarray(self.t_bar, dtype=np.float32)
        self.t_bar_shift = self.t_bar + 20
        self.prods['t_bar']['data'] = self.t_bar_shift
        
        self.dp_bar = self.nbm.DPT.tolist()
        self.dp_bar = np.asarray(self.dp_bar, dtype=np.float32)        
        
        self.wdir_list = self.nbm.WDR.tolist()

        self.wdir = self.nbm.loc[:, ['WDR']]
        self.wspd = self.nbm.loc[:, ['WSP']]
        self.wdir_bar = self.nbm.WDR.tolist()
        self.wspd_bar = self.nbm.WSP.tolist()
        self.u_norm = []
        self.v_norm = []
        for w in range(0,len(self.wspd_bar)):
            u = (math.sin(math.radians(self.wdir_bar[w])) * self.wspd_bar[w]) * -1.0
            v = (math.cos(math.radians(self.wdir_bar[w])) * self.wspd_bar[w]) * -1.0
            un = u / np.sqrt(u**2 + v**2)
            vn = v / np.sqrt(u**2 + v**2)
            self.u_norm.append(un)
            self.v_norm.append(vn)


        self.wgst_bar = self.nbm.WGS.tolist()
        self.wspd_list = self.nbm.WSP.tolist()
        self.wgst_list = self.nbm.WGS.tolist()
        self.sky_list = self.nbm.SKY.tolist()
        self.cig_list = self.nbm.CIG.tolist()
        self.vis_list = self.nbm.VIS.tolist()        



    def taf(self):
        self.taf_df = self.nbm
        self.taf_dict = {}
        
        def ccalc(cig,hour='current'):
            if str(cig) == '-88':
                cigf = ''
                ccat = 5
            elif cig >= 35:
                cigf = 'BKN' + '{:03d}'.format(cig)
                ccat = 5
            elif cig >= 30:
                cigf = 'BKN' + '{:03d}'.format(cig)
                ccat = 4
            elif cig >= 20:
                cigf = 'BKN' + '{:03d}'.format(cig)
                ccat = 3
            elif cig >= 10:
                cigf = 'BKN' + '{:03d}'.format(cig)
                ccat = 2
            elif cig >= 5:
                cigf = 'OVC' + '{:03d}'.format(cig)
                ccat = 1
            elif cig >= 3:
                cigf = 'OVC' + '{:03d}'.format(cig)
                ccat = 0
            else:
                cigf = 'VV' + '{:03d}'.format(cig)
                ccat = 0

            return cigf, ccat

        def vcalc(vsby,hour='current'):
            #vsby_prev = self.vis_list[t-0]
            vsby = self.vis_list[t]
            if vsby > 6:
                visf = 'P6'
                vcat = 5        #'VFR'
            elif vsby >= 3:
                visf = '{:.0f}'.format(vsby)
                vcat = 4        #'MVFR'
            # elif vsby >= 2:
            #     visf = '{:.0f}'.format(vsby)
            #     vcat = 3     
            elif vsby >= 1:
                visf = '{:.0f}'.format(vsby)
                vcat = 2        #'IFR'
            elif vsby >= 0.5:
                visf = '3/4'
                vcat = 1        #'LIFR'
            else:
                visf = '1/4'
                vcat = 0        #'VLIFR'

            return visf, vcat

        def wx_str(v):
            thresh = 30
            wx = ''
            q = self.qp_bar[v]
            s = self.sn_bar[v]
            z = self.zr_bar[v]
    
            if self.aptsf_bar[v] > thresh:
                wx = wx + 'TS'
            if self.apraf_bar[v] > thresh:
                wx = wx + 'RA'
#            if self.apraf_bar[v] > thresh or q > 0:
#                wx = wx + 'RA'
            if self.applf_bar[v] > thresh:
                wx = wx + 'PL'
            if self.apsnf_bar[v] > thresh or s > 0:
                wx = wx + 'SN'
            if self.apzrf_bar[v] > thresh or z > 0:
                wx = wx + 'ZR'
            if (wx != ''):
                if self.vis_list[v] < 1:
                    wx = '+' + wx
                elif self.vis_list[v] <= 2:
                    wx = wx
                elif self.vis_list[v] <= 3:
                    wx = '-' + wx + ' BR'
                else:
                    wx = '-' + wx
            else:
                if self.vis_list[v] < 3:
                    wx = 'FG'
                elif self.vis_list[v] < 6:
                    wx = 'BR'
            return wx


        def calc_wgst(wgst):
            spd_gst_diff = wgst - wspd
            if (spd_gst_diff > 8 and wspd > 8) or (spd_gst_diff > 5 and wspd > 12):
                g = f'G{wgst}KT'
                flag = True
            else:
                g =''
                flag = False
            return g,flag


        #--------------------------------------------

        self.idx24 = self.idx0_utc + timedelta(days=1)
        header_dts = self.idx0_utc.strftime('%d%H%M')
        self.taf_text = ''

        dyhr_beg = self.idx0_utc.strftime('%d%H')
        dyhr_end = self.idx24.strftime('%d%H')
        header_str = 'TAF\n{} {}Z {}/{} {}\n'.format(self.station, header_dts, dyhr_beg, dyhr_end, self.ob_str)
        self.taf_text = self.taf_text + header_str

        for t in range(2,len(self.data_times)):
            wc = False
            dt = self.data_times[t]
            
            dts = self.data_times[t].strftime('FM%d%H%M')

            wdir_prev2 = self.wdir_list[t-2]
            wdir_prev = self.wdir_list[t-1]
            wdir = self.wdir_list[t]
            wdirf = '{:02d}'.format(wdir) + '0'
            wdir_diff = np.absolute((wdir_prev + 50) - (wdir + 50))
            wdir_diff2 = np.absolute((wdir_prev2 + 50) - (wdir + 50))
            
            wspd_prev2 = self.wspd_list[t-2]
            wspd_prev = self.wspd_list[t-1]
            wspd = self.wspd_list[t]
            wspdf = '{:02d}'.format(wspd)
            wspd_diff = np.absolute(wspd_prev - wspd)
            wspd_diff2 = np.absolute(wspd_prev2 - wspd)

            #----- GUSTS
            g_prev, g_flag_prev = calc_wgst(self.wgst_list[t-1])
            g, g_flag = calc_wgst(self.wgst_list[t])
            if g_flag_prev != g_flag:
                gc = True
            else:
                gc = False

            if gc is False:
                wspdf = wspdf + 'KT'
            
            if (wdir_diff > 30 and wdir_diff2 > 30 and wspd > 8) or (wspd_diff > 8 and wspd_diff2 > 8) or gc:
                wc = True

            #------ WX    
            wx = wx_str(t)
            wx_prev = wx_str(t-1)
            wx_prev2 = wx_str(t-2)
            
            if wx != wx_prev and wx != wx_prev2:
                delta_wx = True
            else:
                delta_wx = False

            #------ LCB    
            lcb = self.lcb_list[t]
            if np.abs(self.cig_list[t] - lcb) > 2:
                lcbf = 'SCT' + '{:03d}'.format(lcb)
            else:
                lcbf = ''

            #------ CIG
            cigf_prev2,ccat_prev2 = ccalc(self.cig_list[t-2])
            cigf_prev,ccat_prev = ccalc(self.cig_list[t-1])
            cigf,ccat = ccalc(self.cig_list[t])            
            ccat_diff = np.absolute(ccat_prev - ccat)
            ccat_diff2 = np.absolute(ccat_prev2 - ccat)


            #------ VIS                            
            visf_prev2,vcat_prev2 = vcalc(self.vis_list[t-2]) 
            visf_prev,vcat_prev = vcalc(self.vis_list[t-1])         
            visf,vcat = vcalc(self.vis_list[t])
            vcat_diff = np.absolute(vcat - vcat_prev)
            vcat_diff2 = np.absolute(vcat - vcat_prev2)
            
            #------ CATEGORY   
            if (vcat_diff > 0 and vcat_diff2 > 0) or (ccat_diff > 0 and ccat_diff2 > 0):
                delta_cat = True
            else:
                delta_cat = False

            
            self.taf_dict[dt] = {'dt_str': dts,
                                 'wdir_str': wdirf,
                                 'delta_wdir': wdirf,
                                 'wspd_str': wspdf,
                                 'delta_wspd': wspd_diff,
                                 'wgst_str': g,
                                 'cig_str': cigf,
                                 'lcb_str': lcbf,
                                 'delta_ccat': ccat_diff,
                                 'vis_str': visf,
                                 'wx_str': wx,
                                 'delta_vcat': vcat_diff
                                 }    


            
            line_str = '{} {}{}{} {}SM {} {} {}'.format(dts, wdirf, wspdf, g, visf, wx, lcbf, cigf)
            taf_line = re.sub('\s+',' ',line_str)
            taf_line = '     ' + taf_line + '\n'

            if wc or delta_cat or delta_wx:
                self.taf_text = self.taf_text + taf_line

            
        self.taf_text = self.taf_text[:-1] + '=\n'
        print(self.taf_text)



    def u_v_components(self):
        # since the convention is "direction from"
        # we have to multiply by -1
        # If an arrow is drawn, it needs a dx of 2/(number of arrows) to fit in the row of arrows
        u = (math.sin(math.radians(self.this_wdir)) * self.this_wspd) * -1.0
        v = (math.cos(math.radians(self.this_wdir)) * self.this_wspd) * -1.0
        return u,v

    def plotting(self):
        self.gs = self.GridShader(self.a, facecolor="lightgrey", first=True, alpha=0.0) 
        self.a.set_yticks(self.prods[self.y]['yticks'], minor=False)
        self.a.set_yticklabels(self.prods[self.y]['ytick_labels'],minor=False)
        self.a.grid(which='major', axis='y')
        self.a.set_xticks(self.data_times)
        self.a.set_ylim(self.prods[self.y]['ymin'],self.prods[self.y]['ymax'])
        return
           
    def plot(self):
        hours = mdates.HourLocator()
        myFmt = DateFormatter("%d%h")
        myFmt = DateFormatter("%d%b\n%HZ")
        myFmt = DateFormatter("%I\n%p")
        myFmt = DateFormatter("%I")    
        
        bar_align = "center"   # "edge"
        if self.bulletin_type == 'nbhtx':
            reg_bar_width = 1/35
            pop_bar_width = 1/32
        else:
            reg_bar_width = 4/35
            pop_bar_width = 7/35            
        fig_set = {'4':(14,13),'5':(14,15),'6':(12,18)}
        fig, axes = plt.subplots(len(self.products),1,figsize=fig_set[str(len(self.products))],sharex=False,subplot_kw={'xlim': (self.idx[0],self.idx[-1])})
        #fig = plt.figure(constrained_layout=True,figsize=(12,18),sharex=False,subplot_kw={'xlim': (self.idx[0],self.idx[-1])})    
        #gsp = fig.add_gridspec(6, 1)
        #fig, axes = plt.subplots(len(products),1,figsize=(16,8),sharex=True,subplot_kw={'xlim': (start_time,end_time)})
        plt.subplots_adjust(bottom=0.1, left=0.17, top=0.9)
        
        plt.suptitle(self.fcst_type + ' Forecast -- ' + self.run_dt_local.strftime('%B %d, %Y') + ' ... Updated ' + self.run_dt_local.strftime('%I %p EST') + '\n' +  self.station_description  )
        
    
        for self.y,self.a in zip(self.products,axes.ravel()):
            print(self.y)
    
            dx = 0/72.; dy = 0/72. 
            offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    
            # apply offset transform to all x ticklabels.
    
            for label in self.a.xaxis.get_majorticklabels():
                label.set_transform(label.get_transform() + offset)
    
    
            self.a.xaxis.set_major_locator(hours)
            self.a.xaxis.set_major_formatter(myFmt)
    
            plt.setp( self.a.xaxis.get_majorticklabels(), ha="center",rotation=0 )
            
            self.a.yaxis.set_label_coords(-0.112,0.25)
            self.a.xaxis.grid(True, linewidth=20, color=(0.96,0.96,0.96), zorder=1)  
            #a.xaxis.grid(True, linewidth=20, alpha = 0.12, zorder=1)  
    
    
            if self.y == 'apraf_ts':
                self.plotting()
                this_title = 'Probability of:\n\nAny Precip\n(gray dash)\n\nSnow (blue)\n\nIce (purple)'
                self.a.set_ylabel(this_title, rotation=0)
    

                self.a.plot(self.prods['apraf_ts']['data'],linewidth=2, zorder=10,color=self.prods['apraf_ts']['color'])
                self.a.plot(self.prods['apsnf_ts']['data'],linewidth=2, zorder=8,color=self.prods['apsnf_ts']['color'])
                self.a.plot(self.prods['apzrf_ts']['data'],linewidth=2, zorder=7,color=self.prods['apzrf_ts']['color'])
                self.a.plot(self.prods['popf_ts']['data'],linewidth=6,linestyle=':', zorder=6,color=self.prods['popf_ts']['color'])

 

            if self.y == 'apraf_bar':
                self.plotting()
                this_title = 'Probability of:\n\nAny Precip\n(gray dash)\n\nSnow (blue)\n\nIce (purple)'
                self.a.set_ylabel(this_title, rotation=0)
    
                self.a.bar(self.data_times,self.prods['pop_bar']['data'],width=pop_bar_width, zorder=10,align='center',bottom=self.prods[self.y]['bottom'],color=self.prods['pop_bar']['color'])
                self.a.bar(self.data_times,self.prods['apra_bar']['data'],width=pop_bar_width*0.80, zorder=10,align='center',bottom=self.prods[self.y]['bottom'],color=self.prods['apra_bar']['color'])
                self.a.bar(self.data_times,self.prods['apsn_bar']['data'],width=pop_bar_width*-0.25, zorder=10,align='edge',bottom=self.prods[self.y]['bottom'],color=self.prods['apsn_bar']['color'])
                self.a.bar(self.data_times,self.prods['apzr_bar']['data'],width=pop_bar_width*0.25, zorder=10,align='edge',bottom=self.prods[self.y]['bottom'],color=self.prods['apzr_bar']['color'])
    
    
            if self.y == 'wind':
                plt.rc('font', size=12) 
                self.a.set_ylim(0,1)
    
                dx = 0/72.; dy = 0/72. 
                offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    
                # apply offset transform to all x ticklabels.
                self.a.set_yticks(self.prods[self.y]['yticks'], minor=False)
                self.a.set_yticklabels(self.prods[self.y]['ytick_labels'],minor=False)
                self.a.get_yaxis().set_visible(True)
                this_title = ''
                self.a.set_ylabel(this_title, rotation=0)
                self.a.set_xticks(self.idx)
                self.a.get_xaxis().set_visible(True)
    
                #gs = GridShader(a, facecolor="lightgrey", first=first_gray, alpha=0.5)

                self.a.set_xticks(self.idx)   
                for s,d,g,p in zip(self.wspd_list,self.wdir_list,self.wgst_list,self.data_times):
                    self.this_wspd = s
                    self.this_wdir = d * 10
                    #print(d,s)
                    u,v = self.u_v_components()
                    u_norm = u / np.sqrt(u**2 + v**2)
                    v_norm = v / np.sqrt(u**2 + v**2)
                    self.a.quiver(p, 0.6, u_norm, v_norm, scale=30, width=0.004,color=[0,0,1,0.9], zorder=10,pivot='middle')
                    self.a.text(p, 0.35, f'{s}',horizontalalignment='center',color=[0,0,1,0.9])
                    self.a.text(p, 0.25, f'{g}',horizontalalignment='center',color=[0,0,102/255,1])

                
    
            # specialized treatment for ranges and gridlines
            if self.y in ['acsn_bar','aczr_bar','acqp_bar','sn_bar','zr_bar','qp_bar']:
                self.plotting()
                self.a.bar(self.data_times,self.prods[self.y]['data'],width=pop_bar_width, zorder=10,align=bar_align,bottom=self.prods[self.y]['bottom'],color=self.prods[self.y]['color'])
                self.a.set_ylabel(self.prods[self.y]['title'], rotation=0)
                self.a.get_xaxis().set_visible(True)


            # specialized treatment for ranges and gridlines
            if self.y in ['winter_bar']:
                self.gs = self.GridShader(self.a, facecolor="lightgrey", first=True, alpha=0.0) 
                self.a.grid(which='major', axis='y')
                self.a.set_xticks(self.data_times)
                color = 'tab:red'
                self.a.bar(self.data_times,self.prods['acsn_bar']['data'],width=pop_bar_width, zorder=10,align=bar_align,bottom=self.prods['acsn_bar']['bottom'],color=self.prods['acsn_bar']['color'])
                self.a.set_ylabel(self.prods['acsn_bar']['title'], rotation=0, color=self.prods['acsn_bar']['color'])  # we already handled the x-label with ax1                
      
                
                color = 'tab:blue'
                self.a2 = self.a.twinx()  # instantiate a second axes that shares the same x-axis
                self.a2.bar(self.data_times,self.prods['aczr_bar']['data'],width=pop_bar_width, zorder=10,align=bar_align,bottom=self.prods['aczr_bar']['bottom'],color=self.prods['aczr_bar']['color'])
                self.a2.set_ylabel(self.prods['aczr_bar']['title'], rotation=0, color=self.prods['acsn_bar']['color'])  # we already handled the x-label with ax1
                self.a2.tick_params(axis='y', labelcolor=color)
                

    
            if self.y in ['t_bar','wc_bar']:
                tick_list = self.prods['wc_bar']['yticks']
                tick_labels = self.prods['wc_bar']['ytick_labels']
                self.plotting()
                self.a.set_ylim(tick_list[0],tick_list[-1])
                self.a.bar(self.data_times,self.prods['t_bar']['data'],width=reg_bar_width, zorder=10,align=bar_align,bottom=self.prods[self.y]['bottom'],color=self.prods['t_bar']['color'])
                self.a.bar(self.data_times,self.prods['wc_bar']['data'],width=reg_bar_width, zorder=10,align=bar_align,bottom=self.prods[self.y]['bottom'],color=self.prods['wc_bar']['color'])
                self.a.set_ylabel(self.prods['wc_bar']['title'], rotation=0)
                self.a.get_xaxis().set_visible(True)


        
            # these are lists that use matplotlib bar to create bar graphs
            if self.y in ['apra_bar','apzr_bar','apsn_bar']:
                self.gs = self.GridShader(self.a, facecolor="lightgrey", first=True, alpha=0.0) 
                self.a.set_xticks(self.idx)
                self.a.set_ylim(self.prods[self.y]['ymin'],self.prods[self.y]['ymax'])
                #a.bar(data_list,prods[y]['data'],width=1/25, align="edge",color=prods[y]['color'])
                self.a.bar(self.data_times,self.prods[self.y]['data'],width=pop_bar_width, align="center",color=self.prods[self.y]['color'])
                self.a.set_ylabel(self.prods[self.y]['title'], rotation=0)
                self.a.get_xaxis().set_visible(True)
                self.a.set(yticks = self.prods[self.y]['yticks'], yticklabels = self.prods[self.y]['ytick_labels'])

                
            if self.y in ['ttfb_bar','vis_cat_bar','zr_cat_bar']:
                self.gs = self.GridShader(self.a, facecolor="lightgrey", first=True, alpha=0.0) 
                self.a.grid(which='major', axis='y')
                self.a.set_xticks(self.idx)
                self.a.set_ylim(self.prods[self.y]['ymin'],self.prods[self.y]['ymax'])
                self.a.bar(self.data_times,self.prods[self.y]['data'],width=reg_bar_width,  zorder=10,align=bar_align,color=self.prods[self.y]['color'])
    
                self.a.set_ylabel(self.prods[self.y]['title'], rotation=0)
                self.a.get_xaxis().set_visible(True)
                self.a.set(yticks = self.prods[self.y]['yticks'], yticklabels = self.prods[self.y]['ytick_labels'])
        
        self.image_file = self.station + '_NBM_' + self.bulletin_type + '.png'
        self.image_dst_path = os.path.join(NBM_dir,self.image_file)
        #plt.show()
        plt.savefig(self.image_dst_path,format='png')
        plt.close()
        return



test = LAMP('KCAR', True, False)

