# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 14:43:12 2020

@author: thomas.turnage

virtual card - https://vlab.ncep.noaa.gov/web/mdl/nbm-textcard-v4.0

"""

element_dict = {'P01':'PoP','P06':'PoP',
                'Q01':'QPF','Q06':'QPF',
                'S01':'SN','S06':'SN',
                'I01':'ZR','I06': 'ZR',
                'T01':'TST', 'T03':'TST','T06':'TST'}


import re
import requests
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

class NBM:

    
    from my_nbm_functions import basic_elements, hourly_elements, short_elements, categorize
    from my_nbm_functions import temperature_bounds, prods, wind_chill, GridShader, nbm_station_dict
    station_master = nbm_station_dict(scripts_dir)
    def __init__(self, station, bulletin_type, download=True, plot_flag=True):
        self.station = station   # single station
        self.station_description = self.station_master[self.station]['name']
        self.bulletin_type = bulletin_type
        self.download = download
        self.plot_flag = plot_flag
        if self.bulletin_type == 'nbhtx':
            self.elements = self.basic_elements + self.hourly_elements
            self.fcst_type = 'Hourly'
        else:
            self.elements = self.basic_elements + self.short_elements
            self.fcst_type = 'Short Term'            

        self.raw_file = f'nbm_raw_{self.bulletin_type}.txt'
        self.trimmed_file = f'nbm_trimmed_{self.bulletin_type}.txt'
        self.name = 'nbm'
        #self.idx = None
        self.tz_shift = 5
        self.now = datetime.utcnow()
        self.current_hour = self.now.replace(minute=0, second=0, microsecond=0)
        self.model_download_time = self.current_hour - timedelta(hours=1)

        self.raw_path = os.path.join(NBM_dir, self.raw_file)
        self.trimmed_path = os.path.join(NBM_dir, self.trimmed_file)
        
        self.data = []
        
        self.nbm = None
        self.nbm_old = None
        self.products = ['apraf_ts','t_bar','wind','acqp_bar','acsn_bar','aczr_bar']
        #self.products = ['apra_ts','t_bar','wind','acqp_bar', 'winter_bar']
        self.master()
        

    def master(self):
        self.get_nbm()
        self.create_trimmed_file()
        self.make_idx()
        self.create_df()
        self.expand_df()
        if self.plot_flag:
            self.plot()
        if self.bulletin_type == 'nbhtx':
            self.taf()

    def get_nbm(self):
        if self.download:
            self.ymd = self.model_download_time.strftime('%Y%m%d')
            self.hour = self.model_download_time.strftime('%H')
            self.url = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod/blend.' + self.ymd + '/' + self.hour + '/text/blend_' + self.bulletin_type + '.t' + self.hour + 'z'
            self.r = requests.get(self.url)
            print('downloading ... ' + str(self.url))
            open(self.raw_path, 'wb').write(self.r.content)
            print('Download Complete!')
        else:
            print('Nothing Downloaded!')
        
        return


        
    def file_path(self):
        self.raw_file = f'nbm_raw_{self.bulletin_type}.txt'
        self.trimmed_file = f'nbm_trimmed_{self.bulletin_type}.txt'
        self.raw_path = os.path.join(self.NBM_dir, self.raw_file)
        self.trimmed_path = os.path.join(self.NBM_dir, self.trimmed_file)



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

        self.run_dt_local = self.run_ymdh - timedelta(hours=self.tz_shift)

        if self.bulletin_type == 'nbhtx':
            self.fh0_utc = self.run_ymdh + timedelta(hours=1)
            self.idx0_utc = self.fh0_utc - timedelta(hours=1)


        elif self.bulletin_type == 'nbstx':
            if self.run_ymdh.hour%3 == 0:
                self.hr_delta = 6
            elif self.run_ymdh.hour%3 == 1:
                self.hr_delta = 5            
            else:
                self.hr_delta = 4
                
            self.fh0_utc = self.run_ymdh + timedelta(hours=self.hr_delta)
            self.idx0_utc = self.fh0_utc - timedelta(hours=3)


        self.idx0_local = self.idx0_utc - timedelta(hours=self.tz_shift)
        self.fh0_local = self.fh0_utc - timedelta(hours=self.tz_shift)

    
        pTime_local = pd.Timestamp(self.idx0_local)
        if self.bulletin_type == 'nbhtx':
            self.idx = pd.date_range(pTime_local, periods=27, freq='H')
            #self.idx6 = None
        else:
            #p6Time = pd.Timestamp(self.idx6_local)
            #self.idx6 = pd.date_range(p6Time, periods=8, freq='6H')
            self.idx = pd.date_range(pTime_local, periods=25, freq='3H')
        
        self.data_times = self.idx[1:-1]
        return self.data_times

    def create_trimmed_file(self):
        self.stn = re.compile(self.station)
        self.sol = re.compile('SOL')
        self.ymdh = re.compile('[0-9]+/[0-9]+/[0-9]+\s+[0-9]+')
        self.column_list = []
        self.stn_found = False
        self.sol_found = False
        with open(self.raw_path, 'r') as self.raw:          
            with open(self.trimmed_path, 'w') as self.trimmed:
                for self.line in self.raw:
                    self.stn_match = self.stn.search(self.line)
                    self.ymdh_match = self.ymdh.search(self.line)
                    self.sol_match = self.sol.search(self.line)
                    if (self.stn_match is not None and self.stn_found == False):
                        self.station = self.stn_match[0]
                        self.dt_str = self.ymdh_match[0]
                        self.run_ymdh = datetime.strptime(self.ymdh_match[0], '%m/%d/%Y  %H%M')
                        self.idx = self.make_idx()
                        self.stn_found = True
                        
                    if self.stn_found and self.sol_match is None:
                        self.start = str(self.line[1:4])
                        if self.start in self.elements:
                            if self.start in element_dict:
                                self.start = element_dict[self.start]
                                
                            self.column_list.append(self.start)
                            self.trimmed.write(self.line)

                    
                    if self.stn_found and self.sol_match is not None:
                        return


    def create_df(self):
        if self.bulletin_type == 'nbhtx':
            self.nbm_old = pd.read_fwf(self.trimmed_path, widths=(5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3))
        else:
            self.nbm_old = pd.read_fwf(self.trimmed_path, widths=(5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3))

        self.nbm = self.nbm_old.transpose()
        self.old_column_names = self.nbm.columns.tolist()
        try:
            self.nbm.drop(['UTC'], inplace=True)
        except:
            pass

        self.nbm.set_index(self.data_times, inplace=True)
        self.elements = self.column_list[1:]
        self.col_rename_dict = {i:j for i,j in zip(self.old_column_names,self.elements)}
        self.nbm.rename(columns=self.col_rename_dict, inplace=True)
        return


    # To plot time series lines  -- slice the dataframe with 'loc'
    # To plot a bar graph        -- convert the slice to a list.
    # Either can be done independently, but I usually do both to have the later 
    # option of plotting either way.
    def expand_df(self):

        #self.pop_bar = self.nbm.PoP.tolist()

        #self.pop_ts = self.nbm['PoP'] 
        self.nbm['PoPF'] = self.nbm.PoP.ffill()
        self.popf_ts =self.nbm.loc[:, ['PoPF']]
        #self.pop_fill_ts[0] = 0.0
        self.prods['popf_ts']['data'] = self.popf_ts
        #self.prods['pop_ts']['data'] = self.pop_ts
        self.popf_bar = self.nbm.PoPF.to_list()
        self.prods['popf_bar']['data'] = self.popf_bar
        

        self.nbm['APTSF'] = self.nbm.TST.ffill()
        # this in not conditional probability like the others
        self.aptsf_bar = self.nbm.APTSF.to_list()
        self.aptsf_ts = self.nbm.loc[:, ['APTSF']]
        self.prods['aptsf_bar']['data'] = self.aptsf_bar
        self.prods['aptsf_ts']['data'] = self.aptsf_ts

        self.nbm['PRAF'] = self.nbm.PRA.ffill()
        self.nbm['APRAF'] = self.nbm['PoPF'] * self.nbm['PRAF']/100
        self.apraf_bar = self.nbm.APRAF.to_list()
        self.apraf_ts = self.nbm.loc[:, ['APRAF']]
        self.prods['apraf_bar']['data'] = self.apraf_bar
        self.prods['apraf_ts']['data'] = self.apraf_ts

        self.nbm['PRAF'] = self.nbm.PRA.ffill()
        self.nbm['APRAF'] = self.nbm['PoPF'] * self.nbm['PRAF']/100
        self.apraf_bar = self.nbm.APRAF.to_list()
        self.apraf_ts = self.nbm.loc[:, ['APRAF']]
        self.prods['apraf_bar']['data'] = self.apraf_bar
        self.prods['apraf_ts']['data'] = self.apraf_ts

        self.nbm['PSNF'] = self.nbm.PSN.ffill()
        self.nbm['APSNF'] = self.nbm['PoPF'] * self.nbm['PSNF']/100
        self.apsnf_bar = self.nbm.APSNF.to_list()
        self.apsnf_ts = self.nbm.loc[:, ['APSNF']]
        self.prods['apsnf_bar']['data'] = self.apsnf_bar
        self.prods['apsnf_ts']['data'] = self.apsnf_ts

        self.nbm['PZRF'] = self.nbm.PZR.ffill()
        self.nbm['APZRF'] = self.nbm['PoPF'] * self.nbm['PZRF']/100
        self.apzrf_bar = self.nbm.APZRF.to_list()
        self.apzrf_ts = self.nbm.loc[:, ['APZRF']]
        self.prods['apzrf_bar']['data'] = self.apzrf_bar
        self.prods['apzrf_ts']['data'] = self.apzrf_ts

        self.nbm['PPLF'] = self.nbm.PPL.ffill()
        self.nbm['APPLF'] = self.nbm['PoPF'] * self.nbm['PPLF']/100
        self.applf_bar = self.nbm.APPLF.to_list()
        self.applf_ts = self.nbm.loc[:, ['APPLF']]
        self.prods['applf_bar']['data'] = self.applf_bar
        self.prods['applf_ts']['data'] = self.applf_ts

    
        self.nbm['SN'] = self.nbm['SN']*0.1
        self.sn_bar = self.nbm.SN.tolist()
        self.prods['sn_bar']['data'] = self.sn_bar

        self.nbm['ACSN'] = self.nbm['SN'].cumsum()
        self.acsn_max = self.nbm.ACSN.max()
        if self.acsn_max > 7:
            self.prods['acsn_bar']['ymax'] = 13
            self.prods['acsn_bar']['yticks'] = [0,2,4,6,8,10,12] 
            self.prods['acsn_bar']['ytick_labels'] = ['0','2','4','6','8','10','12']            
        elif self.acsn_max > 4:
            self.prods['acsn_bar']['ymax'] = 8
            self.prods['acsn_bar']['yticks'] = [0,1,2,4,6,8] 
            self.prods['acsn_bar']['ytick_labels'] = ['0','1','2','4','6','8']
        elif self.acsn_max > 2:
            self.prods['acsn_bar']['ymax'] = 4
            self.prods['acsn_bar']['yticks'] = [0,1,2,3,4] 
            self.prods['acsn_bar']['ytick_labels'] = ['0','1','2','3','4']
        else:
            self.prods['acsn_bar']['ymax'] = 2.5
            self.prods['acsn_bar']['yticks'] = [0,0.5,1,2] 
            self.prods['acsn_bar']['ytick_labels'] = ['0','0.5','1','2']            

        self.acsn_bar = self.nbm.ACSN.tolist()
        self.prods['acsn_bar']['data'] = self.acsn_bar
        
        self.nbm['QPF'] = self.nbm['QPF']*0.01
        self.qp_bar = self.nbm.QPF.tolist()
        self.prods['qp_bar']['data'] = self.qp_bar

        self.nbm['ACQP'] = self.nbm['QPF'].cumsum()
        self.acqp_max = self.nbm.ACQP.max()
        if self.acqp_max > 8:
            self.prods['acqp_bar']['ymax'] = 11
            self.prods['acqp_bar']['yticks'] = [0,1,2,4,6,8,10] 
            self.prods['acqp_bar']['ytick_labels'] = ['0','1','2','4','6','8','10']            
        elif self.acqp_max > 6:
            self.prods['acqp_bar']['ymax'] = 6.5
            self.prods['acqp_bar']['yticks'] = [0,1,2,4,6] 
            self.prods['acqp_bar']['ytick_labels'] = ['0','1','2','4','6']
        elif self.acqp_max > 4:
            self.prods['acqp_bar']['ymax'] = 6.1
            self.prods['acqp_bar']['yticks'] = [0,1,2,4,6] 
            self.prods['acqp_bar']['ytick_labels'] = ['0','1','2','4','6']
        elif self.acqp_max > 3:
            self.prods['acqp_bar']['ymax'] = 4.1
            self.prods['acqp_bar']['yticks'] = [0,1,2,3,4] 
            self.prods['acqp_bar']['ytick_labels'] = ['0','1','2','3','4']
        elif self.acqp_max > 2:
            self.prods['acqp_bar']['ymax'] = 3.1
            self.prods['acqp_bar']['yticks'] = [0,1,2,3] 
            self.prods['acqp_bar']['ytick_labels'] = ['0','1','2','3']
        elif self.acqp_max > 1:
            self.prods['acqp_bar']['ymax'] = 2.1
            self.prods['acqp_bar']['yticks'] = [0, 0.5, 1, 1.5, 2] 
            self.prods['acqp_bar']['ytick_labels'] = ['0', '0.5', '1', '1.5', '2']
        else:
            self.prods['acqp_bar']['ymax'] = 1.1
            self.prods['acqp_bar']['yticks'] = [0, 0.25, 0.5, 0.75, 1] 
            self.prods['acqp_bar']['ytick_labels'] = ['0', '0.25', '0.50', '0.75', '1.00']            


        self.acqp_bar = self.nbm.ACQP.tolist()
        self.prods['acqp_bar']['data'] = self.acqp_bar

        self.nbm['ZR'] = self.nbm['ZR']*0.01
        self.zr_bar = self.nbm.ZR.tolist()
        self.prods['zr_bar']['data'] = self.zr_bar
        
        self.nbm['ACZR'] = self.nbm['ZR'].cumsum()
        self.aczr_bar =self.nbm.ACZR.tolist()
        self.prods['aczr_bar']['data'] = self.aczr_bar

        self.nbm['VIS'] = self.nbm['VIS']*0.1


        self.nbm['SKT'] = self.nbm['WSP'] * 1.15
        self.nbm['GKT'] = self.nbm['GST'] * 1.15
        self.decimals = pd.Series([1, 1], index=['SKT', 'GKT'])
        self.nbm.round(self.decimals)


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


        self.wgst_bar = self.nbm.GST.tolist()
        self.wspd_list_kt = self.nbm.WSP.tolist()
        self.wspd_list = np.multiply(self.wspd_list_kt,1.151)
        self.wspd_list = [round(x) for x in self.wspd_list]
        self.wspd_arr = np.asarray(self.wspd_list)
        self.wspd_list = self.wspd_arr.astype(int)

        self.wgst_list_kt = self.nbm.GST.tolist()
        self.wgst_list = np.multiply(self.wgst_list_kt,1.151)
        self.wgst_list = [round(x) for x in self.wgst_list]
        self.wgst_arr = np.asarray(self.wgst_list)
        self.wgst_list = self.wgst_arr.astype(int)


        self.sky_list = self.nbm.SKY.tolist()

        if self.bulletin_type == 'nbhtx':
            self.cig_list = self.nbm.CIG.tolist()
            self.lcb_list = self.nbm.LCB.tolist()
            self.vis_list = self.nbm.VIS.tolist()        

        self.wc_bar = []
        self.ttfb_bar = []
        for chill in range(0,len(self.wspd_bar)):
            t = self.t_bar[chill]
            s = self.wspd_bar[chill]
            self.wc = 35.74 + 0.6215*t - 35.75*(s**0.16) + 0.4275*t*(s**0.16)
            if self.wc <= t:
                self.final = round(self.wc)
            else:
                self.final = round(t)

            if self.final >= -15:   #   0 to -15 : >30 minutes
                self.fbt = 4
            if self.final < -15:    # -15 to -30 : 15 to 30 minutes
                self.fbt = 3
            if self.final < -30:    # -30 to -50 : <15 minutes
                self.fbt = 2
            if self.final < -50:    #    <  -50  : < 5 minutes
                self.fbt = 1

            self.wc_bar.append(self.final)
            self.ttfb_bar.append(self.fbt)

        self.wc_bar = np.asarray(self.wc_bar, dtype=np.float32)
        #self.wc_cat = self.categorize(self.wc_bar,'wc')
        self.wc_bar_shift = self.wc_bar + 20
        self.prods['wc_bar']['data'] = self.wc_bar_shift
        #map_plot_stations[key] = {'lon':lon,'lat':lon,'wc_cat':wc_cat[0]}

        # Temp (t) and wind chill (wc) go on same panel, 
        # so using min(wc) and max(t) to define bounds for 'twc'
        # using a temperature_bounds function
        #self.twc_tick_list,self.twc_tick_labels = self.temperature_bounds(self.t_bar_shift,self.wc_bar_shift)

        # sometimes we have to convert units because they're in tenths or hundredths

        self.vis = self.nbm.loc[:, ['VIS']]
        self.vis.clip(upper=7.0,inplace=True)
        self.vis_list = self.nbm.VIS.tolist()
        return


    def taf(self):

        self.taf_dict = {}
        
        def ccalc(cig,hour='current'):
            if str(cig) == '-88':
                cigf = ''
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
            else:
                cigf = 'VV' + '{:03d}'.format(cig)
                ccat = 0

            return cigf, ccat

        def vcalc(vsby,hour='currrent'):
            #vsby_prev = self.vis_list[t-0]
            vsby = self.vis_list[t]
            if vsby > 6:
                visf = 'P6'
                vcat = 5        #'VFR'
            elif vsby >= 3:
                visf = '{:.0f}'.format(vsby)
                vcat = 4        #'MVFR'
            elif vsby >= 2:
                visf = '{:.0f}'.format(vsby)
                vcat = 3     
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


        for t in range(1,len(self.data_times)):
            dt = self.data_times[t]
            dts = self.data_times[t].strftime('FM%d%H%M')

            wdir_prev = self.wdir_list[t-1]
            wdir = self.wdir_list[t]
            wdirf = '{:02d}'.format(wdir) + '0'
            wdir_diff = np.absolute((wdir_prev + 50) - (wdir + 50))

            wspd_prev = self.wspd_list[t-1]
            wspd = self.wspd_list[t]
            wspdf = '{:02d}'.format(wspd)
            wspd_diff = np.absolute(wspd_prev - wspd)

            wgst_prev = self.wgst_list[t-0]
            wgst = self.wgst_list[t]

            delta_wgst = wgst - wgst_prev
            spd_gst_diff = wgst - wspd
            if (spd_gst_diff > 8 and wspd > 8) or (spd_gst_diff > 5 and wspd > 12) or delta_wgst > 8:
                g = f'G{wgst}KT'
            else:
                g =''

            
            wx = ''
            if self.aptsf_bar[t] > 60:
                wx = wx + 'TS'
                nots = False
            else:
                nots = True

            if self.apraf_bar[t] > 60:
                wx = wx + 'RA'
                nora = False
            else:
                nora = True

            if self.applf_bar[t] > 60:
                wx = wx + 'PL'
                nopl = False
            else:
                nopl = True

            if self.apsnf_bar[t] > 60:
                wx = wx + 'SN'
                nosn = False
            else:
                nosn = True

            if self.apzrf_bar[t] > 60:
                wx = wx + 'ZR'
                nozr = False
            else:
                nozr = True

            if (wx == ''):
                if self.vis_list[t] < 3:
                    wx = 'FG'
                elif self.vis_list[t] < 6:
                    wx = 'BR'

                     
            cigf_prev,ccat_prev = ccalc(self.cig_list[t-1])
            cigf,ccat = ccalc(self.cig_list[t])            
            ccat_diff = np.absolute(ccat_prev - ccat)


            lcb = self.lcb_list[t]
            if np.abs(self.cig_list[t] - lcb) > 2:
                lcbf = 'SCT' + '{:03d}'.format(lcb)
            else:
                lcbf = ''
            
            
            visf_prev,vcat_prev = vcalc(self.vis_list[t-0])         
            visf,vcat = vcalc(self.vis_list[t])
            vcat_diff = np.absolute(vcat - vcat_prev)
            
            

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

            c_change = False
            w_change = False


            if np.max([vcat_diff,ccat_diff]) > 0:
                c_change = True

            if  (wdir_diff > 30 and wspd > 8) or delta_wgst > 10:
                w_change = True
            
            #elements = [dts, wdirf, wspdf, g, visf, lcbf, cigf, change]
            line_str = '{} {}{}KT{} {}SM {} {} {} {}'.format(dts, wdirf, wspdf, g, visf, wx, lcbf, cigf, c_change)
            if w_change or c_change:
                
                print(line_str)

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



test = NBM('KDLH', 'nbhtx', False, False)

