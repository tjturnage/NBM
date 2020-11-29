# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 14:43:12 2020

@author: thomas.turnage

virtual card - https://vlab.ncep.noaa.gov/web/mdl/nbm-textcard-v4.0

"""

element_dict = {'P01':'PoP','P06':'PoP',
                'Q01':'QPF','Q06':'QPF',
                'S01':'SN','S06':'SN',
                'I01':'ICE','I06':'ICE',
                'T01':'TSTM', 'T03':'TSTM','T06':'TSTM'}


import re
import requests
from datetime import datetime, timedelta
import os
import sys
import pandas as pd

try:
    os.listdir('/usr')
    scripts_dir = '/data/scripts'
except:
    scripts_dir = 'C:/data/scripts'
    sys.path.append(os.path.join(scripts_dir,'resources'))

NBM_dir = os.path.join(scripts_dir,'NBM')
ymdh = re.compile('[0-9]+/[0-9]+/[0-9]+\s+[0-9]+')

class NBM:

    from my_nbm_functions import basic_elements, hourly_elements
    
    def __init__(self, station, bulletin_type, download=True):
        self.station = station   # single station
        self.bulletin_type = bulletin_type
        self.download = download
        self.raw_file = f'nbm_raw_{self.bulletin_type}.txt'
        self.trimmed_file = f'nbm_trimmed_{self.bulletin_type}.txt'
        self.name = 'nbm'
        self.tz_shift = 5
        self.now = datetime.utcnow()
        self.current_hour = self.now.replace(minute=0, second=0, microsecond=0)
        self.model_download_time = self.current_hour - timedelta(hours=1)

        self.raw_path = os.path.join(NBM_dir, self.raw_file)
        self.trimmed_path = os.path.join(NBM_dir, self.trimmed_file)
        
        self.data = []
        
        self.nbm = None
        self.nbm_old = None
        self.get_nbm()



    def get_nbm(self):
        if self.download:
            self.ymd = self.model_download_time.strftime('%Y%m%d')
            self.hour = self.model_download_time.strftime('%H')
            self.url = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod/blend.' + self.ymd + '/' + self.hour + '/text/blend_' + self.bulletin_type + '.t' + self.hour + 'z'
            self.r = requests.get(self.url)
            print('downloading ... ' + str(self.url))
            open(self.raw_path, 'wb').write(self.r.content)
            print('Download Complete!')
            return 
        else:
            print('Nothing Downloaded!')
        
        self.create_trimmed_file()

        
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

        if self.bulletin_type == 'nbhtx':
            self.fh0_utc = self.run_ymdh + timedelta(hours=1)

        elif self.bulletin_type == 'nbstx':
            if self.run_ymdh%3 == 0:
                self.hr_delta = 6
            elif self.run_ymdh%3 == 1:
                self.hr_delta = 5            
            else:
                self.hr_delta = 4
            self.fh0_utc = self.run_ymdh + timedelta(hours=self.hr_delta)                
            
        self.fh0_local = self.fh0_utc - timedelta(hours=self.tz_shift)
    
        pTime = pd.Timestamp(self.fh0_local)
        if self.bulletin_type == 'nbhtx':
            idx = pd.date_range(pTime, periods=25, freq='H')
        else:
            idx = pd.date_range(pTime, periods=23, freq='3H')        
        return idx


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
                        if self.start in self.basic_elements or self.start in self.hourly_elements:
                            if self.start in element_dict:
                                self.start = element_dict[self.start]
                                
                            self.column_list.append(self.start)
                            self.trimmed.write(self.line)

                    
                    if self.stn_found and self.sol_match is not None:
                        break

        self.create_df()


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

        self.nbm.set_index(self.idx, inplace=True)
        self.elements = self.column_list[1:]
        self.col_rename_dict = {i:j for i,j in zip(self.old_column_names,self.elements)}
        self.nbm.rename(columns=self.col_rename_dict, inplace=True)
        self.expand_df()

    def expand_df(self):
        self.nbm['SN'] = self.nbm['SN']*0.1
        self.nbm['QPF'] = self.nbm['QPF']*0.01
        self.nbm['VIS'] = self.nbm['VIS']*0.1
        self.nbm['APRA'] = self.nbm['PoP'] * self.nbm['PRA']/100
        self.nbm['APSN'] = self.nbm['PoP'] * self.nbm['PSN']/100
        self.nbm['APZR'] = self.nbm['PoP'] * self.nbm['PZR']/100
        self.nbm['APPL'] = self.nbm['PoP'] * self.nbm['PPL']/100
        self.nbm['ACSN'] = self.nbm['SN'].cumsum()
        self.nbm['ACQP'] = self.nbm['QPF'].cumsum()
        self.nbm['SKT'] = self.nbm['WSP'] * 1.15
        self.nbm['GKT'] = self.nbm['GST'] * 1.15
        self.decimals = pd.Series([1, 1], index=['SKT', 'GKT'])
        self.nbm.round(self.decimals)

        # self.wdir = self.nbm.WDR.tolist()
        # self.wspd_kt = self.nbm.WSP.tolist()
        # self.wspd = self.np.multiply(self.wspd_kt,1.151)
        # self.wspd = [round[x] for x in self.wspd]
        # self.wgst_kt = self.nbm.GST.tolist()
        # self.wgst = self.np.multiply(self.wgst_kt,1.151)
        # self.wgst = [round[x] for x in self.wgst]         

    def colors(self):

    
    def ticks(self):
        self.prob_yticks = [0, 20, 40, 60, 80, 100]
        self.prob_ytick_labels = ["0","20", "40","60","80","100"]
        self.p_min = -5
        self.p_max = 105
       

    # class Avn:
    #      def __init__(self):
    #          self.elements = ('CIG','MVC','IFC','LIC','LCB','MVV','IFV','LIV')

    # # class Short:    
           




test5 = NBM('KDFW', 'nbhtx', False)
#test4.get_nbm()



