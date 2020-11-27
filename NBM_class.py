# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 14:43:12 2020

@author: thomas.turnage
"""

class NBM:
    import os
    import re
    import sys
    from datetime import datetime, timedelta
    import requests
    import pandas as pd
    
    def __init__(self, station, bulletin_type, download=True):

        self.tz_shift = 5
        self.scripts_dir = 'C:/data/scripts'
        self.station = station   # single station
        self.bulletin_type = bulletin_type
        self.stn = self.re.compile(self.station)
        self.sol = self.re.compile('SOL')

        #self.elements = elements   # list of elements

        self.download = download
        #if self.download:
        #    self.get_nbm()
        self.data = []
        self.column_list = []
        self.trimmed_nbm_file = self.os.path.join(self.scripts_dir,'NBM','nbm_trimmed.txt')


        self.now = self.datetime.utcnow()
        self.current_hour = self.now.replace(minute=0, second=0, microsecond=0)
        self.model_time = self.current_hour - self.timedelta(hours=1)
        self.ymd = self.model_time.strftime('%Y%m%d')
        self.hour = self.model_time.strftime('%H')
        self.url = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod/blend.' + self.ymd + '/' + self.hour + '/text/blend_' + self.bulletin_type + '.t' + self.hour + 'z'

        self.qpf_color = (0.1, 0.9, 0.1, 1)
        self.ra_color = (0, 153/255, 0, 1)
        self.sn_color = (0, 153/255, 204/255, 1.0)
        self.zr_color = (204/255,153/255,204/255, 1.0)
        self.pl_color = (240/255,102/255,0,1.0)
        

        self.prob_yticks = [0, 20, 40, 60, 80, 100]
        self.prob_ytick_labels = ["0","20", "40","60","80","100"]
        self.p_min = -5
        self.p_max = 105

        self.nbm = None
        self.nbm_old = None
        
        if self.bulletin_type == 'nbhtx':
            self.raw_file = 'nbm_raw_hourly.txt'
            self.fhr_zero = self.model_time + self.timedelta(hours=1)
            self.trimmed_nbm_file = self.os.path.join(self.scripts_dir,'NBM','nbm_trimmed_hourly.txt')
            self.element_list = ['TMP','TSD','DPT','DSD','SKY','SSD','WDR','WSP','WSD','GST','GSD','P01',
                                 'P06','Q01','T01','PZR','PSN','PPL','PRA','S01','SLV','I01',
                                 'CIG','MVC','IFC','LIC','LCB','VIS','MVV','IFV','LIV','MHT','TWD','TWS','HID']
        else:
            self.raw_file = 'nbm_raw_short.txt'
            self.fhr_zero = self.model_time + self.timedelta(hours=self.fh_zero_timedelta())
            self.trimmed_nbm_file = self.os.path.join(self.scripts_dir,'NBM','nbm_trimmed_short.txt')

        self.raw_file_path = self.os.path.join(self.scripts_dir,'NBM',self.raw_file)

        if self.download:
            self.get_nbm()
        self.raw_file_path = self.os.path.join('C:/data/scripts/NBM',self.raw_file)
        self.fhr_zero_local = self.fhr_zero - self.timedelta(hours=self.tz_shift)
        self.pTime = self.pd.Timestamp(self.fhr_zero_local)

        if self.bulletin_type == 'nbhtx':
            self.idx = self.pd.date_range(self.pTime, periods=27, freq='H')
        if self.bulletin_type == 'nbstx':
            self.idx = self.pd.date_range(self.pTime, periods=23, freq='3H')
            


    def get_nbm(self):

        self.r = self.requests.get(self.url)
        print('downloading ... ' + str(self.url))
        open(self.raw_file_path, 'wb').write(self.r.content)
        return           

    def fh_zero_timedelta(self):
        if int(self.hour)%3 == 0:
            self.hr_delta = 6
        elif int(self.hour)%3 == 1:
            self.hr_delta = 5            
        else:
            self.hr_delta = 4
        return self.hr_delta    

    def create_df(self):

        with open(self.raw_file_path, 'r') as self.raw:  
            
            with open(self.trimmed_nbm_file, 'w') as self.trimmed:  
                for self.line in self.raw:
                    self.stn.search(self.line)
                    self.sol_match = self.sol.search(self.line)                    
                    if self.stn.search is not None:
                        stn_matched = True
                        if self.sol_match is None:
                            self.start = str(self.line[1:4])
                            self.column_list.append(self.start)
                            self.trimmed.write(self.line)
                        else:
                            break
                        


        if self.bulletin_type == 'nbhtx':
            self.nbm_old = self.pd.read_fwf(self.trimmed_nbm_file, widths=(5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3))
        else:
            self.nbm_old = self.pd.read_fwf(self.trimmed_nbm_file, widths=(5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3))

test2 = NBM('KGRR', 'nbhtx', False)
#test.get_nbm()
test2.create_df()
