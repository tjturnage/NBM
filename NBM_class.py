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
    
    def __init__(self, station, elements, bulletin_type, download=True):

        self.tz_shift = 5
        self.scripts_dir = 'C:/data/scripts'
        self.station = station   # single station
        self.stn = self.re.compile(self.station)
        self.sol = self.re.compile('SOL')

        self.elements = elements   # list of elements
        self.bulletin_type = bulletin_type
        self.download = download
        self.data = []
        self.column_list = []
        self.trimmed_nbm_file = self.os.path.join(self.scripts_dir,'NBM','nbm_trimmed.txt')


        self.now = self.datetime.utcnow()
        self.current_hour = self.now.replace(minute=0, second=0, microsecond=0)
        self.model_time = self.current_hour - self.timedelta(hours=1)
        self.ymd = self.model_time.strftime('%Y%m%d')
        self.hour = self.model_time.strftime('%H')


    
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
            self.fname = 'nbm_raw_hourly.txt'
            self.fhr_zero = self.model_time + self.timedelta(hours=1)
        else:
            self.fname = 'nbm_raw_short.txt'
            self.fhr_zero = self.model_time + self.timedelta(hours=self.fh_zero_timedelta())

        self.raw_file_path = self.os.join(self.scripts_dir,'NBM',self.fname)
        self.fhr_zero_local = self.fhr_zero - self.timedelta(self.tz_shift)
        self.pTime = self.pd.Timestamp(self.fhr_zero_local)

        if self.bulletin_type == 'nbhtx':
            self.idx = self.pd.date_range(self.pTime, periods=27, freq='H')
        if self.bulletin_type == 'nbstx':
            self.idx = self.pd.date_range(self.pTime, periods=23, freq='3H')
            
        self.url = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod/blend.' + self.ymd + '/' + self.hour + '/text/blend_' + self.bulletin_type + '.t' + self.hour + 'z'
        self.dst = self.os.path.join('C:/data/scripts/NBM',self.fname)

    def get_nbm(self):
        if self.download:
            self.r = self.requests.get(self.url)
            print('downloading ... ' + str(self.url))
            open(self.dst, 'wb').write(self.r.content)
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
        if self.bulletin_type == 'nbhtx':
            self.nbm_old = self.pd.read_fwf(self.trimmed_nbm_file, widths=(5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3))
        else:
            self.nbm_old = self.pd.read_fwf(self.trimmed_nbm_file, widths=(5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3))
            
        with open(self.raw_file_path) as fp:  
            for line in fp:
                self.stn_match = self.stn.search(line)
                self.sol_match = self.sol.search(line)
                if self.stn_match is not None:
                    if self.sol_match is None:
                        start = str(line[1:4])
                        self.column_list.append(start)
                        self.dst.write(line)             
                    else:
                        self.dst.close()