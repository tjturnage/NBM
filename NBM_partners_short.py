# -*- coding: utf-8 -*-
"""
Grabs NBM V3.2 houlry bulletins from:
#https://sats.nws.noaa.gov/~downloads/nbm/bulk-textv32/current/

https://para.nomads.ncep.noaa.gov/pub/data/nccf/com/blend/para/blend.20191105/07/text/blend_nbhtx.t07z
    
    Bulletin Key:
        https://vlab.ncep.noaa.gov/web/mdl/nbm-textcard-v4.0

"""

import os
import sys


try:
    os.listdir('/usr')
    scripts_dir = '/data/scripts'
    sys.path.append(os.path.join(scripts_dir,'resources'))
except:
    scripts_dir = 'C:/data/scripts'
    sys.path.append(os.path.join(scripts_dir,'resources'))

from reference_data import set_basic_paths
data_dir,scripts_dir,image_dir = set_basic_paths()
nbm_script_dir = os.path.join(scripts_dir,'NBM')
nbm_image_dir = os.path.join(image_dir,'NBM')

# ensure image directory is created
try:
    os.makedirs(nbm_image_dir)
except:
    pass

import re
import math
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import numpy as np
import matplotlib.pyplot as plt
from my_nbm_functions import u_v_components, temperature_bounds, myround, wind_chill
from my_nbm_functions import dtList_nbm, nbm_station_dict, categorize, GridShader, download_nbm_bulletin
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import itertools, operator
import matplotlib.transforms


bulletin_type = 'nbstx'
download_nbm_bulletin(bulletin_type,download=True)


fin = os.path.join(scripts_dir,'NBM_stationtable_20190819.csv')
if bulletin_type == 'nbhtx':
    trimmed_nbm_file = os.path.join(scripts_dir,'NBM','nbm_trimmed.txt')
else:
    trimmed_nbm_file = os.path.join(scripts_dir,'NBM','nbm_trimmed_short.txt')
#fout = 'C:/data/scripts/NBM/NBM_MI_stations.txt'

station_master = nbm_station_dict()

mi_stations = []
ia_stations = []
for key in station_master:
    if station_master[key]['state'] == 'MI' and '45' not in key :
        mi_stations.append(key)

        
# [q,s,i]01_amount_bar (hourly)
# [q,s,i]01_accum_bar
# abs_p[ra,sn,zr,pl]_[bar,ts]
# 'time_fb_bar','vis_cat_bar','zr_cat_bar'
if bulletin_type != 'nbstx':
    products = ['abs_pra_ts','t_bar','wind','q01_accum_bar','s01_accum_bar','i01_accum_bar']
    fname = 'nbm_raw_hourly.txt'
else:
    products = ['abs_pra_ts','t_bar','wind','q06_accum_bar','s06_accum_bar','i06_accum_bar']
    fname = 'nbm_raw_short.txt'    
    
#fname = 'C:/data/scripts/NBM/20191209/blend_nbhtx.t09z'
#C:\data\scripts\NBM\20191109
map_plot_stations = {}



download = False

stations = ('KMBL','KCAD','KHTL','KLDM','KMEAR','BDWM4','KRQB','EVAM4','LEOM4','KMOP','KIKW','KMKG','KFFX','KENTC',
            'KBELD','KEDMR','KAMN','KBIV','KHUDS','KGRR','KY70','KLAN','KRNP','KFNT','KFENN','BDLM4',
            'KFPK','KTEW','KOZW','KPTK','KLWA','KPAWP','KAZO','KBTL','KRMY','KJXN','KARB','KBEH','KHAI',
            'KIRS','KOEB','KADG')

stations = ('KMBL','KCAD')

nbm_match = re.compile('NBM')
ymdh = re.compile('\d{2}/\d{2}/\d{4}(\s)+\d{2}')
sol = re.compile('SOL')
fhrs = re.compile('FHR')
get_ymdh = True
get_fhr_zero = True
get_idx = True



#for key in mi_stations:
# KPHN, RSCM4, KMBS, KTVC, KDTX, KPTK, KPAWP, KGLR, KARB, KBTL, KRQB, KOEB, KDTW, KEMPR, KFPK, KFNT, KFFX
# KBIV, KHTL, KY70,  KENTC, KMBL, KMQT, KRMY, KMQT, KRMY, KTEW, KIRS
for key in (stations):
    station_id = key
    print(station_id)
    station_description = station_master[key]['name']
    lat = station_master[key]['lat']
    lon = station_master[key]['lon']
    column_list = []
    station_found = False
    utc_shift = station_master[key]['time_shift']
    stn = re.compile(key)

    if download:
        raw_file_path = download_nbm_bulletin(url,fname,True)
        download = False
    else:
        raw_file_path = download_nbm_bulletin(url,fname,False)
        #raw_file_path = 'C:/data/scripts/NBM/20191209/blend_nbstx.t09z'
        
    dst = open(trimmed_nbm_file, 'w')
    with open(raw_file_path) as fp:  
        for line in fp:
            if get_ymdh:
                ymdh_match = ymdh.search(line)
                if ymdh_match is not None:
                    run_ymdh = datetime.strptime(ymdh_match[0], '%m/%d/%Y  %H')
                    #idx,model_run_local = dtList_nbm(run_ymdh,'nbstx',utc_shift)
                    get_ymdh = False
                    
            if get_fhr_zero:
                fhrs_match = fhrs.search(line)
                if fhrs_match is not None:
                    fhr_zero_shift = int(line.split(' ')[2:3])
                    get_fhr_zero = False                

            if get_ymdh == False and get_fhr_zero == False and get_idx == True:
                fhr_zero = run_ymdh + timedelta(hours=fhr_zero_shift)
                idx,model_run_local = dtList_nbm(fhr_zero,'nbstx',utc_shift)
                start_time = idx[0]
                end_time = idx[-1]
                get_idx = False

            stn_match = stn.search(line)
            sol_match = sol.search(line)

            if (stn_match is not None):
                if sol is None:
                    start = str(line[1:4])
                    column_list.append(start)
                    dst.write(line)
                else:
                    dst.close()
                    break 

    nbm_old = None
    nbm = None
    
    #nbm_old = pd.read_fwf(trimmed_nbm_file)
    nbm_old = pd.read_fwf(trimmed_nbm_file, widths=(5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3))
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

    #nbm.set_index(data_list, inplace=True)

    # Now will start building a dictionary of products to plot
    # some products are only in a certain bulletin
    #   e.g., "nbm.X01" is hourly only, nbm.X06 is 6hourly and not in hourly product
    prods = {}
    
    # To plot time series lines  -- slice the dataframe with 'loc'
    # To plot a bar graph        -- convert the slice to a list.
    # Either can be done independently, but I usually do both to have the later 
    # option of plotting either way.
    
    t_list = nbm.TMP.tolist()
    t_list = np.asarray(t_list, dtype=np.float32)
    t_shifted_list = t_list + 40
    dp_list = nbm.DPT.tolist()
    wdir_list = nbm.WDR.tolist()

    wdir = nbm.loc[:, ['WDR']]
    wspd = nbm.loc[:, ['WSP']]
    wspd_list_kt = nbm.WSP.tolist()
    wspd_list = np.multiply(wspd_list_kt,1.151)
    wspd_list = [round(x) for x in wspd_list]
    wspd_arr = np.asarray(wspd_list)
    wspd_list = wspd_arr.astype(int)

    wgst_list_kt = nbm.GST.tolist()
    wgst_list = np.multiply(wgst_list_kt,1.151)
    wgst_list = [round(x) for x in wgst_list]
    wgst_arr = np.asarray(wgst_list)
    wgst_list = wgst_arr.astype(int)

    sky_list = nbm.SKY.tolist()

    wind_chill_list = []
    time_to_fb_list = []
    for chill in range(0,len(wspd_list)):
        wc = wind_chill(t_list[chill],wspd_list[chill])
        #time_to_fb = time_to_frostbite(wc)
        wind_chill_list.append(wc)
        #time_to_fb_list.append(time_to_fb)
    wind_chill_list = np.asarray(wind_chill_list, dtype=np.float32)
    wc_cat = categorize(wind_chill_list,'wc')
    wind_chill_shifted_list = wind_chill_list + 40
    map_plot_stations[key] = {'lon':lon,'lat':lon,'wc_cat':wc_cat[0]}
    # Temp (t) and wind chill (wc) go on same panel, 
    # so using min(wc) and max(t) to define bounds for 'twc'
    # using a temperature_bounds function
    twc_tick_list,twc_tick_labels = temperature_bounds(t_shifted_list,wind_chill_shifted_list)

    # sometimes we have to convert units because they're in tenths or hundredths
    nbm.VIS = nbm.VIS.multiply(0.1)
    vis = nbm.loc[:, ['VIS']]
    vis.clip(upper=7.0,inplace=True)
    vis_list = nbm.VIS.tolist()
 
    
    # hourly precip amount, convert integers to hundredths of an inch
    nbm.Q01 = nbm.Q01.multiply(0.01)
    q01_amount_list = nbm.Q01.tolist()
    q01_accum_list = list(itertools.accumulate(q01_amount_list, operator.add))


    # 6 hourly precip amount, convert integers to hundredths of an inch
    nbm.Q06 = nbm.Q06.multiply(0.01)
    q06_amount_list = nbm.Q06.tolist()
    q06_accum_list = list(itertools.accumulate(q06_amount_list, operator.add))

    # hourly snow amount, convert integers to tenths of an inch
    nbm.S01 = nbm.S01.multiply(0.1)
    s01_amount_list = nbm.S01.tolist()
    s01_accum_list = list(itertools.accumulate(s01_amount_list, operator.add))
    s01_accum_max = np.max(s01_accum_list)
    sn_accum_ticks = [0,1,2,4,6,8,10,12,14]

    # 6 hourly snow amount, convert integers to tenths of an inch
    nbm.S06 = nbm.S06.multiply(0.1)
    s06_amount_list = nbm.S06.tolist()
    s06_accum_list = list(itertools.accumulate(s06_amount_list, operator.add))
    s06_accum_max = np.max(s06_accum_list)
    sn_accum_ticks = [0,1,2,4,6,8,10,12,14]

    s01_accum_ticks, s01_accum_tick_labels = myround(s01_accum_max)
    s06_accum_ticks, s06_accum_tick_labels = myround(s06_accum_max)
    
    nbm.I01 = nbm.I01.multiply(0.01)
    i01_amount_list = nbm.I01.tolist()
    i01_accum_list = list(itertools.accumulate(i01_amount_list, operator.add))

    nbm.I06 = nbm.I06.multiply(0.01)
    i06_amount_list = nbm.I06.tolist()
    i06_accum_list = list(itertools.accumulate(i06_amount_list, operator.add))

    # here's where we create categories for snow and ice accumulations
    # by caling the categorize function    

    sn_cat_list = categorize(s01_amount_list,'sn')
    zr_cat_list = categorize(s01_amount_list,'zr')
    vis_cat_list = categorize(vis_list,'vis')

    #  P01 is one hour Probability for any type of precip, aka PPI
    pop1_list = nbm.P01.tolist()
    pop1_ts = nbm['P01'] 
    #multiply P01 by ProbRain to get absolute rain probability
    #multiply P01 to ProbSnow to get absolute snow probability
    nbm['ABS_PROB_RA'] = nbm['P01'] * nbm['PRA']/100
    nbm['ABS_PROB_SN'] = nbm['P01'] * nbm['PSN']/100
    nbm['ABS_PROB_ZR'] = nbm['P01'] * nbm['PZR']/100
    nbm['ABS_PROB_PL'] = nbm['P01'] * nbm['PPL']/100

    #  P06 is one hour Probability for any type of precip, aka PPI
    pop6_list = nbm.P06.tolist()
    pop6_ts = nbm['P06'] 
    #multiply P01 by ProbRain to get absolute rain probability
    #multiply P01 to ProbSnow to get absolute snow probability
    nbm['ABS_PROB_RA'] = nbm['P06'] * nbm['PRA']/100
    nbm['ABS_PROB_SN'] = nbm['P06'] * nbm['PSN']/100
    nbm['ABS_PROB_ZR'] = nbm['P06'] * nbm['PZR']/100
    nbm['ABS_PROB_PL'] = nbm['P06'] * nbm['PPL']/100



    abs_prob_ra = nbm['ABS_PROB_RA']
    abs_prob_sn = nbm['ABS_PROB_SN']
    abs_prob_zr = nbm['ABS_PROB_ZR']
    abs_prob_pl = nbm['ABS_PROB_PL']    
    abs_prob_ra_list = nbm.ABS_PROB_RA.to_list()
    abs_prob_sn_list = nbm.ABS_PROB_SN.to_list()
    abs_prob_zr_list = nbm.ABS_PROB_ZR.to_list()
    abs_prob_pl_list = nbm.ABS_PROB_PL.to_list()



    ### -----------------------------  Uncomment Below for Synthetic Data ---------------------------
    #Begin Synthetic Data
    """
    t_list = np.arange(20,-5,-1)
    t_shifted_list = t_list + 40

    wspd_list_temp = np.arange(17,29.5,0.5)
    wspd_arr = np.rint(wspd_list_temp)
    wspd_list = wspd_arr.astype(int)
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
        wc = wind_chill(t_list[chill],wspd_list[chill])
        time_to_fb = time_to_frostbite(wc)
        wind_chill_list.append(wc)
        time_to_fb_list.append(time_to_fb)
    wind_chill_list = np.asarray(wind_chill_list, dtype=np.float32)
    wc_cat = categorize(wind_chill_list,'wc')
    wind_chill_shifted_list = wind_chill_list + 40
    map_plot_stations[key] = {'lon':lon,'lat':lon,'wc_cat':wc_cat[0]}
    # Temp (t) and wind chill (wc) go on same panel, 
    # so using min(wc) and max(t) to define bounds for 'twc'
    # using a temperature_bounds function
    twc_tick_list,twc_tick_labels = temperature_bounds(t_shifted_list,wind_chill_shifted_list)


    """
    ### -----------------------------  Uncomment Above for Synthetic Data ---------------------------



    qpf_color = (0.1, 0.9, 0.1, 1)
    ra_color = (0, 153/255, 0, 1)
    sn_color = (0, 153/255, 204/255, 1.0)
    zr_color = (204/255,153/255,204/255, 1.0)
    pl_color = (240/255,102/255,0,1.0)
    # conditional probabilities for rain (PRA), snow (PSN), freezing rain (PZR), sleet (PPL)
    # define y axis range and yticks/ylabels for any element that's probabilistic 
    prob_yticks = [0, 20, 40, 60, 80, 100]
    prob_ytick_labels = ["0","20", "40","60","80","100"]
    p_min = -5
    p_max = 105

    prods['pop1_bar'] = {'data': pop1_list, 'color':(0.5, 0.5, 0.5, 1), 'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Precip\nChances\n(%)'}
    prods['pop1_ts'] = {'data': pop1_ts, 'color':(0.7, 0.7, 0.7, 1), 'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Precip\nChances\n(%)'}
    prods['pop6_bar'] = {'data': pop6_list, 'color':(0.5, 0.5, 0.5, 1), 'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Precip\nChances\n(%)'}
    prods['pop6_ts'] = {'data': pop6_ts, 'color':(0.7, 0.7, 0.7, 1), 'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Precip\nChances\n(%)'}

    #-------------- Rain
        
    prods['abs_pra_bar'] = {'data': abs_prob_ra_list, 'color':ra_color,
         'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Prob Rain\n(%)'}


    prods['abs_pra_ts'] = {'data': abs_prob_ra, 'color':ra_color,
         'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Prob Rain\n(%)'}
 
    prods['q01_amount_bar'] = {'data': q01_amount_list, 'color':qpf_color,
         'ymin':0.0,'ymax':0.50,
         'yticks':[0.0,0.1,0.2,0.3], 'ytick_labels':['0','0.1','0.2','0.3'],
         'title':'Precip\nAmount\n(inches)'}    

    prods['q06_amount_bar'] = {'data': q06_amount_list, 'color':qpf_color,
         'ymin':0.0,'ymax':3.00,
         'yticks':[0.0,0.5,1.0,2.0], 'ytick_labels':['0','0.5','1.0','2.0'],
         'title':'Precip\nAmount\n(inches)'}  

    prods['q01_accum_bar'] = {'data': q01_accum_list, 'color':qpf_color,
         'ymin':0,'ymax':4.01,'bottom':0,
         'yticks':[0,0.5,1,1.5,2,3],'ytick_labels':['0','0.5','1.0','1.5','2.0','3.0'],    
         'title':'Rain\nAccum\n(in)' }

    prods['q06_accum_bar'] = {'data': q06_accum_list, 'color':qpf_color,
         'ymin':0,'ymax':8.01,'bottom':0,
         'yticks':[0,1,2,3,4,5,6,7],'ytick_labels':['0','1','2','3','4','5','6','7'],    
         'title':'Rain\nAccum\n(in)' }


    #-------------- Snow
    
    prods['abs_psn_bar'] = {'data': abs_prob_sn_list, 'color':sn_color,
         'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels,
         'title':'Probability\nSnow(%)'}

    prods['abs_psn_ts'] = {'data': abs_prob_sn, 'color':sn_color,
         'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Probability\nSnow(%)'}

    prods['s01_amount_bar'] = {'data': s01_amount_list, 'color':sn_color,
         'ymin':0.0,'ymax':1.01,'yticks':[0,0.25,0.5,0.75,1],
         'ytick_labels':['0','1/4','1/2','3/4','1'], 'title':'Hourly\nSnow' }

    prods['s06_amount_bar'] = {'data': s06_amount_list, 'color':sn_color,
         'ymin':0.0,'ymax':12.01,'yticks':[0,1,2,3,4,6,8,12],
         'ytick_labels':['0','1','2','3','4','6','8','12'], 'title':'6 Hour\nSnow' }

    prods['s01_accum_bar'] = {'data': s01_accum_list, 'color':sn_color,
         'ymin':0,'ymax':s01_accum_ticks[-1],'bottom':0,
         'yticks':s01_accum_ticks,'ytick_labels':s01_accum_tick_labels,    
         'title':'Snow\nAccum\n(in)' }

    prods['s06_accum_bar'] = {'data': s06_accum_list, 'color':sn_color,
         'ymin':0,'ymax':s06_accum_ticks[-1],'bottom':0,
         'yticks':s06_accum_ticks,'ytick_labels':s06_accum_tick_labels,    
         'title':'Snow\nAccum\n(in)' }


    prods['sn_cat_bar'] = {'data': sn_cat_list, 'color':sn_color,
         'ymin':0,'ymax':7,'bottom':0,
         'yticks':[0,1,2,3,4,5,6,7],'ytick_labels':['0.0','0.1','0.2','0.5','0.8','1.0','1.5','2.0'],    
         'title':'Snow\nAmount\n(in)' }

    #-------------- Freezing Rain

    prods['abs_pzr_bar'] = {'data': abs_prob_zr_list, 'color':zr_color,
         'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels,
         'title':'Probability\nIce(%)'}

    prods['abs_pzr_ts'] = {'data': abs_prob_zr, 'color':zr_color,
         'ymin':p_min,'ymax':p_max,'yticks':prob_yticks,'ytick_labels':prob_ytick_labels, 'title':'Probability\nSnow(%)'}

    prods['i01_amount_bar'] = {'data': i01_amount_list, 'color':zr_color,
         'ymin':0.0,'ymax':0.21,'yticks':[0.05, 0.10, 0.15, 0.20],
         'ytick_labels':['0','.05','.10','.15',',20'], 'title':'Hourly\nIce' }

    prods['i06_amount_bar'] = {'data': i06_amount_list, 'color':zr_color,
         'ymin':0.0,'ymax':0.21,'yticks':[0.05, 0.10, 0.15, 0.20],
         'ytick_labels':['0','.05','.10','.15',',20'], 'title':'Hourly\nIce' }

    prods['i01_accum_bar'] = {'data': i01_accum_list, 'color':zr_color,
         'ymin':0,'ymax':1.01,'bottom':0,
         'yticks':[0,0.1,0.25,0.5,0.75,1],'ytick_labels':['0','0.1','0.25','0.5','0.75','1'],    
         'title':'Ice\nAccum\n(in)' }

    prods['i06_accum_bar'] = {'data': i06_accum_list, 'color':zr_color,
         'ymin':0,'ymax':1.01,'bottom':0,
         'yticks':[0,0.1,0.25,0.5,0.75,1],'ytick_labels':['0','0.1','0.25','0.5','0.75','1'],    
         'title':'Ice\nAccum\n(in)' }

    prods['zr_cat_bar'] = {'data': zr_cat_list, 'color':zr_color,
         'ymin':0,'ymax':5,'bottom':0,
         'major_yticks':[0,1,2,3,4,5],'major_yticks_labels':['0.01','0.03','0.05','0.10','0.25','0.5'],    
         'title':'Ice\nAccum\n(in)' }



    prods['wind'] = {'data': wspd, 'color':(0.5, 0.5, 0.5, 0.8),'ymin':0,'ymax':1,'yticks':[0.27,0.37,0.62],
         'ytick_labels':['Gust (mph)','Speed (mph)','Wind\nDirection'],'title':''}# Wind Speed\nand Gusts\n(mph)'}

    prods['t_bar'] = {'data': t_shifted_list, 'color':(255/255, 0, 0, 1.0),
         'ymin':0,'ymax':80,'bottom':0,
         'yticks':[30,45,62,75],'ytick_labels':['0','15','32','45'],
         'minor_yticks':[60],'minor_yticks_labels':['30']}#,
         #'title':'Temperature\nWind Chill\n(F)' }

    prods['wc_bar'] = {'data': wind_chill_shifted_list, 'color':(0, 0, 204/255, 1.0),
         'ymin':0,'ymax':75,'bottom':0,
         'yticks':[30,45,62,75],'ytick_labels':['0','15','32','45'],
         'minor_yticks':[60],'minor_yticks_labels':['30'],
         'title':'Temperature\n(red)\n\nWind Chill\n(blue)'}

    prods['time_fb_bar'] = {'data': time_to_fb_list, 'color':(0.9, 0.9, 0.2, 1.0),
         'ymin':0.5,'ymax':4.5,'yticks':[1,2,3,4],'ytick_labels':['under 5','5','15-30','30+'],
         'title':'Time to\nFrostbite\n(min)'}

    prods['sky_bar'] = {'data': sky_list, 'color':(0.6, 0.6, 0.6, 1.0),
         'ymin':-5,'ymax':105,'yticks':[0,25,50,75,100],
         'ytick_labels':['0','25','50','75','100'], 'title':'Sky cover\n(%)'}

    prods['vis_cat_bar'] = {'data': vis_cat_list, 'color':(150/255,150/255,245/255, 1.0),
         'ymin':0,'ymax':6,'bottom':0,
         'yticks':[0,1,2,3,4,5,6],'ytick_labels':['0.00','0.25','0.50','1.00','2.00','3.00',' > 6'],    
         'title':'Visibility\n(miles)' }


    ### -----------------------------  Begin Plotting ---------------------------

    hours = mdates.HourLocator()
    myFmt = DateFormatter("%d%h")
    myFmt = DateFormatter("%d%b\n%HZ")
    myFmt = DateFormatter("%I\n%p")
    myFmt = DateFormatter("%I")    
    
    grid_alpha = 0.0    #0.3
    first_gray = True
    bar_align = "center"   # "edge"
    bar_width = 1/35
    fig_set = {'4':(14,13),'5':(14,15),'6':(12,18)}
    fig, axes = plt.subplots(len(products),1,figsize=fig_set[str(len(products))],sharex=False,subplot_kw={'xlim': (start_time,end_time)})

    #fig, axes = plt.subplots(len(products),1,figsize=(16,8),sharex=True,subplot_kw={'xlim': (start_time,end_time)})
    plt.subplots_adjust(bottom=0.1, left=0.17, top=0.9)


#    plt.suptitle('Hourly Forecast -- ' + model_run_local.strftime('%B %d, %Y') + ' ... Updated ' + model_run_local.strftime('%I %p EST') + '\n' + station_description  )
    plt.suptitle('Hourly Forecast -- ' + model_run_local.strftime('%B %d, %Y') + ' ... Updated ' + model_run_local.strftime('%I %p EST') + '\n' + station_description  )
    

    for y,a in zip(products,axes.ravel()):

        #plt.setp( ax.xaxis.get_majorticklabels(), rotation=-45) 

        # Create offset transform by 5 points in x direction
        #dx = 7/72.; dy = 0/72. 
        dx = 0/72.; dy = 0/72. 
        offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

        # apply offset transform to all x ticklabels.

        for label in a.xaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + offset)


        a.xaxis.set_major_locator(hours)
        a.xaxis.set_major_formatter(myFmt)

        plt.setp( a.xaxis.get_majorticklabels(), ha="center",rotation=0 )
        
        a.yaxis.set_label_coords(-0.112,0.25)
        a.xaxis.grid(True, linewidth=20, color=(0.96,0.96,0.96), zorder=1)  
        #a.xaxis.grid(True, linewidth=20, alpha = 0.12, zorder=1)  


        if y == 'abs_pra_ts':
            gs = GridShader(a, facecolor="lightgrey", first=first_gray, alpha=grid_alpha) 
            a.set_yticks(prods[y]['yticks'], minor=False)
            a.set_yticklabels(prods[y]['ytick_labels'],minor=False)
            a.grid(which='major', axis='y')
            a.yaxis.set_label_coords(-0.112,0.1)
            a.get_xaxis().set_visible(True)
            a.set_xticks(data_list)
            a.set_ylim(prods[y]['ymin'],prods[y]['ymax'])
            this_title = 'Probability of:\n\nAny Precipitation\n(gray dash)\n\nSnow (blue)\n\nIce (purple)'
            a.set_ylabel(this_title, rotation=0)

            a.plot(prods['abs_pra_ts']['data'],linewidth=2, zorder=10,color=prods['abs_pra_ts']['color'])
            a.plot(prods['abs_psn_ts']['data'],linewidth=2, zorder=10,color=prods['abs_psn_ts']['color'])
            a.plot(prods['abs_pzr_ts']['data'],linewidth=2, zorder=10,color=prods['abs_pzr_ts']['color'])
            a.plot(prods['pop1_ts']['data'],linewidth=6,linestyle=':', zorder=9,color=prods['pop1_ts']['color'])



        if y == 'wind':
            plt.rc('font', size=12) 
            a.set_ylim(0,1)

            dx = 0/72.; dy = 0/72. 
            offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

            # apply offset transform to all x ticklabels.
            a.set_yticks(prods[y]['yticks'], minor=False)
            a.set_yticklabels(prods[y]['ytick_labels'],minor=False)
            a.get_yaxis().set_visible(True)
            this_title = ''
            a.set_ylabel(this_title, rotation=0)
            a.set_xticks(data_list)
            a.get_xaxis().set_visible(True)

            #gs = GridShader(a, facecolor="lightgrey", first=first_gray, alpha=0.5)
            a.set_xticks(data_list)    
            for s,d,g,p in zip(wspd_list,wdir_list,wgst_list,data_list):
                #print(d,s)
                u,v = u_v_components(d*10,s)
                u_norm = u / np.sqrt(u**2 + v**2)
                v_norm = v / np.sqrt(u**2 + v**2)
                a.quiver(p, 0.6, u_norm, v_norm, scale=30, width=0.004,color=[0,0,1,0.9], zorder=10,pivot='middle')

                #a.quiverkey(q, X=0.0, Y=0.0, U=10,zorder=10)
                #a.barbs(p, 0.7, u, v, length=7, color=[0,0,1,0.9], pivot='middle')
                a.text(p, 0.35, f'{s}',horizontalalignment='center',color=[0,0,1,0.9])
                a.text(p, 0.25, f'{g}',horizontalalignment='center',color=[0,0,102/255,1])


        # specialized treatment for ranges and gridlines
        if y in ['s01_accum_bar','i01_accum_bar','q01_accum_bar','s01_amount_bar','i01_amount_bar','q01_amount_bar']:
            gs = GridShader(a, facecolor="lightgrey", first=first_gray, alpha=grid_alpha) 
            a.set_yticks(prods[y]['yticks'], minor=False)
            a.set_yticklabels(prods[y]['ytick_labels'],minor=False)
            a.grid(which='major', axis='y')
            a.set_xticks(data_list)
            a.set_ylim(prods[y]['ymin'],prods[y]['ymax'])

            a.bar(data_list,prods[y]['data'],width=bar_width, zorder=10,align=bar_align,bottom=prods[y]['bottom'],color=prods[y]['color'])

            a.set_ylabel(prods[y]['title'], rotation=0)
            a.get_xaxis().set_visible(True)

        if y in ['s06_accum_bar','i06_accum_bar','q06_accum_bar','s06_amount_bar','i06_amount_bar','q06_amount_bar']:
            gs = GridShader(a, facecolor="lightgrey", first=first_gray, alpha=grid_alpha) 
            a.set_yticks(prods[y]['yticks'], minor=False)
            a.set_yticklabels(prods[y]['ytick_labels'],minor=False)
            a.grid(which='major', axis='y')
            a.set_xticks(data_list)
            a.set_ylim(prods[y]['ymin'],prods[y]['ymax'])

            a.bar(data_list,prods[y]['data'],width=bar_width, zorder=10,align=bar_align,bottom=prods[y]['bottom'],color=prods[y]['color'])

            a.set_ylabel(prods[y]['title'], rotation=0)
            a.get_xaxis().set_visible(True)


        if y in ['t_bar','wc_bar']:
            gs = GridShader(a, facecolor="lightgrey", first=first_gray, alpha=grid_alpha) 
            a.set_yticks(twc_tick_list, minor=False)
            a.set_yticklabels(twc_tick_labels,minor=False)
            a.grid(which='major', axis='y')
            a.set_xticks(data_list)
            a.set_ylim(twc_tick_list[0],twc_tick_list[-1])
            a.bar(data_list,prods['t_bar']['data'],width=bar_width, zorder=10,align=bar_align,bottom=prods[y]['bottom'],color=prods['t_bar']['color'])
            a.bar(data_list,prods['wc_bar']['data'],width=bar_width, zorder=10,align=bar_align,bottom=prods[y]['bottom'],color=prods['wc_bar']['color'])
            a.set_ylabel(prods['wc_bar']['title'], rotation=0)
            a.get_xaxis().set_visible(True)
    
        # these are lists that use matplotlib bar to create bar graphs
        if y in ['abs_pra_bar','abs_pzr_bar','abs_psn_bar','abs_psn_bar']:
            gs = GridShader(a, facecolor="lightgrey", first=first_gray, alpha=grid_alpha) 
            a.set_xticks(data_list)
            a.set_ylim(prods[y]['ymin'],prods[y]['ymax'])
            #a.bar(data_list,prods[y]['data'],width=1/25, align="edge",color=prods[y]['color'])
            a.bar(data_list,prods[y]['data'],width=bar_width, align="center",color=prods[y]['color'])
            a.set_ylabel(prods[y]['title'], rotation=0)
            a.get_xaxis().set_visible(True)
            a.set(yticks = prods[y]['yticks'], yticklabels = prods[y]['ytick_labels'])
            
        if y in ['time_fb_bar','vis_cat_bar','zr_cat_bar']:
            gs = GridShader(a, facecolor="lightgrey", first=first_gray, alpha=grid_alpha) 
            a.grid(which='major', axis='y')
            a.set_xticks(data_list)
            a.set_ylim(prods[y]['ymin'],prods[y]['ymax'])
            a.bar(data_list,prods[y]['data'],width=bar_width,  zorder=10,align=bar_align,color=prods[y]['color'])

            a.set_ylabel(prods[y]['title'], rotation=0)
            a.get_xaxis().set_visible(True)
            a.set(yticks = prods[y]['yticks'], yticklabels = prods[y]['ytick_labels'])
    
    image_file = key + '_NBM_' + bulletin_type + '.png'
    image_dst_path = os.path.join(nbm_image_dir,image_file)
    #plt.show()
    plt.savefig(image_dst_path,format='png')
    plt.close()

