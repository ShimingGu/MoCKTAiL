import numpy as np
import scipy.signal as con 
import itertools as itt
from os import system
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc
import h5py
#from numba import jit,njit,vectorize,int32,float64
import sys
'''
sys.path.append('../catalogue_tools')
from np2h5 import Sind
'''

def Neicun(msp,Mao,Kato,bds):
    if msp == 0:
        lab = 'ra'
    elif msp == 1:
        lab = 'dec'
    elif msp == 2:
        lab = 'z_obs'
    elif msp == 3:
        lab = 'z_cos'
    elif msp == 4:
        lab = 'app_mag'
    elif msp == 5:
        lab = 'abs_mag'
    elif msp == 6:
        lab = 'color'
    elif msp == 7:
        lab = 'gal_type'
    elif msp == 8:
        lab = 'halo_mass'
    elif msp == 9:
        lab = 'random_seed'
    sq = np.array(Mao[str(lab)]);sq = np.array(sq[bds])
    Kato[str(lab)] = sq
    return 0

def Separa(y):
    Conf = h5py.File('./Config.h5','r')
    x_sep = np.array(Conf['Separation'])[()]
    ZorM = np.array(Conf['ZorM'])[()]
    Abs_App = np.array(Conf['Abs_App'])[()]
    qTar = np.array(Conf['Old_GALFORM'])[()]
    if qTar < 0.5:
        TAR = 'RusGal'
        GeT = 'Alt_GALFORM_path'
    else:
        TAR = 'GALFORM'
        GeT = 'GALFORM_path'
    MXp = Conf.attrs['MXXL_path'].tostring().decode("utf-8")
    GAp = Conf.attrs[str(GeT)].tostring().decode("utf-8") 
    Cache = Conf.attrs['catalogue_cache_path'].tostring().decode("utf-8") 
    Conf.close()

    x_low = round(y,5)
    x_up = x_low + round(x_sep,5)
    print (x_up)

    for EU in [MXp,GAp]:
        RO = str(EU)
        if EU == MXp:
            Taitou = 'MXXL'
        else:
            Taitou = TAR
        
        if ZorM < 0.5:
            ZM = 'z_obs'
        elif Abs_App > 0.5:
            ZM = 'app_mag'
        else:
            ZM = 'abs_mag'

        Cat = h5py.File(RO,'r')
        jud = np.array(Cat[str(ZM)])
        lb = jud>=x_low
        ub = jud<x_up
        del jud;gc.collect()
        bds = np.logical_and(lb,ub)
        del lb,ub;gc.collect()
        system('mkdir '+Cache+ZM+'/')
        Katze = h5py.File(Cache+ZM+'/'+str(Taitou)+'_'+str(x_low)+'_'+str(x_up)+'.h5','w')
        for i in [0,1,2,3,4,5,6,7,8,9]:
            Neicun(i,Cat,Katze,bds)
        del bds;gc.collect()
        return 0

def Conca(x):
    return 0

