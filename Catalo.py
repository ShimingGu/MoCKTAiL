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
    print (x_sep)
    ZorM = np.array(Conf['ZorM'])[()]
    Abs_App = np.array(Conf['Abs_App'])[()]
    qTar = np.array(Conf['Old_GALFORM'])[()]
    if qTar < 0.5:
        TAR = 'QuoGal'
        GeT = 'Alt_GALFORM_path'
        Lbl = ''
    else:
        TAR = 'GALFORM'
        GeT = 'GALFORM_path'
        Lbl = ''
    MXp = Conf.attrs['MXXL_path'].tostring().decode("utf-8")
    GAp = Conf.attrs[str(GeT)].tostring().decode("utf-8") 
    Cache = Conf.attrs['catalogue_cache_path'].tostring().decode("utf-8") 
    Conf.close()

    x_low = round(y,5)
    x_up = round(x_low + round(x_sep,5),5)
    print (x_low,x_up,x_sep)

    for EU in [0,1]:#[MXp,GAp]:
        if EU < 0.5:
            Taitou = 'MXXL'
            RO = str(MXp)
        elif EU > 0.5:
            Taitou = str(TAR)
            RO = str(GAp)
        
        if ZorM < 0.5:
            ZM = 'z_obs'
        elif Abs_App > 0.5:
            ZM = ''+'app_mag'
        else:
            ZM = ''+'abs_mag'

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
        Katze.close()
    return 0


def Conca_Pica(Sp,Cache,ZM,xl,xu):
    rusgal0 = str(Cache+ZM+'/'+'AltGal_'+str(xl[0])+'_'+str(xu[0])+'.h5')
    altgal = str(Sp)
    Qata0 = h5py.File(rusgal0,'r')
    Nada = h5py.File(altgal,'w')
    for Yaoshi in Qata0.keys():
        print (str(Yaoshi))
        Cada0 = Qata0[str(Yaoshi)]
        for i in range(1,len(xl)):
            x_l = xl[i];x_u = xu[i]
            print (x_l,x_u)
            Qata = h5py.File(Cache+ZM+'/'+'AltGal_'+str(x_l)+'_'+str(x_u)+'.h5','r')
            Cada = Qata[str(Yaoshi)]
            Cada0 = np.concatenate((Cada0,Cada))
            del Cada;gc.collect()
            Qata.close()
        Nada[str(Yaoshi)] = Cada0
        del Cada0;gc.collect()
    for Ysh in Nada.keys():
        print (str(Ysh))
    Qata0.close()
    Nada.close()
    return 0


def Conca(x):
    Conf = h5py.File('./Config.h5','r')
    x_sep = np.array(Conf['Separation'])[()]
    print (x_sep)
    ZorM = np.array(Conf['ZorM'])[()]
    Abs_App = np.array(Conf['Abs_App'])[()]
    Sp = Conf.attrs['Alt_GALFORM_path'].tostring().decode("utf-8")
    Cache = Conf.attrs['catalogue_cache_path'].tostring().decode("utf-8")
    Conf.close()
    if ZorM < 0.5:
        ZM = 'z_obs'
    elif Abs_App > 0.5:
        ZM = 'app_mag'
    else:
        ZM = 'abs_mag'
    
    x_low = np.round(x,5)
    x_up = np.round(x_low + np.round(x_sep,5),5)
    

    Conca_Pica(Sp,Cache,ZM,x_low,x_up)

    return 0



