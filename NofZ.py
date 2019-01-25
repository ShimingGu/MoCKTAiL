import numpy as np
import scipy.signal as con 
import itertools as itt
#import multiprocessing
from os import system
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc
import h5py
from numba import jit
import sys
'''
sys.path.append('../catalogue_tools')
from np2h5 import Sind
'''

#cores = multiprocessing.cpu_count()
#pool = multiprocessing.Pool(processes=cores)

@jit(nopython = True)
def E2C(edges):
    centres = np.empty(len(edges)-1)
    for i in range(len(centres)):
        centres[i] = (edges[i+1] - edges[i])/2 + edges[i]
    return centres

@jit(nopython = True)
def nofzhist(Ara,Bin):
    H1,E1 = np.histogram(Ara,Bin)
    Cens = E2C(Bin)
    return H1,Cens

def nofzread(pathe):
    Katze = h5py.File(pathe,'r')
    #z = np.array(Katze['z_obs'])
    z = np.array(Katze['z_obs'])
    Beans = np.arange(0,0.6,0.01)
    return nofzhist(z,Beans)

def NofZmain(y):
    Conf = h5py.File('./Config.h5','r')
    x_sep = np.array(Conf['Separation'])[()]
    ZorM = np.array(Conf['ZorM'])[()]
    Abs_App = np.array(Conf['Abs_App'])[()]
    qTar = np.array(Conf['Old_GALFORM'])[()]
    wTar = np.array(Conf['plot_old_galform'])[()]
    if qTar < 0.5:
        TAR0 = 'QuoGal'
    else:
        TAR0 = 'GALFORM'
    Cache = Conf.attrs['catalogue_cache_path'].tostring().decode("utf-8")
    picpath = Conf.attrs['picsave_path'].tostring().decode("utf-8")
    Conf.close()

    x_low = round(y,5)
    x_up = round(x_low + round(x_sep,5),5)
    print (str(x_up)+'\n')

    #if ZorM < 0.5:
        #ZM = 'z_obs'
    if Abs_App > 0.5:
        ZM = 'app_mag'
    else:
        ZM = 'abs_mag'

    nzM,cen = nofzread(Cache+ZM+'/'+'MXXL_'+str(x_low)+'_'+str(x_up)+'.h5')
    lM = len(nzM)
    nzG,cen = nofzread(Cache+ZM+'/'+'GALFORM_'+str(x_low)+'_'+str(x_up)+'.h5')
    lG = len(nzG)
    lA = 1
    if qTar < 0.5:
        nzA,cen = nofzread(Cache+ZM+'/'+'QuoGal_'+str(x_low)+'_'+str(x_up)+'.h5')
        lA = len(nzA)
    if lM*lG*lA == 0:
        print ('Bug occurred at x_low = '+str(x_low)+'\n')

    plt.figure()
    plt.plot(cen,nzM,label='MXXL')
    if wTar > 0.5:
        plt.plot(cen,nzG,label='GALFORM')
    if qTar < 0.5:
        plt.plot(cen,nzA,label='alt_GALFORM')
    plt.legend()
    plt.ylabel('dn/dz')
    plt.xlabel('z')
    plt.title('dn/dz of MXXL and GALFORM from mag range '+str(x_low)+'_to_'+str(x_up))
    plt.savefig(picpath+'NofZ_mag_'+str(x_low)+'_to_'+str(x_up)+'.png',dpi = 340)
    plt.close('all')
    
    return 0


