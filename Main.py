#!/usr/bin/python -u
import numpy as np
import scipy.signal as con
import multiprocessing
import os
import h5py
import cython
import gc
import sys

#
# Default Parameter Settings
#

MXXL_path = b"/gpfs/data/ddsm49/GALFORM/catalogues/MXXL.h5"
GALFORM_path = b"/gpfs/data/ddsm49/GALFORM/catalogues/GALF.h5"
Alt_GALFORM_path = b"/gpfs/data/ddsm49/GALFORM/catalogues/Altg.h5"
picsave_path = b"./pics"
catalogue_cache_path = b"/gpfs/data/ddsm49/GALFORM/Cache2/"

# PLEASE ADD THE PATH AFTER THE LETTER b

Mag_min = 9.5
Mag_max = 19.5
Mag_Sep = 0.5 

Abs_App = 'App'

Z_min = 0
Z_max = 0.50
Z_Sep = 0.005

#
# AUTOMATIC MODES (NOT YET WRITTEN)
#

Mode = 0

# 0 = NO AUTOMATIC MODES
# 1 = CHECK THE NEW REDSHIFT DISTRIBUTION DIRECTLY FROM OLD GALFORM CATALOGUE
# 2 = CHECK THE NEW COLOUR DISTRIBUTION DIRECTLY FROM THE OLD GALFORM CATALOGUE
# 4 = CHECK THE NEW ANGULAR CORRELATION FUNCTION DIRECTLY FROM THE OLD GALFORM CATALOGUE (STILL NOT IMPLEMENTED)
# 12 = 1 + 2
# 14 = 1 + 4
# 24 = 2 + 4
# 124 = 1 + 2 + 4

#
# MANUAL CATALOGUE SETTINGS
#

Use_Auld_GALFORM_Catalogue = 1
## Mainly a useless parameter, will be totally ignored in the auto-case

Catalogue_Separation = 2
# 0 = "REUSE", 1 = "REDO", 2 = "DELETE AND REDO"

Fenli = 'Magnitude'

#
# PLOT SETTINGS
#

Cumulative_N_OF_Z = 0

Interpolate_LF = 0
Plot_LF = 1
LF_Interpolation = '1-D Interpolation'
Mag_limit_for_LF = 19.5
k_correction = 1
Fractions = 1

Concatenate_Catalogues = 1

Color_Distribution = 0
PLOT_OLD_GALFORM = 1

###################################################################
######################### I AM A LINE #############################
###################################################################

cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=cores)

os.system('mkdir '+str(catalogue_cache_path.decode("utf-8")))
os.system('mkdir '+str(picsave_path.decode("utf-8")))

Mags = np.arange(Mag_min,Mag_max,Mag_Sep)
Zs = np.arange(Z_min,Z_max,Z_Sep)
MS = np.array([Mag_Sep])
ZD = np.array([Z_Sep])
Mags = np.around(10000*Mags);Mags = Mags.astype('int');Mags = Mags/10000.0
Zs = np.around(10000*Zs);Zs = Zs.astype('int');Zs = Zs/10000.0

###################################################################
######################### I AM A LINE #############################
###################################################################

if Abs_App == 'Abs':
    Abs_App = 0
if Abs_App == 'Absolute':
    Abs_App = 0
if Abs_App == 'App':
    Abs_App = 1
if Abs_App == 'Apparent':
    Abs_App = 1

if Fenli == 'Redshift':
    Fenli = 0
if Fenli == 'redshift':
    Fenli = 0
if Fenli == 'Magnitude':
    Fenli = 1
if Fenli == 'magnitude':
    Fenli = 1

if Interpolate_LF == 1:
    Fenli = 0
    print("\n Interpolating Luminosity Function, Forced to the Mode of Redshift Separation \n")

if LF_Interpolation == '1-D Interpolation' or '1-D interpolation' or '1DI':
    LFI = 1
if LF_Interpolation == '2-D Interpolation' or '2-D interpolation' or '2DI':
    LFI = 2
if LF_Interpolation == '1-D Semi-Interpolation' or '1-D semi-interpolation' or '1DSI':
    LFI = 1.5
if LF_Interpolation == '2-D Semi-Interpolation' or '2-D semi-interpolation' or '2DSI':
    LFI = 2.5
if LF_Interpolation == '1-D Regression' or '1-D Regression' or '1DR':
    LFI = 0
if LF_Interpolation == '2-D Regression' or '2-D Regression' or '2DR':
    LFI = 0.5

###################################################################
######################### I AM A LINE #############################
###################################################################

#
# Code
#

if Fenli < 0.5:
    Gua = 'z_obs'
elif Abs_App < 0.5:
    Gua = 'abs_mag'
else:
    Gua = 'app_mag'

print (Gua,Fenli)

if Catalogue_Separation > 0.5:
    if Catalogue_Separation > 1.5:
        os.system('rm /gpfs/data/ddsm49/GALFORM/Cache2/'+Gua+'/*.h5')


if Fenli > 0.5:
    Har = Mags
    FEN = Mag_Sep
    print ('Mags')
else:
    Har = Zs
    FEN = Z_Sep
    print ('Zs')

Conf = h5py.File('Config.h5','w')
Conf['Separation'] = FEN
Conf['ZorM'] = Fenli
Conf['Old_GALFORM'] = Use_Auld_GALFORM_Catalogue
Conf['Abs_App'] = Abs_App
Conf['LF_Mag_limit'] = Mag_limit_for_LF
Conf['LF_Interpolation_Technique'] = LFI
Conf['k_corr'] = k_correction
Conf.attrs['MXXL_path'] = np.void(MXXL_path)
Conf.attrs['GALFORM_path'] = np.void(GALFORM_path)
Conf.attrs['Alt_GALFORM_path'] = np.void(Alt_GALFORM_path)
Conf.attrs['picsave_path'] = np.void(picsave_path)
Conf.attrs['catalogue_cache_path'] = np.void(catalogue_cache_path)
Conf['plot_old_galform'] = PLOT_OLD_GALFORM
Conf['FRAC'] = Fractions
Conf.close()

sys.path.append('./src')
from Catalo import Separa,Conca
from NofZ import NofZmain
from Lumino import LFmain
#from Colodis import CDmain
#from FUNCS import EucProd,Sep,nofzMaine,LfMaine,Colodis

def Func0(Har):
    for i in pool.map(Separa,Har):
        p = 0
    return 0

def Func1(Har):
    if Fenli > 0.5 and Plot_N_of_Z > 0:
        for i in pool.map(NofZmain,Har):
            p = 0
    return 0

def Func1(Har):
    if Fenli < 0.5 and Interpolate_LF > 0:
        for i in pool.map(NofZmain,Har):
            p = 0
    return 0

Func0(Har)
Func1(Har)
Func2(Har)


#Separ(Har)

'''
Nota = np.array([FEN,Separation,Use_Auld_GALFORM_Catalogue,Abs_App,Mag_limit_for_LF,LFI,k_correction])

np.save('Sep.npy',Nota)
if entropy > 1.5:
    p = 0
    for q in pool.map(Sep,Har):
        p = 0

if Separation > 0.5:
    for pu in pool.map(nofzMaine,Har):
        print (pu)

elif Interpolate_LF > 0.5:
    for sh in pool.map(LfMaine,Har):
        print (sh)



if Concatenate_Catalogues > 0.5:
    x_low = Har[0];x_up = x_low+Z_Sep
    x_low = np.float(x_low);x_up = np.float(x_up)
    Q0 = np.load('/gpfs/data/ddsm49/GALFORM/Cache1/ZSep/QuoGal_'+str(x_low)+'_'+str(x_up)+'.npy')
    for i in range(1,len(Har)):
        x_low = Har[i];x_up = x_low+Z_Sep
        x_low = np.float(x_low);x_up = np.float(x_up)
        Q1 = np.load('/gpfs/data/ddsm49/GALFORM/Cache1/ZSep/QuoGal_'+str(x_low)+'_'+str(x_up)+'.npy')
        # FasGal
        Q0 = np.concatenate((Q0,Q1))

    np.save('/gpfs/data/ddsm49/GALFORM/hamcatalogues/RusGal.npy',Q0)

if Color_Distribution > 0.5:
    for qgru in pool.map(Colodis,Har):
        print (qgru)
'''
