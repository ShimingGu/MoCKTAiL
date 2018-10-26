#!/usr/bin/python -u
import numpy as np
import scipy.signal as con
import multiprocessing
import os
import h5py
import cython
import gc
import sys
import datetime

t0 = datetime.datetime.now()

#
# Default Parameter Settings
#

MXXL_path = b"/gpfs/data/ddsm49/GALFORM/catalogues/MXXL.h5"
GALFORM_path = b"/gpfs/data/ddsm49/GALFORM/catalogues/GALF.h5"
Alt_GALFORM_path = b"/gpfs/data/ddsm49/GALFORM/catalogues/Altg.h5"
picsave_path = b"./pics/"
catalogue_cache_path = b"/gpfs/data/ddsm49/GALFORM/Cache2/"
# PLEASE ADD THE PATH AFTER THE LETTER b

App_Mag_min = 9.5
App_Mag_max = 19.5
App_Mag_Sep = 0.5
# The choice of the apparent magnitude limit and the width of the each apparent magnitude range

Abs_Mag_min = -30.0
Abs_Mag_max = -5.0
Abs_Mag_Sep = 0.5
# The choice of the absolute magnitude limit and the width of the each absolute magnitude range

#Mag_min = 9.5
#Mag_max = 19.5
#Mag_Sep = 0.5
# The choice of the magnitude limit and the width of the each magnitude range

Abs_App = 'Abs'
# The choice of using apparent magnitude or the absolute magnitude

Z_min = 0
Z_max = 0.50
Z_Sep = 0.01
# The choice of the redshift limit and the width of the each redshift range

#
# AUTOMATIC MODES
#

Automatic_Mode = 1
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

Catalogue_Separation = 1
# Separate the catalogues in order to make the code parallel
# 0 = "REUSE", 1 = "REDO", 2 = "DELETE AND REDO"

Separation_Mode = 'Magnitude'

#
# PLOT SETTINGS
#

Cumulative_N_OF_Z = 0

Interpolate_LF = 0
LF_iterations = 8
Plot_LF = 1
LF_Interpolation = '1-D Interpolation'
Mag_limit_for_LF = 19.5
k_correction = 1
Cross_Iteration = 0
Fractions = 0.17

Concatenate_Catalogues = 0
# Plot the Luminosity Functions only or apply the correction to a new catalogue

Color_Distribution = 0
PLOT_OLD_GALFORM = 1

###################################################################
######################### I AM A LINE #############################
###################################################################

cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=cores)

os.system('mkdir '+str(catalogue_cache_path.decode("utf-8")))
os.system('mkdir '+str(picsave_path.decode("utf-8")))

#Mags = np.arange(Mag_min,Mag_max,Mag_Sep)
#Zs = np.arange(Z_min,Z_max,Z_Sep)
#MS = np.array([Mag_Sep])
#ZD = np.array([Z_Sep])
#Mags = np.around(10000*Mags);Mags = Mags.astype('int');Mags = Mags/10000.0
#Zs = np.around(10000*Zs);Zs = Zs.astype('int');Zs = Zs/10000.0

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

if Abs_App < 0.5:
    Mag_min = Abs_Mag_min
    Mag_max = Abs_Mag_max
    Mag_Sep = Abs_Mag_Sep
else:
    Mag_min = App_Mag_min
    Mag_max = App_Mag_max
    Mag_Sep = App_Mag_Sep

Fenli = Separation_Mode

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

if Interpolate_LF == 1 and LF_iterations < 1:
    LF_iterations = 1

###################################################################
######################### I AM A LINE #############################
###################################################################

Mags = np.arange(Mag_min,Mag_max,Mag_Sep)
Zs = np.arange(Z_min,Z_max,Z_Sep)
MS = np.array([Mag_Sep])
ZD = np.array([Z_Sep])
Mags = np.around(10000*Mags);Mags = Mags.astype('int');Mags = Mags/10000.0
Zs = np.around(10000*Zs);Zs = Zs.astype('int');Zs = Zs/10000.0

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

LFI = 1

Conf = h5py.File('Config.h5','w')
Conf['Separation'] = FEN
Conf['ZorM'] = Fenli
Conf['Old_GALFORM'] = Use_Auld_GALFORM_Catalogue
Conf['Abs_App'] = Abs_App
Conf['LF_Mag_limit'] = Mag_limit_for_LF
Conf['LF_Interpolation_Technique'] = LFI
Conf['LF_Iteration_Numbers'] = LF_iterations
Conf['Cross_Iteration'] = Cross_Iteration
Conf['k_corr'] = k_correction
Conf.attrs['MXXL_path'] = np.void(MXXL_path)
Conf.attrs['GALFORM_path'] = np.void(GALFORM_path)
Conf.attrs['Alt_GALFORM_path'] = np.void(Alt_GALFORM_path)
Conf.attrs['picsave_path'] = np.void(picsave_path)
Conf.attrs['catalogue_cache_path'] = np.void(catalogue_cache_path)
Conf['plot_old_galform'] = PLOT_OLD_GALFORM
Conf['FRAC'] = Fractions
Conf.close()

#sys.path.append(r'./src')
from Catalo import Separa,Conca
from NofZ import NofZmain
from Lumino import LFmain
#from Colodis import CDmain

def Func0(Har):
    if Catalogue_Separation > 0.5:
        for i in pool.map(Separa,Har):
            p = 0
    return 0

def Func1(Har,Plot_N_of_Z = 0):
    global NofZmain
    if Fenli > 0.5 and Plot_N_of_Z > 0:
        for i in pool.map(NofZmain,Har):
            p = 0
    return 0

def Func2(Har,ILF,Fl):
    global LFmain
    if Fl < 0.5 and ILF > 0:
        for i in pool.map(LFmain,Har):
            p = 0
    return 0


#Func0(Zs)
#Func2(Zs)
#Conca(Zs)

if Automatic_Mode == 1:
    Conf = h5py.File('Config.h5','r+')
    del Conf['ZorM'],Conf['Separation']
    Conf['ZorM'] = 0
    Conf['Separation'] = Z_Sep
    Conf.close()
    #Func0(Zs)
    #Func2(Zs,Interpolate_LF,Fenli)
    #Conca(Zs)
    Conf = h5py.File('Config.h5','r+')
    del Conf['ZorM'],Conf['Separation']
    del Conf['Old_GALFORM'],Conf['plot_old_galform']
    Conf['ZorM'] = 1
    Conf['Separation'] = Mag_Sep
    Conf['Old_GALFORM'] = 0
    Conf['plot_old_galform'] = 1
    Conf.close()
    Func0(Mags)
    Func1(Mags,Plot_N_of_Z = 1)

t1 = datetime.datetime.now() - t0
print ('Total Time Cost = '+str(t1)+' \n')
