import numpy as np
import copy
import gc
import os
import h5py
from numba import jit
import astropy.coordinates
import astropy.units as u
import astropy.cosmology
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#
# k correction calculations:
#

# k correction

@jit(nopython = True)
def korr0(z,jud):
    kA4 = -0.103
    A = np.array([[0.131,0.2145,-45.33,35.28,-6.604,-0.4805,kA4],[0.298,0.3705,-20.08,20.14,-4.620,-0.04824,kA4],[0.443,0.523,-10.98,14.36,-3.676,0.3395,kA4],[0.603,0.694,-3.428,9.478,-2.703,0.7646,kA4],[0.785,0.859,6.717,3.250,-1.176,1.113,kA4],[0.933,1.0,16.76,-2.514,0.3513,1.307,kA4],[1.067,200.0,20.30,4.189,0.5619,1.494,kA4]])
    #A = np.load('k_cor.npy')
    korrz = 0
    if jud < A[0,1]:
        for i in range(0,5):
            korrz = korrz+A[0,i+2]*(z-0.1)**(4-i)
    elif jud < A[1,1]:
        for i in range(1,5):
            korrz = korrz+A[1,i+2]*(z-0.1)**(4-i)
    elif jud < A[2,1]:
        for i in range(2,5):
            korrz = korrz+A[2,i+2]*(z-0.1)**(4-i)
    elif jud < A[3,1]:
        for i in range(3,5):
            korrz = korrz+A[3,i+2]*(z-0.1)**(4-i)
    elif jud < A[4,1]:
        for i in range(4,5):
            korrz = korrz+A[4,i+2]*(z-0.1)**(4-i)
    elif jud < A[5,1]:
        for i in range(5,5):
            korrz = korrz+A[5,i+2]*(z-0.1)**(4-i)
    else:
        for i in range(6,5):
            korrz = korrz+A[6,i+2]*(z-0.1)**(4-i)
    return korrz

def korr(z,jud):
    if np.isscalar(z) is True:
        return korr0(z,jud)
    else:
        korrz = 1.0*z
        for i in range(len(z)):
            korrz[i] = korr0(z[i],jud[i])
        return korrz

# luminosity distance

@jit(nopython = True)
def lum_dis(Omm,z):
    q = 0.489456 - 3.859143*Omm + 7.027757*Omm**2 + 3.345856*Omm**3 + 14.471052*Omm**4 -70.475186*Omm**5
    q = q + 2981.394565*z + 3083.987229*z**2 - 15.492082*z**3 - 277.416978*z**4 + 62.848594*z**5
    q = q + 148.567432*Omm*z - 3053.048403*Omm*z**2 - 3920.010186*Omm*z**3 + 1203.677375*Omm*z**4
    q = q - 390.857385*Omm**2*z + 2653.349969*Omm**2*z**2 + 3968.235102*Omm**2*z**3
    q = q + 68.111828*Omm**3*z - 3050.132999*Omm**3*z**2 + 647.435083*Omm**4*z
    return q

# main function of this part

#@jit(nopython = True)
def M_abs(cos,z,jud,r):
    M = np.zeros(len(r))
    for i in range(len(z)):
        M[i] = r[i] - 0.77451 - 5*np.log10(lum_dis(cos,z[i])) - 25 - korr(z[i],jud[i])
    return M

@jit(nopython = True)
def M_abs_i(cos,z,jud,r):
    return r - 0.77451 - 5*np.log10(lum_dis(cos,z)) - 25 - korr0(z,jud)

# apply it to the catalogues

def Nova_abs_mag_r(z_obs,color,app_m,cos):
    return M_abs(cos,z_obs,color,app_m)


#################################################################
########################## I AM A LINE ##########################
#################################################################


#
# Luminosity Function Calculations:
#


# apprarent magnitude

@jit(nopython = True)
def M_app(abs_mag,z,jud,cos):
    m = np.zeros(len(abs_mag))
    for i in range(len(abs_mag)):
        m[i] = abs_mag[i] - 0.77451 - 5*np.log10(lum_dis(cos,z[i])) + 25 + korr0(z[i],jud[i])
    return m

def Cata_abs_2_app(abs_m,z,color,cos):
    return M_app(abs_m,z,color,cos)

# find max luminosity distance

@jit(nopython = True)
def dl_max(abs_mag,mag_lim,z,jud):
    l = mag_lim + 5*np.log10(0.7) - abs_mag - 25 - korr0(z,jud)
    l = l/5
    l = 10 ** l
    return l

@jit(nopython = True)
def M_app_i(abs_mag,z,jud,cos):
    return abs_mag - 0.77451 - 5*np.log10(lum_dis(cos,z)) + 25 + korr0(z,jud)

#@jit(nopython = True)
def find_l_max(abs_mag,cos,mag_lim,z,jud,z_max):
    l_lim = dl_max(abs_mag,mag_lim,z,jud)
    m_app = M_app_i(abs_mag,z_max,jud,cos)
    if m_app < mag_lim:
        return lum_dis(cos,z_max)
    else:
        return l_lim

# calculate the volume

@jit(nopython = True)
def Volume(l_small,l_big,sqdeg,frac):
    sterad = sqdeg * (np.pi/180) ** 2
    v_small = sterad/3 * l_small ** 3
    v_big = sterad/3 * l_big ** 3
    Vol = v_big - v_small
    Vol = Vol * frac
    return Vol

#@jit(nopython = True)
def Add_vol(gal_m_abs,gal_z_obs,gal_color,sqdeg,frac,mag_lim,z_min,z_max,cos):
    l_small = lum_dis(cos,z_min)
    
    ht = len(gal_m_abs)
    
    Vols = np.zeros(ht)
    
    for i in range(ht):
        l_big = find_l_max(gal_m_abs[i],cos,mag_lim,gal_z_obs[i],gal_color[i],z_max)
        Vols[i] = Volume(l_small,l_big,sqdeg,frac)
        Vols[i] = 1/(Vols[i])
    
    return Vols

#@jit(nopython = True)
def add_Vol2(gal_m_abs,gal_z_obs,gal_color,sqdeg,frac,mag_lim,z_min,z_max,cos):
    l_small = lum_dis(cos,z_min)
    l_big = find_l_max(gal_m_abs,cos,mag_lim,gal_z_obs,gal_color,z_max)
    vols = Volume(l_small,l_big,sqdeg,frac)
    vols = 1/(vols)
    return Vols

@jit(nopython = True)
def E2C(edges):
    centres = np.empty(len(edges)-1)
    for i in range(len(centres)):
        centres[i] = (edges[i+1] - edges[i])/2 + edges[i]
    return centres

def flf1(x,pa,pb,pc,pd,pe,pf):
    p = 0.0
    return pa + pb/(x-p) + pc/((x-p)**2) + pd/((x-p)**3) + pe/((x-p)**4) + pf/((x-p)**5)

def flf2(x,pa,pb,pc,pd,pe,pf):
    p = 0.0
    return pa + pb/(x-p) + pc/((x-p)**2) + pd/((x-p)**3) + pe/((x-p)**4) + pf/((x-p)**5)

def flf3(x,qq,pa,pb,pc,pd,pe,pf):
    p = -5.0
    return qq + pa*np.exp(pb*(x-p)) + pc*np.exp(pd*(x-p)) + pe*np.exp(pf*(x-p))

def glg1(x,qa,qb,qc):#,qe,qg):
    return qa + qb*x + qc*x**3# + qe*x**5 + qg*x**7

def gita(x,qq,qa,qb,qc,qd,qe,qf):
    p = -30
    return qq + qa*np.log(qb*(x-p)) + qc*np.log(qd*(x-p)) + qe*np.log10(qf*(x-p))

def Lf(appa,gal_m_app,gal_m_abs,gal_z_obs,gal_color,frac,mag_lim,z_min,z_max,cos,M_bins):
    sqdeg = 4*np.pi/(np.pi/180)**2
    
    if appa > 0.5:
        yoxi = len(gal_m_app)
    else:
        yoxi = len(gal_m_abs)

    if yoxi > 0:
        wei = Add_vol(gal_m_abs,gal_z_obs,gal_color,sqdeg,frac,mag_lim,z_min,z_max,cos)
        if appa <= 0.5:
            gal_m_app = gal_m_abs
        del gal_z_obs,gal_color
        gc.collect()
        
        hist,E3 = np.histogram(gal_m_app,bins = M_bins,weights = wei)
        cumsum = np.cumsum(hist)
        del gal_m_abs,gal_m_app,wei
        gc.collect()
        
        fnz = len(hist)
        lnz = len(hist)
        for i in range(len(hist)):
            if hist[i]!= 0:
                fnz = i
                break
        for i in range(len(hist)):
            if hist[len(hist)-1-i] != 0:
                lnz = len(hist)-i
                break
        centres = E2C(E3)
        centres = M_bins
    else:
        centres = []
        cumsum = []
        fnz = 41
        lnz = 41
    return centres,np.log10(cumsum),fnz,lnz

def LFread(pathe):
    Katze = h5py.File(pathe,'r')
    z = np.array(Katze['z_obs'])
    abs = np.array(Katze['abs_mag'])
    app = np.array(Katze['app_mag'])
    color = np.array(Katze['color'])
    return z,abs,app,color

def RanCho(n0,n1):
    if n0 > n1:
        n0 = n1
    q = np.zeros(n1,dtype = bool)
    for i in range(n0):
        q[i] = True
    return np.random.permutation(q)

def LFI010(Refcum,Refcen,Tarcum,Tarcen,appa,gapp,gabs,mag_lim,z_ga,c_ga,Omm):
    afhj = interp1d(Refcum,Refcen,fill_value="extrapolate")
    ntcen = afhj(Tarcum)
    SHIF = ntcen - Tarcen
    maha = interp1d(Tarcen,ntcen,fill_value="extrapolate")
    if appa > 0.5:
        gapp2 = maha(gapp)
        gabs2 = M_abs(Omm,z_ga,c_ga,gapp2)
    else:
        gabs2 = maha(gabs)
        gapp2 = M_app(gabs2,z_ga,c_ga,Omm)
    del gapp,gabs;gc.collect()
    for i in range(len(gapp2)):
        if np.isnan(gapp2[i]) == True:
            gapp2[i] = mag_lim + 0.5
            gabs2[i] = M_abs_i(Omm,z_ga[i],c_ga[i],gapp2[i])
    return gabs2,gapp2

def LFmain(y):
    Conf = h5py.File('./Config.h5','r')
    x_sep = np.array(Conf['Separation'])[()]
    ZorM = np.array(Conf['ZorM'])[()]
    appa = np.array(Conf['Abs_App'])[()]
    mag_lim = np.array(Conf['LF_Mag_limit'])[()]
    qTar = np.array(Conf['Old_GALFORM'])[()]
    frac = np.array(Conf['FRAC'])[()]
    LFI = np.array(Conf['LF_Interpolation_Technique'])[()]
    print ('LFI = '+str(LFI)+'\n')
    k_corr = np.array(Conf['k_corr'])[()]
    wTar = np.array(Conf['plot_old_galform'])[()]
    Iter = Conf['LF_Iteration_Numbers'][()]
    CrossIter = Conf['Cross_Iteration'][()]
    if qTar < 0.5:
        TAR0 = 'RusGal'
    else:
        TAR0 = 'GALFORM'
    Cache = Conf.attrs['catalogue_cache_path'].tostring().decode("utf-8")
    picpath = Conf.attrs['picsave_path'].tostring().decode("utf-8")
    Conf.close()

    x_low = round(y,5)
    x_up = round(x_low + round(x_sep,5),5)
    print (x_up)

    galf_Omm = 0.307
    mxxl_Omm = 0.25
    apM = np.linspace(9.5**2,19.5**2,1000);app_M_bins = np.sqrt(apM)
    abs_M_bins = np.arange(-30,-5,0.001)
    app_M_bins = np.arange(9.5,19.5,0.001)

    if ZorM < 0.5:
        ZM = 'z_obs'
    elif Abs_App > 0.5:
        ZM = 'app_mag'
    else:
        ZM = 'abs_mag'
    z_mx,mabs,mapp,c_mx = LFread(Cache+ZM+'/'+'MXXL_'+str(x_low)+'_'+str(x_up)+'.h5')
    z_ga,gabs,gapp,c_ga = LFread(Cache+ZM+'/'+'GALFORM_'+str(x_low)+'_'+str(x_up)+'.h5')

    if k_corr > 0.5:
        mabs = M_abs(mxxl_Omm,z_mx,c_mx,mapp)
        gabs = M_abs(galf_Omm,z_ga,c_ga,gapp)

    lm0 = len(mabs)
    lm = np.int(np.around(frac*lm0))
    lg0 = len(gabs)
    lg = np.int(np.around(frac*lg0))
    qc = RanCho(lm,lm0)
    c_mx = c_mx[qc];z_mx = z_mx[qc];mapp = mapp[qc];mabs = mabs[qc]
    del qc;gc.collect()

    if appa > 0.5:
        M_bins = app_M_bins
    else:
        M_bins = abs_M_bins
        del mapp;gc.collect()
        mapp = 0

    Refcen,Refcum,Reffnz,Reflnz = Lf(appa,mapp,mabs,z_mx,c_mx,frac,mag_lim,x_low,x_up,mxxl_Omm,M_bins)
    del c_mx,z_mx,mapp,mabs;gc.collect()
    Refceno = Refcen[Reffnz:Reflnz];Refcumo = Refcum[Reffnz:Reflnz]
    del Refcen;gc.collect()

    for iTe in range(Iter):        
        bc = RanCho(lg,lg0)
        qc = bc.copy()
        c_ga1 = c_ga[qc];z_ga1 = z_ga[qc];gapp1 = gapp[qc];gabs1 = gabs[qc]
        del qc;gc.collect()
    
        if CrossIter > 0.5:
            appa = ((-1)**iTe)/2.0 + 0.5

        Tarcen,Tarcum,Tarfnz,Tarlnz = Lf(appa,gapp1,gabs1,z_ga,c_ga,frac,mag_lim,x_low,x_up,galf_Omm,M_bins)
        del c_ga1,z_ga1,gapp1,gabs1;gc.collect()

        if Tarfnz > Reffnz:
            fnz = Tarfnz
        else:
            fnz = Reffnz
        if Tarlnz < Reflnz:
            lnz = Tarlnz
        else:
            lnz = Reflnz

        Tarceno = Tarcen[Tarfnz:Tarlnz];Tarcumo = Tarcum[Tarfnz:Tarlnz]

        if iTe > 0.5 and np.max(np.abs((Tarcum[fnz:lnz] - Refcum[fnz:lnz])/(Refcum[fnz:lnz]))) < 0.017:
            break

        del Tarcum,Tarcen;gc.collect()

        if iTe == 0:
            Tarcum0 = Tarcumo
            Tarcen0 = Tarceno

        if LFI > 0.75 and LFI < 1.25:
            gabs,gapp = LFI010(Refcumo,Refceno,Tarcumo,Tarceno,appa,gapp,gabs,mag_lim,z_ga,c_ga,galf_Omm)
                
    plt.figure()
    plt.plot(Refceno,Refcumo,label = 'MXXL')
    plt.plot(Tarcen0,Tarcum0,label = 'GALFORM')
    plt.plot(Tarceno,Tarcumo,label = 'alt_GALFORM')
    plt.legend()
    plt.ylabel('log Cumulative LF')
    plt.xlabel('Magnitude')
    plt.title('LF of z-range '+str(x_low)+'_'+str(x_up))
    plt.savefig(picpath+'Comparison_LF_'+str(x_low)+'_'+str(x_up)+'.png',dpi = 170)
    
    del Refcumo,Refceno,Tarcum0,Tarcen0,Tarcumo,Tarceno,z_ga,c_ga;gc.collect()

    rusgal = str(Cache+ZM+'/'+'RusGal_'+str(x_low)+'_'+str(x_up)+'.h5')
    oldgal = str(Cache+ZM+'/'+'GALFORM_'+str(x_low)+'_'+str(x_up)+'.h5')
    os.system('cp '+oldgal+' '+rusgal)
    Conf0 = h5py.File(oldgal,'r')
    Conf1 = h5py.File(rusgal,'r+')
    Conf1['old_app_mag'] = np.array(Conf0['app_mag'])
    del Conf1['app_mag']
    Conf1['app_mag'] = gapp
    Conf1['old_abs_mag'] = np.array(Conf0['abs_mag'])
    del Conf1['abs_mag']
    Conf1['abs_mag'] = gabs
    Conf0.close()
    Conf1.close()

    return 0



