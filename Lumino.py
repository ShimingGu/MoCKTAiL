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
    #korrz = 0
    if jud < A[0,1]:
        for i in range(0,5):
            korrz = 0
            korrz += A[0,i+2]*(z-0.1)**(4-i)
    elif jud < A[1,1]:
        for i in range(1,5):
            korrz = 0
            korrz = korrz+A[1,i+2]*(z-0.1)**(4-i)
    elif jud < A[2,1]:
        for i in range(2,5):
            korrz = 0
            korrz = korrz+A[2,i+2]*(z-0.1)**(4-i)
    elif jud < A[3,1]:
        for i in range(3,5):
            korrz = 0
            korrz = korrz+A[3,i+2]*(z-0.1)**(4-i)
    elif jud < A[4,1]:
        for i in range(4,5):
            korrz = 0
            korrz = korrz+A[4,i+2]*(z-0.1)**(4-i)
    elif jud < A[5,1]:
        for i in range(5,5):
            korrz = 0
            korrz = korrz+A[5,i+2]*(z-0.1)**(4-i)
    else:
        for i in range(6,5):
            korrz = 0
            korrz = korrz+A[6,i+2]*(z-0.1)**(4-i)
    return korrz

def korr(z,jud):
    if np.isscalar(z) is True:
        return korr0(z,jud)
    else:
        lz = len(z)
        korrz = np.zeros(lz)
        for i in range(lz):
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
    return 0.7*q

@jit(nopython = True)
def com_dis(x):
    return -6.8617e-03+4.2832e+03*x-9.2173e+02*x**2-1.9009e+02*x**3+1.7728e+02*x**3

@jit(nopython = True)
def lum_2_com(x):
    return 1.8004e-02-1.2001*x+4.3006e+03*x**2-1.0170e+03*x**3+28.221*x**4

@jit(nopython = True)
def Lum_d2_com(x):
    return 0.941+0.9873*x-1.9278e-04*x**2+4.0229e-08*x**3-4.2675e-12*x**4

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
        m[i] = abs_mag[i] + 0.77451 + 5*np.log10(lum_dis(cos,z[i])) + 25 + korr0(z[i],jud[i])
    return m

def Cata_abs_2_app(abs_m,z,color,cos):
    return M_app(abs_m,z,color,cos)

# find max luminosity distance

@jit(nopython = True)
def old_dl_max(abs_mag,mag_lim,z,jud):
    l = mag_lim - 0.77451 - abs_mag - 25 - korr0(z,jud)
    l = l/5
    l = 10 ** l
    l = l/0.7
    l = Lum_d2_com(l)
    return l
    
@jit(nopython = True)
def M_app_i(abs_mag,z,jud,cos):
    return abs_mag + 0.77451 + 5*np.log10(lum_dis(cos,z)) + 25 + korr0(z,jud)

def dl_max(abs_mag,mag_lim,jud,z_min,z_max,Itr = 100,Prec = 0.0001):
    zx = z_min;zd = z_max;sxdz = 10
    sxdx = M_app_i(abs_mag,zx,jud,0.285)
    sxdd = M_app_i(abs_mag,zd,jud,0.285)
    if sxdx > mag_lim:
        z_cmv = zx
    elif sxdd < mag_lim:
        z_cmv = zd
    else:
        for itrx in range(Itr):
            if sxdz < 19.5 + Prec and sxdz > 19.5 - Prec:
                break
            else:
                z_zj = 0.5*zx + 0.5*zd
                sxdz = M_app_i(abs_mag,z_zj,jud,0.285)
                if sxdz < mag_lim:
                    zx = z_zj
                elif sxdz > mag_lim:
                    zd = z_zj
        z_cmv = z_zj
        
    return z_cmv
        # NOT FINISHED       

#@jit(nopython = True)
def find_l_max(abs_mag,cos,mag_lim,z,jud,z_min,z_max):
    #l_lim = dl_max(abs_mag,mag_lim,z,jud)
    z_lim = dl_max(abs_mag,mag_lim,jud,z_min,z_max)
    m_app = M_app_i(abs_mag,z_max,jud,cos)
    if m_app < mag_lim:
        return 2,com_dis(z_max)
    else:
        return 1,com_dis(z_lim)

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
    l_small = com_dis(z_min)
    
    ht = len(gal_m_abs)
    
    Vols = np.zeros(ht)
    
    for i in range(ht):
        mag = gal_m_abs[i];zed = gal_z_obs[i];grc = gal_color[i]
        Sindkat,l_big = find_l_max(mag,cos,mag_lim,zed,grc,z_min,z_max)
        Vols[i] = Volume(l_small,l_big,sqdeg,frac)
        Vols[i] = 1/(Vols[i])
        if Vols[i] < 0:
            print ('ERROR!, '+str(Sindkat)+','+str(l_small)+','+str(l_big)+'\n')
    
    return Vols

#@jit(nopython = True)
def add_Vol2(gal_m_abs,gal_z_obs,gal_color,sqdeg,frac,mag_lim,z_min,z_max,cos):
    l_small = lum_dis(cos,z_min)
    l_big = find_l_max(gal_m_abs,cos,mag_lim,gal_z_obs,gal_color,z_min,z_max)
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

def flfc(x,pa,pb,pc,pd,pe,pf):
    return pa*(x-pb) + pc*np.cos(pd*x) + pe*np.sin(pf*x)

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
    sqdeg = 41252.96125
    
    if appa > 0.5:
        yoxi = len(gal_m_app)
    else:
        yoxi = len(gal_m_abs)

    if yoxi > 0:
        wei = Add_vol(gal_m_abs,gal_z_obs,gal_color,sqdeg,frac,mag_lim,z_min,z_max,cos)
        if appa <= 0.5:
            xax = gal_m_abs
        else:
            xax = gal_m_app
        del gal_color
        gc.collect()
        
        hist,E3 = np.histogram(xax,bins = M_bins,weights = wei)
        cumsum = np.cumsum(hist)
        del wei,xax
        gc.collect()
        
        fnz = len(hist)
        lnz = len(hist)
        for i in range(len(hist)):
            if cumsum[i]!= 0 and np.isnan(cumsum[i]) !=True:
                fnz = i
                break
        for i in range(len(hist)):
            nom = hist[len(cumsum)-1-i]
            if nom != 0 and np.isnan(nom) != True and np.isnan(nom)<100:
                lnz = len(hist)-i
                break
        centres = E2C(E3)
        centres = M_bins
    else:
        centres = []
        cumsum = []
        fnz = 41
        lnz = 41
    centres = centres[fnz:lnz];cumsum = cumsum[fnz:lnz]
    return centres,np.log10(cumsum),fnz,lnz

def LFread(pathe):
    Katze = h5py.File(pathe,'r')
    z = np.array(Katze['z_obs'])
    ##absm = np.array(Katze['abs_mag'])
    #try:
        #appm = np.array(Katze['new_app_mag'])
    #except:
    for i in [1]:
        appm = np.array(Katze['app_mag'])
    color = np.array(Katze['color'])
    return z,appm,color

def RanCho(n0,n1):
    if n0 > n1:
        n0 = n1
    q = np.zeros(n1,dtype = bool)
    for i in range(n0):
        q[i] = True
    return np.random.permutation(q)

def LFI010(Refcum,Refcen,Tarcum,Tarcen,appa,gapp,gabs,mag_lim,z_ga,c_ga,Omm):
    print ('INTERPOLATION')
    afhj = interp1d(Refcum,Refcen,fill_value="extrapolate")
    ntcen = afhj(Tarcum)
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
            gapp2[i] = 2*mag_lim
            gabs2[i] = mag_lim + M_abs_i(Omm,z_ga[i],c_ga[i],gapp2[i])
    return gabs2,gapp2

def Fitfunc(x,x0,m1,p1,m2,p2,m3,p3):
    bas = x0
    mns = m1*x**(-1) + m2*x**(-2) + m3*x**(-3)
    pls = p1*x**(1) + p2*x**(2) + p3*x**(3)
    return bas+mns+pls

def Fitfunk(x,b,a0,r,a1,k1,a2,k2,a3,a4):
    bas = b + a0*(x-r);mns = 0; pls = 0
    mns = a1*np.cos(k1*(x-r))+a3*np.cos(2*k1*(x-r))
    pls = a2*np.sin(k2*(x-r))+a4*np.sin(2*k2*(x-r))
    return bas+mns+pls

def LFR010(Refcum,Refcen,Tarcum,Tarcen,appa,gapp,gabs,mag_lim,z_ga,c_ga,Omm):
    print ('REGRESSION')
    asadal,nss = curve_fit(Fitfunk,Refcum,Refcen,bounds=(-1000,1000))
    tangun = Fitfunk(Tarcum,*asadal)
    kokrea,nss = curve_fit(Fitfunk,Tarcen,tangun,bounds=(-1000,1000))
    if appa > 0.5:
        gapp3 = Fitfunk(gapp,*kokrea)
        gabs3 = M_abs(Omm,z_ga,c_ga,gapp3)
    else:
        gabs3 = Fitfunk(gabs,*kokrea)
        gapp3 = M_app(gabs3,z_ga,c_ga,Omm)
    del gapp,gabs;gc.collect()
    for i in range(len(gapp3)):
        if gapp3[i] > mag_lim + 2:
            gapp3[i] = mag_lim + 2
            gabs3[i] = M_abs_i(Omm,z_ga[i],c_ga[i],gapp3[i])
    return gabs3,gapp3
    
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
    NorF = Conf['NorF'][()]
    Object_Numbers = Conf['LFN'][()]
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
    #apM = np.linspace(9.5**2,19.5**2,1000);app_M_bins = np.sqrt(apM)
    #abs_M_bins = np.arange(-30,-5,0.1)
    #app_M_bins = np.arange(9.5,19.5,0.1)

    if ZorM < 0.5:
        ZM = 'z_obs'
    elif Abs_App > 0.5:
        ZM = 'app_mag'
    else:
        ZM = 'abs_mag'
    
    #ZAA = 'AltGal'
    #ZAA = 'RusGal'
    ZAA = 'GALFORM'

    z_mx,mapp,c_mx = LFread(Cache+ZM+'/'+'MXXL_'+str(x_low)+'_'+str(x_up)+'.h5')
    z_ga,gapp,c_ga = LFread(Cache+ZM+'/'+ZAA+'_'+str(x_low)+'_'+str(x_up)+'.h5')

    #mapp = mappq[mappq<mag_lim]
    #gapp = gapp[gapp<mag_lim]

    if k_corr > -0.5:
        mabs = M_abs(mxxl_Omm,z_mx,c_mx,mapp)
        gabs = M_abs(galf_Omm,z_ga,c_ga,gapp)
        #gabsq = M_abs(galf_Omm,z_ga,c_ga,gappq)

    gabsori = np.array(1.0*M_abs(galf_Omm,z_ga,c_ga,gapp)).copy()
    gappori = np.array(1.0*M_app(gabs,z_ga,c_ga,galf_Omm)).copy()
    gabsori = np.array(gabsori)
    gappori = np.array(gappori)
    
    #print ((np.max(gappori-gapp),'fc_max_dif_app',x_low))
    #print ((np.mean(gappori-gapp),'fc_avg_dif_app',x_low))

    if appa > 0.5:
        lowm = np.min(mapp)
        lowg = np.min(gapp)
        lowv = np.min((lowm,lowg))
        higm = np.max(mapp)
        higg = np.max(gapp)
        higv = np.max((higm,higg))
        app_M_bins = np.linspace(lowv,higv,1000)
    else:
        lowm = np.min(mabs)
        lowg = np.min(gabs)
        lowv = np.min((lowm,lowg))
        higm = np.max(mabs)
        higg = np.max(gabs)
        higv = np.max((higm,higg))
        abs_M_bins = np.linspace(lowv,higv,1000)  

    if NorF > 0.5:
        lm0 = len(mabs)
        lm = np.int(np.around(frac*lm0))
        lg0 = len(gabs)
        lg = np.int(np.around(frac*lg0))
    else:
        lg0 = len(gabs)
        if lg0 < 2*Object_Numbers:
            lg = np.int(np.around(0.5*lg0))
            frac = 0.5
        else:
            lg = Object_Numbers
            frac = lg/lg0
        lm0 = len(mabs)
        lm = np.int(np.around(frac*lm0))

    #mc = RanCho(lm,lm0)
    #qc = mc.copy()
    #c_mx = c_mx[qc];z_mx = z_mx[qc];mapp = mapp[qc];mabs = mabs[qc]
    #del qc;gc.collect()

    if appa > 0.5:
        M_bins = app_M_bins
    else:
        M_bins = abs_M_bins
        #mapp = 0

    Refceno,Refcumo,Reffnz,Reflnz = Lf(appa,mapp,mabs,z_mx,c_mx,frac,mag_lim,x_low,x_up,mxxl_Omm,M_bins)
    #del c_mx,z_mx,mapp,mabs;gc.collect()

    Refcenap = 0
    if CrossIter > 0.5:
        Refcenap,Refcumap,Refapfnz,Refaplnz = Lf(1,mapp,mabs,z_mx,c_mx,frac,mag_lim,x_low,x_up,mxxl_Omm,app_M_bins)
    del c_mx,z_mx,mapp,mabs,Refcenap;gc.collect()
   
    if appa > 0.5:
        Refcumap = Refcum #; Reffnzap = Reffnz; Reflnzap = Reflnz
     
    for iTe in range(Iter+1):        
        ##bc = RanCho(lg,lg0)
        ##sc = bc.copy()
        ##c_ga1 = c_ga[sc];z_ga1 = z_ga[sc];gapp1 = gapp[sc];gabs1 = gabs[sc]
        sc=1;c_ga1 = 1.0*c_ga;z_ga1 = 1.0*z_ga;gapp1 = 1.0*gapp;gabs1 = 1.0*gabs
        del sc;gc.collect()
    
        if CrossIter > 0.5:
            appa = ((-1)**(iTe+1))/2.0 + 0.5

        if appa > 0.5:
            M_bins = app_M_bins
        else:
            del gapp1;gc.collect()
            gapp1 = 0
            M_bins = abs_M_bins

        Tarceno,Tarcumo,Tarfnzq,Tarlnzq = Lf(appa,gapp1,gabs1,z_ga1,c_ga1,frac,mag_lim,x_low,x_up,galf_Omm,M_bins)
        #print ('Tarcum = '+str(Tarcum))
        del c_ga1,z_ga1,gapp1,gabs1;gc.collect()
        Infolen = len(Tarcumo)

        #if appa > 0.5:
            #recu = Refcumap
            #rfnz = Reffnzap
            #rlnz = Reflnzap
        #else:
            #recu = Refcum
            #rfnz = Reffnz
            #rlnz = Reflnz

        print ('Ite,Infolen,Abs_App,z_low,z_up,Reg_Int')
        print (iTe,Infolen,appa,x_low,x_up,LFI)
        
        #Tarceno = Tarcen[Tarfnz:Tarlnz];Tarcumo = Tarcum[Tarfnz:Tarlnz]

        #del Tarcum,Tarcen;gc.collect()

        if iTe == 0:
            Tarfnz0 = Tarfnzq
            Tarlnz0 = Tarlnzq
            Tarcum0 = Tarcumo
            Tarcen0 = Tarceno

        if iTe == 1:
            Tarfnz1 = Tarfnzq
            Tarlnz1 = Tarlnzq
            Tarcum1 = Tarcumo
            Tarcen1 = Tarceno

        if iTe == 2:
            Tarfnz2 = Tarfnzq
            Tarlnz2 = Tarlnzq
            Tarcum2 = Tarcumo
            Tarcen2 = Tarceno

        if LFI > 0.75 and LFI < 1.25 and iTe < Iter:
            gabs1,gapp = LFI010(Refcumo,Refceno,Tarcumo,Tarceno,appa,gapp,gabs,mag_lim,z_ga,c_ga,galf_Omm)
            plt.figure()
            plt.scatter(gabs,gabs1,c='red',label = 'NEWMAG at '+str(iTe))
            gabs = gabs1;del gabs1;gc.collect()
            plt.xlabel('oldmag_abs');plt.ylabel('newmag_abs')
            plt.title('SHIFT at z '+str(x_low)+'_'+str(x_up)+' at '+str(iTe))
            plt.savefig(picpath+'SF_'+str(x_low)+'_'+str(x_up)+' at '+str(iTe)+'.png',dpi = 170)
        if LFI > -0.25 and LFI < 0.25 and iTe < Iter:
            gabs1,gapp1 = LFR010(Refcumo,Refceno,Tarcumo,Tarceno,appa,gapp,gabs,mag_lim,z_ga,c_ga,galf_Omm)
            plt.figure()
            plt.scatter(gabs,gabs1,c='red',label = 'NEWMAG at '+str(iTe))
            gabs = gabs1;del gabs1;gc.collect()
            plt.xlabel('oldmag_abs');plt.ylabel('newmag_abs')
            plt.title('SHIFT at z '+str(x_low)+'_'+str(x_up)+' at '+str(iTe))
            plt.savefig(picpath+'SF_'+str(x_low)+'_'+str(x_up)+' at '+str(iTe)+'.png',dpi = 170)

    plt.figure()
    plt.plot(Refceno,Refcumo,c = 'blue',label = 'MXXL')
    plt.plot(Tarcen0,Tarcum0,c = 'orange',label = 'GALFORM')
    try:
        plt.plot(Tarcen1,Tarcum1,c = 'yellow',label = 'alt_GALFORM_1')
    except:
        ertwo = 1+1
    try:
        plt.plot(Tarcen2,Tarcum2,c = 'red',label = 'alt_GALFORM_2')
    except:
        santhree = 1+1+1
    plt.plot(Tarceno,Tarcumo,c = 'green',label = 'alt_GALFORM_f')
    plt.legend()
    plt.ylabel('log Cumulative LF')
    plt.xlabel('Magnitude')
    plt.title('LF of z-range '+str(x_low)+'_'+str(x_up))
    plt.savefig(picpath+'LF_'+str(x_low)+'_'+str(x_up)+'.png',dpi = 170)
    
    print ('Critlen:',len(Tarcum0))

    #plt.figure()
    #try:
        #Quartz = Tarcum0[(Tarfnz1-Tarfnz0):-(Tarlnz0-Tarlnz1)]
        #plt.plot(Tarcen1,Tarcum1-Quartz,c = 'orange',label = 'First_Iteration')
    #except:
        #ertwo = 1+1
    #try:
        #Quartz = Tarcum1[(Tarfnz2-Tarfnz0):-(Tarlnz0-Tarlnz2)]
        #plt.plot(Tarcen1,Tarcum2-Quartz,c = 'red',label = 'Second_Iteration')
    #except:
        #santhree = 1+2
    #Qfnz = ((Tarfnzq)-(Tarfnz0)); Qlnz = ((Tarlnz0)-(Tarlnzq))
    #print ("fnzlnz's:",Tarfnzq,Tarfnz0,Tarlnzq,Tarlnz0,Qfnz,Qlnz)
    #Quartz = Tarcum0[Qfnz:-Qlnz]
    #print ("Length:",len(Quartz),len(Tarcumo))
    #plt.plot(Tarceno,Tarcumo-Quartz,c = 'green',label = 'Deviation')
    #plt.ylabel('Difference')
    #plt.xlabel('Magnitude')
    #plt.title('Shift of z-range '+str(x_low)+'_'+str(x_up))
    #plt.savefig(picpath+'SF_'+str(x_low)+'_'+str(x_up)+'.png',dpi = 170)
    
    del Refcumo,Refceno,Tarcum0,Tarcen0,Tarcumo,Tarceno,z_ga,c_ga;gc.collect()

    rusgal = str(Cache+ZM+'/'+'AltGal_'+str(x_low)+'_'+str(x_up)+'.h5')
    oldgal = str(Cache+ZM+'/'+'GALFORM_'+str(x_low)+'_'+str(x_up)+'.h5')
    #os.system('cp '+oldgal+' '+rusgal)
    try:
        os.system('rm '+rusgal)
    except:
        sifour = 1+1+1+1
    Conf0 = h5py.File(oldgal,'r')
    Conf1 = h5py.File(rusgal,'w')
    Conf1['ra'] = np.array(Conf0['ra'])
    Conf1['dec'] = np.array(Conf0['dec'])
    Conf1['z_obs'] = np.array(Conf0['z_obs'])
    Conf1['z_cos'] = np.array(Conf0['z_cos'])
    app_mag_0 =  np.array(Conf0['app_mag'])
    Conf1['old_app_mag'] = gappori
    print ((np.max(gappori-app_mag_0),'Consistency',x_low))
    print ((np.max(gappori-gapp),'max_dif_app',x_low))
    print ((np.mean(gappori-gapp),'avg_dif_app',x_low))
    ##if np.mean(gapp0 - app_mag_0) > 1E-4:
        ##print (np.max(gapp0 - app_mag_0))
    ##del Conf1['app_mag']
    Conf1['app_mag'] = gapp
    Conf1['old_abs_mag'] = gabsori
    print ((np.max(gabsori-gabs),'max_dif_abs',x_low))
    print ((np.mean(gabsori-gabs),'avg_dif_abs',x_low))
    #del Conf1['abs_mag']
    Conf1['abs_mag'] = gabs
    Conf1['gal_type'] = np.array(Conf0['gal_type'])
    Conf1['color'] = np.array(Conf0['color'])
    Conf1['halo_mass'] = np.array(Conf0['halo_mass'])
    Conf1['random_seed'] = np.array(Conf0['random_seed'])
    Conf0.close()
    Conf1.close()
    return x_up



