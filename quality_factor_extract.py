# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 10:22:29 2016
This is a Python program to extract resonator quality factor using nonlinear fit. 
The following libraries are adapted from the codes from qkitgroup : https://github.com/qkitgroup/qkit
*calibration.py
*circlefit.py
*circuit.py
*utilities.py
@author: Wenyuan Zhang , wzhang@physics.rutgers.edu
"""
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spopt
from scipy import stats
from circuit import *
from utilities import phase_vs_freq
from pprint import pprint
from copy import deepcopy

def loaddata(fname,f_col=2,mag_col=4,phase_col=5,phase_unit='RAD'):
    data = np.loadtxt(fname,skiprows=9,delimiter=',')   
    f = data[:,0]
    mag=data[:,3]
#    mag = 10**(mag/20)
#    phase=-data[:,4]
    phase = np.deg2rad(data[:,4])
    S21 = mag*np.cos(phase)+1j*mag*np.sin(phase)
    return f, S21           

def fr_Ql_init(f,S21):
    S21 = S21 / np.abs(S21)[0]
    S21_abs= np.abs(S21)
    S21_FWHM = 1/2*(np.min(S21_abs)**2+1)
    print(S21_FWHM)
    fr_index = np.argmin(S21_abs)
    fr = f[fr_index]
    f_left = f[np.argmin(np.abs(S21_FWHM-S21_abs[0:fr_index]))]
    S21_left = S21_abs[np.argmin(np.abs(S21_FWHM-S21_abs[0:fr_index]))]
    f_right = f[fr_index+np.argmin(np.abs(S21_FWHM-S21_abs[fr_index:-1]))]
    S21_right = S21_abs[fr_index+np.argmin(np.abs(S21_FWHM-S21_abs[fr_index:-1]))]
    Ql = fr/(f_right-f_left)
    plt.subplots()
    plt.title('init fr, Ql')
    plt.plot(f,S21_abs)
    S21_sim = (np.min(S21_abs)+2j*Ql*(1-f/fr))/(1+2j*Ql*(1-f/fr))
    plt.plot(f,np.abs(S21_sim),f_left,S21_left,'o',f_right,S21_right,'o')
    return fr, Ql
    
    #%%
if __name__ == "__main__":

    plt.close("all")    
    angles=np.linspace(0,2*np.pi,2000)
    REMOVE_BACKGND = False # remove |S21| background slope or not. 

    # set file directory and name
    fdir = 'D:\\Dropbox\\Drive\\Projects\\Resonator\\MKID\\Tom Nb Design\\'
    data_file=[
'qw_smallTL_1res6060_nb_nostrip10_Qc70k-coupl_coarse2.csv'
    ]
    fname = fdir + data_file[0]
    
#    fr = 3.2436
#    absQc = 49694
#    delay = 46.9
#    a = 0.00166
#    phi0 = 0.51214
#    Ql = 17444
#    alpha = 0.398
    f,S21 = loaddata(fname) # phase_unit='DEG' or 'RAD'
#    S21=[]
#    f = np.linspace(3.242,3.246,2000)
#    for fi in f:
#        S21.append( a*np.exp(1j*alpha)*np.exp(-2*np.pi*1j*fi*delay) * ( 1 - Ql/(absQc+absQc*random.uniform(-0.3,0.3))*np.exp(1j*phi0) / (1+2*1j*Ql*(fi-fr)/fr)))
#        
    port = notch_port(f,S21)
    
    if True:
        index=np.arange(400,550)
#        index=np.hstack((index,np.arange(-100,-1)))
        f_data_origin=f[index]
        z_data_origin=S21[index]
    else:
        f_data_origin=f
        z_data_origin=S21
        
    f_data=f_data_origin
    z_data=z_data_origin

    plt.subplots(2,1)
    plt.suptitle('Raw data')
    plt.subplot(211)
    plt.plot(f_data,np.abs(z_data),'+-')
    plt.subplot(212)
    plt.plot(f_data,np.angle(z_data),'+-')
#
if True:
    
    delay = port._guess_delay(f_data,z_data)
#    delay = 0
    delay=port._fit_delay(f_data,z_data,delay,maxiter=500)
#    delay = 50
#    
    A1, A2, A3, A4, frcal, Ql=port._fit_skewed_lorentzian(f_data,z_data)
    print('Initial Ql from skewed lorentzian fit %d'%Ql)
    # remove delay
    z_data = z_data_origin*np.exp(2.*1j*np.pi*delay*f_data)

    if REMOVE_BACKGND:
        plt.subplots()
        plt.title('|S21| before and after background slope removed')
        plt.plot(f_data_origin,np.absolute(z_data_origin),'+')
        plt.plot(f_data,np.absolute(z_data),'.')

    xc, yc, r0 = port._fit_circle(z_data,refine_results=True)

    plt.subplots()
    plt.plot(np.real(z_data),np.imag(z_data),'.')
    plt.title('circle fit')
    xc, yc, r0 = port._fit_circle(z_data,refine_results=True)
    plt.plot(xc+r0*np.cos(angles),yc+r0*np.sin(angles))
## 

    zc = np.complex(xc,yc)
    theta = np.angle(port._center(z_data,zc))[np.argmin(f_data-frcal)]
    theta = 0
    fitparams = port._phase_fit(f_data,port._center(z_data,zc),theta,np.absolute(Ql),frcal)
    theta, Ql, fr = fitparams
#    Ql = 3180
    plt.subplots()
    plt.title('phase fit before after')
    _phase = np.angle(port._center(z_data,zc)) 
    plt.plot(f_data,np.unwrap(_phase),'.')

    if Ql<0:
        raise('Ql is less than zero')
    _phase = theta+2.*np.arctan(2.*Ql*(1.-f_data/fr))
    plt.plot(f_data, _phase)
#    beta = port._periodic_boundary(theta+np.pi,np.pi)
    beta = theta + np.pi
    offrespoint = np.complex((xc+r0*np.cos(beta)),(yc+r0*np.sin(beta)))
    alpha = np.angle(offrespoint)
    a = np.absolute(offrespoint)
#    a = 0.00135
    plt.subplots()
    plt.title('off res point')
    plt.plot(np.real(z_data),np.imag(z_data),'.-')
    plt.plot(xc+r0*np.cos(angles),yc+r0*np.sin(angles))
    plt.plot(offrespoint.real,offrespoint.imag,'o')
    # normalize 
    z_data = z_data/a*np.exp(1j*(-alpha))
    plt.subplots()
    plt.title('after normalization')
    plt.plot(z_data.real,z_data.imag,'.')
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    
    port.fitresults = port.circlefit(f_data,z_data,fr,Ql,refine_results=True,calc_errors=True)
    fr,absQc,Ql,phi0 = [port.fitresults['fr'],
                                      port.fitresults['absQc'],
                                      port.fitresults['Ql'],
                                      port.fitresults['phi0']]
    
#    for i in np.arange(100):
    results_entire_model = port.results_from_fit_entire_model(f_data_origin,z_data_origin,
                                                                     fr,absQc,Ql,phi0,delay,a,alpha,ftol=1e-10,xtol=1e-10,maxfev=2000)
#    fr,absQc,Ql,phi0,delay,a,alpha = popt
#        
    #fit_errs = np.sqrt(np.diag(params_cov)) 
    #fr_err,absQc_err,Ql_err,phi0_err,delay_err,a_err,alpha_err =  fit_errs
#if False:   
    z_data_sim = port._S21_notch(f_data,fr,Ql,absQc,phi0,a,alpha,delay)
    
    plt.subplots(2,1)
    plt.subplot(2,1,1)
    plt.title('raw data vs simulated data, Mag')
    plt.plot(f_data_origin,np.abs(z_data_origin),'.')
    plt.plot(f_data,np.abs(z_data_sim),c='r')
    plt.subplot(2,1,2)
    plt.title('raw data vs simulated data, phase')
    plt.plot(f_data_origin,np.angle(z_data_origin),'.')
    plt.plot(f_data,np.angle(z_data_sim),c='r')
    
    print('===========Results===========')
    pprint(results_entire_model)
    results_old = np.array([port.fitresults['fr'],
               port.fitresults['fr_err'],
               port.fitresults['Qi_dia_corr'],
               port.fitresults['Qi_dia_corr_err'],
               port.fitresults['Qc_dia_corr'],
               port.fitresults['Qc_dia_corr_err'],
               port.fitresults['Ql'],
               port.fitresults['Ql_err'],
               port.fitresults['chi_square_'],
               port.fitresults['absQc'],
               port.fitresults['absQc_err'],
               delay,
               0,
               a,
               0,
               port.fitresults['phi0'],
               port.fitresults['phi0_err']
               ])
    results_old = np.reshape(results_old,(1,results_old.size))
    results = [results_entire_model['fr'],
               results_entire_model['fr_err'],
               results_entire_model['Qi'],
               results_entire_model['Qi_err'],
                results_entire_model['Qc'],
                results_entire_model['Qc_err'],
                results_entire_model['Ql'],
                results_entire_model['Ql_err'],
                results_entire_model['chisquare'],
                results_entire_model['absQc'],
                results_entire_model['absQc_err'], 
                results_entire_model['delay'],
                results_entire_model['delay_err'],
                results_entire_model['a'],
                results_entire_model['a_err'],
                results_entire_model['phi0'],
                results_entire_model['phi0_err']

]
    results = np.reshape(results,(1,np.size(results)))

    print('average number of photons in resonator at -130dBm input power', port.get_photons_in_resonator(-130))
    print('single photon limit power at input port(dBm)',port.get_single_photon_limit())
    
  