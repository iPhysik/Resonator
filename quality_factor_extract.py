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
import os
import matplotlib as mpl
from scipy.stats import chi2

mpl.rcParams['axes.formatter.useoffset'] = False

def loaddata(fname,skiprows=0, f_col=0,mag_col=3,phase_col=4,delimiter='\t',phase_unit='RAD'):
    data = np.loadtxt(fname,skiprows=skiprows,delimiter=delimiter)   
    f = data[:,f_col]
    mag=data[:,mag_col]
#    mag = 10**(mag/20)
    if phase_unit =='RAD':
        phase=-data[:,phase_col]
    else:
        phase = np.deg2rad(data[:,phase_col])
    S21 = mag*np.cos(phase)+1j*mag*np.sin(phase)
    atten = data[0,0]
    return f, S21, atten       

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

def index_btw_fmin_fmax(f_min,f_max,f_data):
    index_min = np.argmin(np.abs(f_data-f_min))
    index_max = np.argmin(np.abs(f_data-f_max))
    index = np.arange(index_min,index_max+1)
    return index

def chisqr_test(port, results_entire_model, f_data_origin,z_data_origin,f_min_chisqr,f_max_chisqr):
    df = results_entire_model['degrees of freedom']
    meas_err = port.measurement_error_estimate(f_data_origin,z_data_origin,[f_min_chisqr,f_max_chisqr])
    chisqr = results_entire_model['residual']/meas_err
    print('chisquare is {},degrees of freedom is {}'.format(chisqr,df))
    results_entire_model.update({'normalized chisqr': chisqr/results_entire_model['degrees of freedom']})
    prob = 1- chi2.cdf(chisqr,df)
    results_entire_model.update({'Probability': prob}) # the probability of chisquare greater than chisquare obtained
    #%%
if __name__ == "__main__":

    plt.close("all")    
    angles=np.linspace(0,2*np.pi,2000)

    # set file directory and name
    fdir = 'D:\\Dropbox\\Drive\\Projects\\Resonator\\'
    data_file=[ 
            'qw_2RES_maxmin_alox3_Lk=438pHsq.csv'
]

    fname = os.path.join(fdir,'MKID',data_file[0])
    TEST = False
    LORENTZIAN = True
    fit_entire_with_delay = False
#    fr = 3.2436
#    absQc = 49694
#    delay = 46.9
#    a = 0.00166
#    phi0 = 0.51214
#    Ql = 17444
#    alpha = 0.398
    if False:
        f,S21, atten = loaddata(fname,skiprows=2, f_col=2,mag_col=3,phase_col=4,phase_unit='RAD',delimiter='\t') # phase_unit='DEG' or 'RAD'
    else:
        f,S21, atten = loaddata(fname,skiprows=10, f_col=0,mag_col=3,phase_col=4,delimiter=',',phase_unit='DEG')
#        
    port = notch_port(f,S21)
    
    f_min,f_max = [5.05,5.15]
    index_min = np.argmin(np.abs(f-f_min))
    index_max = np.argmin(np.abs(f-f_max))
    index = np.arange(index_min,index_max+1)
    f=f[index]
    S21=S21[index]
    f_min_delay,f_max_delay = [f_max-0.01,f_max+0.]
    f_min_chisqr,f_max_chisqr = [f_min_delay,f_max_delay]
    f_min,f_max=[f_min+0.0,f_max-0.0]
    #f_min_chisqr,f_max_chisqr = [f_max-0.001,f_max]
    f_min_circlefit, f_max_circlefit = [f_min+0.00,f_max-0.00]
    index_min = np.argmin(np.abs(f-f_min))
    index_max = np.argmin(np.abs(f-f_max))
    index = np.arange(index_min,index_max+1)
    f_data_origin=f[index]
    z_data_origin=S21[index]
        
    f_data=f_data_origin
    z_data=z_data_origin

    plt.subplots(2,1)
    plt.suptitle('Raw data')
    plt.subplot(211)
    plt.plot(f_data,np.abs(z_data),'+-')
    plt.subplot(212)
    plt.plot(f_data,np.unwrap(np.angle(z_data)),'+-')
    
    delay,delay_err = port._guess_delay(f,S21,f_min=f_min_delay,f_max=f_max_delay)
    print('Linear delay fit results: Delay={},delay_err={}'.format(delay,delay_err))
    if fit_entire_with_delay:
        delay=port._fit_delay(f_data,z_data,delay,maxiter=1000)      
#    
    if LORENTZIAN: # Use fit_skewed_lorentzian to initialize parameters for further fitting 
        A1, A2, A3, A4, frcal, Ql=port._fit_skewed_lorentzian(f_data,z_data)
        print('Initial Ql from skewed lorentzian fit %d'%Ql)
    else:
        Ql = 1000
        frcal = f_data[np.argmin(np.abs(z_data))]
        print('Manually initialize Ql and fr by eyeball. Ql=%d, fr=%.4fGHz:'%(Ql,frcal))
    
    # remove delay
    z_data = z_data*np.exp(2.*1j*np.pi*delay*f_data)
    index=index_btw_fmin_fmax(f_min_circlefit,f_max_circlefit,f_data)
    xc, yc, r0 = port._fit_circle(z_data[index],refine_results=True)

    plt.subplots()
    plt.plot(np.real(z_data),np.imag(z_data),'.-',c='b')
    plt.plot(np.real(z_data[index]),np.imag(z_data[index]),'o',c='r')
    plt.title('circle fit')
    plt.plot(xc+r0*np.cos(angles),yc+r0*np.sin(angles))
##
if TEST == False:
        
    zc = np.complex(xc,yc)
    plt.subplots()
    plt.title('phase fit before after')
    _phase = np.unwrap(np.angle(port._center(z_data,zc)))
    plt.plot(f_data,(_phase),'.')
#    theta = np.angle(port._center(z_data,zc))[np.argmin(np.abs(z_data))]
    theta = (_phase[0]+_phase[-1])/2
    print('Init theta {}'.format(theta))

    fitparams = port._phase_fit(f_data,port._center(z_data,zc),theta,np.absolute(Ql),frcal)
    theta, Ql, fr = fitparams
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
    
    port.fitresults = port.circlefit(f_data,z_data,fr=fr,Ql=Ql,m=[f_min_chisqr,f_max_chisqr],refine_results=True,calc_errors=True)
    fr,absQc,Ql,phi0 = [port.fitresults['fr'],
                                      port.fitresults['absQc'],
                                      port.fitresults['Ql'],
                                      port.fitresults['phi0']]
    results_entire_model = port.results_from_fit_entire_model(f_data_origin,z_data_origin, fr,absQc,Ql,phi0,delay,a,alpha,ftol=1e-8,xtol=1e-8,maxfev=1000,entire=fit_entire_with_delay)
    if fit_entire_with_delay==False:
        if results_entire_model['delay_err'] ==0:
            results_entire_model['delay_err'] = delay_err
        else:
            raise ValueError('delay err is not zero when fit entire model without delay')
            
    fr,absQc,Ql,phi0,delay,a,alpha = [results_entire_model['fr'],
                                       results_entire_model['absQc'],
                                       results_entire_model['Ql'],
                                       results_entire_model['phi0'],
                                       results_entire_model['delay'],
                                       results_entire_model['a'],
                                       results_entire_model['alpha']
               ]

    chisqr_test(port, results_entire_model, f, S21,f_min_chisqr,f_max_chisqr)
    
    z_data_sim = port._S21_notch(f_data_origin,fr,Ql,absQc,phi0,a,alpha,delay)
    
    plt.subplots(2,1)
    plt.subplot(2,1,1)
    plt.title('raw data vs simulated data, Mag')
    plt.plot(f_data_origin,np.abs(z_data_origin),'.')
    plt.plot(f_data_origin,np.abs(z_data_sim),c='r')
    plt.subplot(2,1,2)
    plt.title('raw data vs simulated data, phase')
    plt.plot(f_data_origin,np.angle(z_data_origin),'.')
    plt.plot(f_data_origin,np.angle(z_data_sim),c='r')
    
    pprint(results_entire_model)
    results = [atten,
               results_entire_model['fr'],
               results_entire_model['fr_err'],
               results_entire_model['Qi'],
               results_entire_model['Qi_err'],
                results_entire_model['Qc'],
                results_entire_model['Qc_err'],
                results_entire_model['Ql'],
                results_entire_model['Ql_err'],
                results_entire_model['normalized chisqr'],
                results_entire_model['Probability'],
                results_entire_model['absQc'],
                results_entire_model['absQc_err'], 
                results_entire_model['delay'],
                results_entire_model['delay_err'],
                results_entire_model['a'],
                results_entire_model['a_err'],
                results_entire_model['phi0'],
                results_entire_model['phi0_err'],
                results_entire_model['alpha'],
                results_entire_model['alpha_err'],
                data_file[0]
]
    results = '\t'.join(map(str,results))
    labels = 'fr\t\
              fr_err\t\
              Qi\t\
              Qi_err\t\
              Qc\t\
              Qc_err\t\
              Ql\t\
              Ql_err\t\
              chisquare\t\
              Probability\t\
              asbQc\t\
              absQc_err\t\
              delay\t\
              delay_err\t\
              a\t\
              a_err\t\
              phi0\t\
              phi0_err\t\
              alpha\t\
              alpha_err\t\
              '
              

    print('average number of photons in resonator at -130dBm input power', port.get_photons_in_resonator(-130, results_entire_model['fr'], results_entire_model['Ql'],results_entire_model['Qc']))
    print('single photon limit power at input port(dBm)',port.get_single_photon_limit(results_entire_model['fr'], results_entire_model['Ql'],results_entire_model['Qc']))
    
  