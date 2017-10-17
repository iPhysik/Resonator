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
    data = np.loadtxt(fname,skiprows=3,)   
    f = data[:,2]
    mag=data[:,3]
    phase=-data[:,4]
#    phase = np.deg2rad(data[:,5])
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
    fdir = 'D:\\Dropbox\\Drive\\Projects\\Resonator\\data\\AlOx\\'
    data_file=['170627-ADC_InOx04_f7.11_ps500mK-181421.dat']
    fname = fdir + data_file[0]
    
    f,S21 = loaddata(fname) # phase_unit='DEG' or 'RAD'
    port = notch_port(f,S21)

    if False:
        index=np.arange(580,802)
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

    if REMOVE_BACKGND:
#        frcal = f_data[np.argmin(np.abs(z_data))]
#        linear_var = port._fit_delay_and_linear_var_in_S21(f_data,z_data,frcal,delay,alpha=0,maxiter = 200)
        linear_var = -22
        z_data = z_data/(1+linear_var*(f_data/frcal-1)) 
    else: 
        linear_var = 0
    
    delay = port._guess_delay(f_data,z_data)
#    delay= 
    delay = port._fit_delay(f_data,z_data,delay,maxiter=200)
    A1, A2, A3, A4, frcal, Ql=port._fit_skewed_lorentzian(f_data,z_data)
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
    theta=-4
    fitparams = port._phase_fit(f_data,port._center(z_data,zc),theta,np.absolute(Ql),frcal)
    theta, Ql, fr = fitparams
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
    
    port.fitresults = port.circlefit(f_data,z_data,fr,Ql,refine_results=True,calc_errors=True,m=20)
    
    z_data_sim = port._S21_notch(f_data,fr=port.fitresults["fr"],Ql=port.fitresults["Ql"],Qc=port.fitresults["absQc"],phi=port.fitresults["phi0"],a=a,alpha=alpha,delay=delay) * (1+linear_var*(f_data/port.fitresults["fr"]-1))
    
    plt.subplots(2,1)
    plt.subplot(2,1,1)
    plt.title('raw data vs simulated data, Mag')
    plt.plot(f_data_origin,np.abs(z_data_origin),'+')
    plt.plot(f_data,np.abs(z_data_sim))
    plt.subplot(2,1,2)
    plt.title('raw data vs simulated data, phase')
    plt.plot(f_data_origin,np.angle(z_data_origin),'+')
    plt.plot(f_data,np.angle(z_data_sim))
    
    results = np.array([port.fitresults['fr'],
               port.fitresults['Qi_dia_corr'],
               port.fitresults['Qi_dia_corr_err'],
               port.fitresults['Qi_no_corr'],
               port.fitresults['Qi_no_corr_err'],
               port.fitresults['absQc'],
               port.fitresults['absQc_err'],
               port.fitresults['Qc_dia_corr'],
               port.fitresults['Ql'],
               port.fitresults['Ql_err'],
               port.fitresults['chi_square_'],
               delay,
               a,
               port.fitresults['Qc_dia_corr_err']
               ])
    results = np.reshape(results,(1,results.size))
    print('===========Results===========')
    pprint(port.fitresults)
    
    print('average number of photons in resonator at -130dBm input power', port.get_photons_in_resonator(-130))
    print('single photon limit power at input port(dBm)',port.get_single_photon_limit())
    
    """
#    REFINE=False

    for rounds in np.arange(1):            
        nS21 = port._remove_cable_delay(f,S21,-delay)#/(1+b*((f-fr)/fr))#+c*((f-fr)/fr)**2)
        plt.subplots()    
        plt.title('phase after delay remove')
        plt.plot(f,np.unwrap(np.angle(nS21)),'+')
        plt.subplots()
        plt.suptitle('S21 delay remove before after')
        plt.plot(S21.real,S21.imag,'.-',c='b')
        plt.plot(nS21.real,nS21.imag,'.-',c='r')
        # circle fit to extract alpha and a 
        f_data=f[:]
        z_data=nS21[:] 
        xc,yc,r0 = port._fit_circle(z_data,refine_results=True)
        
        if rounds == 0:
            plt.subplots()
            plt.title("circle fit 1")
            plt.plot(np.real(z_data),np.imag(z_data),'+',xc+r0*np.cos(angles),yc+r0*np.sin(angles))
            
        print('center the circle to obtain a and alpha')
        zp_data=port._center(z_data,xc+1j*yc)
    #    theta0= np.average(np.unwrap(np.arctan2(zp_data.imag,zp_data.real))[np.where(np.abs(f_data-fr)<(f[1]-f[0]))[0]])
        phase = np.unwrap(np.arctan2(zp_data.imag,zp_data.real))
        theta0 = phase[np.argmin(np.abs(S21))]
        theta0,Ql,fr=port._phase_fit(f_data,phase,theta0,Ql,fr=fr)
        print("first phase fit : theta0,Ql,fr :",theta0,Ql,fr)
        if rounds ==0:
            plt.subplots()
            plt.title('phase fit results vs data')
            plt.plot(f_data,np.unwrap(np.angle(zp_data)),f_data,np.unwrap(phase_vs_freq([theta0, Ql, fr],f_data)))
        
        a = np.abs(xc+r0*np.cos(theta0-np.pi)+1j*(yc+r0*np.sin(theta0-np.pi)))
        alpha = np.angle(xc+r0*np.cos(theta0-np.pi)+1j*(yc+r0*np.sin(theta0-np.pi)))
        print('a, alpha:', a, alpha)
        if rounds == 0 :
            plt.subplots()
            plt.title("Check off resonance positon ")
            plt.plot(np.real(z_data),np.imag(z_data),'+',xc+r0*np.cos(angles),yc+r0*np.sin(angles))
            plt.plot(a*np.cos(alpha),a*np.sin(alpha),'o',c='r')
        
            plt.subplots()
            plt.title('Check on resonance position obtained through fitting')
            plt.plot(f,np.abs(nS21),'+',fr,np.abs(xc+r0*np.cos(theta0)+1j*(yc+r0*np.sin(theta0))),'o')
        
        #%%
        # normalize
        z_data=z_data/(a*np.exp(1j*alpha)) 
        z_data=(1-z_data)
        
        xc,yc,r0 = port._fit_circle(z_data,refine_results=True)
    #    xc=xc-0.1
    #    r0=1.15
    #    yc=yc+0.05
        if rounds==0:
            plt.subplots()
            plt.title('S21 after normalization')
            plt.plot(z_data.real,z_data.imag,'.-',xc+r0*np.cos(angles),yc+r0*np.sin(angles))
            plt.xlim(-2,2),plt.ylim(-2,2)
    
        phi= np.arctan2(yc,xc)
        print("phi: ", phi)
        z_data=z_data*np.exp(-1j*phi)
        
        if rounds==0:
            plt.subplots()
            plt.title('plot canoncial posiiton of S21')
            plt.plot(z_data.real,z_data.imag)
            plt.xlim(-2,2),plt.ylim(-2,2)
        
    #%%
        Qc = Ql/2/r0
        Qcp= Qc*np.exp(1j*phi)
        Qi = 1./(1./Ql-np.real(1./(Qcp)))
        if rounds==0:
            S21sim = a*np.exp(1j*alpha-1j*2*np.pi*f*delay)*(1-(Ql/Qc)*np.exp(1j*phi)/(1+2j*Ql*(f/fr-1)))#*(1+b*(f-fr)/fr)
            fig,ax = plt.subplots(2,1)
            plt.suptitle('Raw data(blue dots) vs fitting(red solid line)')
            ax[0].plot(f,np.abs(S21),'.',c='b')
            ax[0].plot(f,np.abs(S21sim),'-',c='r')
            ax[1].plot(f,np.unwrap(np.angle(S21)),'.',c='b')
            ax[1].plot(f,np.unwrap(np.angle(S21sim)),'-',c='r')
            
            print('Results before refining\n: Qi\tQc\tQl\tfr\ta\talpha\tdelay\tphi\tb:')
            print('%d\t%d\t%d\t%.4f\t%.6f\t%.2f\t%.2f\t%.2f\t%.2f'%(Qi,1/np.real(1./(Qcp)),Ql,fr,a,alpha,delay,phi,0))    
        if REFINE == False:
            break
        if REFINE and rounds>0:
            print('Results of refitting after fitting entire model(iteration = %d'%rounds)
            print('Qi\tQc\tQl\tfr\ta\talpha\tdelay\tphi\tb:')
            print('%d\t%d\t%d\t%.4f\t%.6f\t%.2f\t%.2f\t%.2f\t%.2f'%(Qi,1/np.real(1./(Qcp)),Ql,fr,a,alpha,delay,phi,0))   
     
        if REFINE:
            popt, params_cov, infodict, errmsg, ier  =port._fit_entire_model(f,S21,fr,Qc,Ql,phi,delay=delay,a=a,alpha=alpha,maxiter=2000)
            fr,Qc,Ql,phi,delay,a,alpha= popt
           
    if REFINE:
        print('final refine result')
        print('Qi\tQc\tQl\tfr\ta\talpha\tdelay\tphi\tb:')
        print('%d\t%d\t%d\t%.4f\t%.6f\t%.2f\t%.2f\t%.2f\t%.2f'%(Qi,1/np.real(1./(Qcp)),Ql,fr,a,alpha,delay,phi,0))   
        plt.subplots()
        plt.title('phase fit results vs data')
        plt.plot(f_data,np.unwrap(np.angle(zp_data)),f_data,np.unwrap(phase_vs_freq([theta0, Ql, fr],f_data)))
        S21sim = a*np.exp(1j*alpha-1j*2*np.pi*f*delay)*(1-(Ql/Qc)*np.exp(1j*phi)/(1+2j*Ql*(f/fr-1)))#*(1+b*(f-fr)/fr)
        fig,ax = plt.subplots(2,1)
        plt.suptitle('Raw data(blue dots) vs fitting(red solid line)')
        ax[0].plot(f,np.abs(S21),'.',c='b')
        ax[0].plot(f,np.abs(S21sim),'-',c='r')
        ax[1].plot(f,np.unwrap(np.angle(S21)),'.',c='b')
        ax[1].plot(f,np.unwrap(np.angle(S21sim)),'-',c='r')

        
#    num_of_iter = 5
#    for j in np.arange(num_of_iter):
#        popt, params_cov, infodict, errmsg, ier  =port._fit_entire_model(f,S21,fr,Qc,Ql,phi,delay=delay,a=a,alpha=alpha,maxiter=1000)
#        
#        fr,Qc,Ql,phi,delay,a,alpha= popt
#        S21sim = a*np.exp(1j*alpha-1j*2*np.pi*f*delay)*(1-(Ql/Qc)*np.exp(1j*phi)/(1+2j*Ql*(f/fr-1)))#*(1+b*(f-fr)/fr)
##        fig,ax = plt.subplots(2,1)
##        plt.suptitle('Raw data(blue dots) vs fitting(red solid line), after fit entire model')
##        ax[0].plot(f,np.abs(S21),'.',c='b')
##        ax[0].plot(f,np.abs(S21sim),'-',c='r')
##        ax[1].plot(f,np.unwrap(np.angle(S21)),'.',c='b')
##        ax[1].plot(f,np.unwrap(np.angle(S21sim)),'-',c='r')
##        print('Qi\tQc\tQl\tfr\ta\talpha\tdelay\tphi\tb:')
##        print('%d\t%d\t%d\t%.4f\t%.6f\t%.2f\t%.2f\t%.2f\t%.2f'%(Qi,Qc,Ql,fr,a,alpha,delay,phi,b))   
#        
#        nS21 = port._remove_cable_delay(f,S21,-delay)#/(1+b*((f-fr)/fr))#+c*((f-fr)/fr)**2)
#        
#        # circle fit to extract alpha and a 
#        f_data=f[:]
#        z_data=nS21[:] 
#        xc,yc,r0 = port._fit_circle(z_data,refine_results=True)
#        #center the circle to obtain a and alpha
#        zp_data=port._center(z_data,xc+1j*yc)
#        theta0,Ql,fr=port._phase_fit(f_data,zp_data,theta0,Ql,fr)
#        
#        if j == (num_of_iter-1):
#            plt.subplots()
#            plt.title('phase fit results vs data')
#            plt.plot(f_data,np.unwrap(np.angle(zp_data)),f_data,np.unwrap(phase_vs_freq([theta0, Ql, fr],f_data)))
#        
#        a = np.abs(xc+r0*np.cos(theta0-np.pi)+1j*(yc+r0*np.sin(theta0-np.pi)))
#        alpha = np.angle(xc+r0*np.cos(theta0-np.pi)+1j*(yc+r0*np.sin(theta0-np.pi)))
#        # normalize
#        z_data=z_data/(a*np.exp(1j*alpha)) 
#        z_data=(1-z_data)
#        
#        xc,yc,r0 = port._fit_circle(z_data,refine_results=True)
#        if j == (num_of_iter-1):
#            plt.subplots()
#            plt.title('S21 after normalization')
#            plt.plot(z_data.real,z_data.imag,'.-',xc+r0*np.cos(angles),yc+r0*np.sin(angles))
#        
#        phi= np.arctan2(yc,xc)
#        z_data=z_data*np.exp(-1j*phi)
#        Qc = Ql/2/r0
#        Qcp= Qc*np.exp(1j*phi)
#        Qi = 1./(1./Ql-np.real(1./(Qcp)))
#    
#        S21sim = a*np.exp(1j*alpha-1j*2*np.pi*f*delay)*(1-(Ql/Qc)*np.exp(1j*phi)/(1+2j*Ql*(f/fr-1)))#*(1+b*(f-fr)/fr)
#        if j == (num_of_iter-1):
#            fig,ax = plt.subplots(2,1)
#            plt.suptitle('Qi = %d, Qc = %d, Ql = %d' %(Qi,1/np.real(1./(Qcp)),Ql))
#            ax[0].plot(f,np.abs(S21),'.',c='b')
#            ax[0].plot(f,np.abs(S21sim),'-',c='r')
#            ax[1].plot(f,np.unwrap(np.angle(S21)),'.',c='b')
#            ax[1].plot(f,np.unwrap(np.angle(S21sim)),'-',c='r')
#        print('Results of refitting after fitting entire model(iteration = %d'%j)
#        print('Qi\tQc\tQl\tfr\ta\talpha\tdelay\tphi\tb:')
#        print('%d\t%d\t%d\t%.4f\t%.6f\t%.2f\t%.2f\t%.2f\t%.2f'%(Qi,1/np.real(1./(Qcp)),Ql,fr,a,alpha,delay,phi,b))   

"""