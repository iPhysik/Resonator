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

def loaddata(fname):
    data = np.loadtxt(fname,skiprows=2)   
    f = data[:,2]
    mag=data[:,3]
    phase=-data[:,4]
#    f = data[:,0]
#    mag = data[:,3]#/np.max(data[:,3])
#    phase = -np.deg2rad(data[:,4])   
    S21 = mag*np.cos(phase)+1j*mag*np.sin(phase)
    return f, S21           
    
if __name__ == "__main__":

    plt.close("all")    
    angles=np.linspace(0,2*np.pi,2000)

#    fname = "4_11_2013_LCmeander2_S21vsFreq_Pmeas=-133dBm_T=20mK"  
    # set file directory and name
    fdir = 'D:\\temp data\\'
    fname = fdir+"2.98GHz_30dB.dat"
    
    f,S21 = loaddata(fname)
    port = notch_port(f,S21) 
    f=f[:]
    S21=S21[:]

    plt.subplots(2,1)
    plt.suptitle('Raw data')
    plt.subplot(211)
    plt.plot(f,np.abs(S21))
    plt.subplot(212)
    plt.plot(f,np.angle(S21))
    
    
   # # 1st Remove electric delay ===================
    fr,Ql=port._fit_skewed_lorentzian_v2(f,S21)
    b,delay = port.get_delay(f,S21,fr,Ql,delay = None, maxiter=int(10000)) # use _fit_circle
    nS21 = port._remove_cable_delay(f,S21,-delay)/(1+b*((f-fr)/fr))#+c*((f-fr)/fr)**2)
#    delay=port._optimizedelay(f,nS21,Ql,fr,maxiter=100)
#    print("delay:",  delay)pp
#    nS21 = port._remove_cable_delay(f,nS21,-3)
# plot data after electric delay removed
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
    plt.subplots()
    plt.title("circle fit 1")
    plt.plot(np.real(z_data),np.imag(z_data),'+',xc+r0*np.cos(angles),yc+r0*np.sin(angles))

    #center the circle to obtain a and alpha
    zp_data=port._center(z_data,xc+1j*yc)
    fr=f[np.where(np.abs(S21)==np.min(np.abs(S21)))[0][0]]
    theta0= np.average(np.unwrap(np.arctan2(zp_data.imag,zp_data.real))[np.where(np.abs(f_data-fr)<f[1]-f[0])])
    theta0,Ql,fr=port._phase_fit(f_data,zp_data,theta0,Ql,fr)
    print("first phase fit : theta0,Ql,fr :",theta0,Ql,fr)
    plt.subplots()
    plt.title('phase fit results vs data')
    plt.plot(f_data,np.unwrap(np.angle(zp_data)),f_data,np.unwrap(phase_vs_freq([theta0, Ql, fr],f_data)))
    
    a = np.abs(xc+r0*np.cos(theta0-np.pi)+1j*(yc+r0*np.sin(theta0-np.pi)))
    alpha = np.angle(xc+r0*np.cos(theta0-np.pi)+1j*(yc+r0*np.sin(theta0-np.pi)))
    print('a, alpha:', a, alpha)
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
    plt.subplots()
    plt.title('S21 after normalization')
    plt.plot(z_data.real,z_data.imag,'.-',xc+r0*np.cos(angles),yc+r0*np.sin(angles))
    
    phi= np.arctan2(yc,xc)
    print("phi: ", phi)
    z_data=z_data*np.exp(-1j*phi)
    plt.subplots()
    plt.title('plot canoncial posiiton of S21')
    plt.plot(z_data.real,z_data.imag)
    plt.xlim(-1,1),plt.ylim(-1,1)
    
#%%
    Qc = Ql/2/r0
    Qi = 1./(1./Ql-1./Qc)

    S21sim = a*np.exp(1j*alpha-1j*2*np.pi*f*delay)*(1-(Ql/Qc)*np.exp(1j*phi)/(1+2j*Ql*(f/fr-1)))*(1+b*(f-fr)/fr)
    fig,ax = plt.subplots(2,1)
    plt.suptitle('Raw data(blue dots) vs fitting(red solid line)')
    ax[0].plot(f,np.abs(S21),'.',c='b')
    ax[0].plot(f,np.abs(S21sim),'-',c='r')
    ax[1].plot(f,np.unwrap(np.angle(S21)),'.',c='b')
    ax[1].plot(f,np.unwrap(np.angle(S21sim)),'-',c='r')
    
    print('Qi\tQc\tQl\tfr\ta\talpha\tdelay\tphi\tb:')
    print('%d\t%d\t%d\t%.4f\t%.6f\t%.2f\t%.2f\t%.2f\t%.2f'%(Qi,Qc,Ql,fr,a,alpha,delay,phi,b))    
    
    print('Start refine results:')
    num_of_iter = 20
    for j in np.arange(num_of_iter):
        popt, params_cov, infodict, errmsg, ier  =port._fit_entire_model_2(f,S21,fr,Qc,Ql,phi,delay=delay,a=a,alpha=alpha,b=b,maxiter=1000)
        
        fr,Qc,Ql,phi,delay,a,alpha,b = popt
        S21sim = a*np.exp(1j*alpha-1j*2*np.pi*f*delay)*(1-(Ql/Qc)*np.exp(1j*phi)/(1+2j*Ql*(f/fr-1)))*(1+b*(f-fr)/fr)
#        fig,ax = plt.subplots(2,1)
#        plt.suptitle('Raw data(blue dots) vs fitting(red solid line), after fit entire model')
#        ax[0].plot(f,np.abs(S21),'.',c='b')
#        ax[0].plot(f,np.abs(S21sim),'-',c='r')
#        ax[1].plot(f,np.unwrap(np.angle(S21)),'.',c='b')
#        ax[1].plot(f,np.unwrap(np.angle(S21sim)),'-',c='r')
#        print('Qi\tQc\tQl\tfr\ta\talpha\tdelay\tphi\tb:')
#        print('%d\t%d\t%d\t%.4f\t%.6f\t%.2f\t%.2f\t%.2f\t%.2f'%(Qi,Qc,Ql,fr,a,alpha,delay,phi,b))   
        
        nS21 = port._remove_cable_delay(f,S21,-delay)/(1+b*((f-fr)/fr))#+c*((f-fr)/fr)**2)
        
        # circle fit to extract alpha and a 
        f_data=f[:]
        z_data=nS21[:] 
        xc,yc,r0 = port._fit_circle(z_data,refine_results=True)
        #center the circle to obtain a and alpha
        zp_data=port._center(z_data,xc+1j*yc)
        theta0= np.average(np.unwrap(np.arctan2(zp_data.imag,zp_data.real))[np.where(np.abs(f_data-fr)<f[1]-f[0])])
        theta0,Ql,fr=port._phase_fit(f_data,zp_data,theta0,Ql,fr)
        
        if j == (num_of_iter-1):
            plt.subplots()
            plt.title('phase fit results vs data')
            plt.plot(f_data,np.unwrap(np.angle(zp_data)),f_data,np.unwrap(phase_vs_freq([theta0, Ql, fr],f_data)))
        
        a = np.abs(xc+r0*np.cos(theta0-np.pi)+1j*(yc+r0*np.sin(theta0-np.pi)))
        alpha = np.angle(xc+r0*np.cos(theta0-np.pi)+1j*(yc+r0*np.sin(theta0-np.pi)))
        # normalize
        z_data=z_data/(a*np.exp(1j*alpha)) 
        z_data=(1-z_data)
        
        xc,yc,r0 = port._fit_circle(z_data,refine_results=True)
        if j == (num_of_iter-1):
            plt.subplots()
            plt.title('S21 after normalization')
            plt.plot(z_data.real,z_data.imag,'.-',xc+r0*np.cos(angles),yc+r0*np.sin(angles))
        
        phi= np.arctan2(yc,xc)
        z_data=z_data*np.exp(-1j*phi)
        Qc = Ql/2/r0
        Qi = 1./(1./Ql-1./Qc)
    
        S21sim = a*np.exp(1j*alpha-1j*2*np.pi*f*delay)*(1-(Ql/Qc)*np.exp(1j*phi)/(1+2j*Ql*(f/fr-1)))*(1+b*(f-fr)/fr)
        if j == (num_of_iter-1):
            fig,ax = plt.subplots(2,1)
            ax[0].plot(f,np.abs(S21),'.',c='b')
            ax[0].plot(f,np.abs(S21sim),'-',c='r')
            ax[1].plot(f,np.unwrap(np.angle(S21)),'.',c='b')
            ax[1].plot(f,np.unwrap(np.angle(S21sim)),'-',c='r')
        print('Results of refitting after fitting entire model(iteration = %d'%j)
        print('Qi\tQc\tQl\tfr\ta\talpha\tdelay\tphi\tb:')
        print('%d\t%d\t%d\t%.4f\t%.6f\t%.2f\t%.2f\t%.2f\t%.2f'%(Qi,Qc,Ql,fr,a,alpha,delay,phi,b))   
