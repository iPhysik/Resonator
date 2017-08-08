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
REFINE=True
Maryland,UCSB = [True,False][::]

def loaddata(fname):
    data = np.loadtxt(fname,skiprows=19) 
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
    fdir = "D:\Dropbox\Drive\Projects\Resonator\InOx resonator measurement\\08012017"
    fname = fdir+"\\170807-ADC_InOx05_rng200mV_f4.24_30dB-165515.txt"  
    f,S21 = loaddata(fname)
    port = notch_port(f,S21) 
    start,end=[0,f.size]
    f=f[start:end]
    S21=S21[start:end]
    
    plt.subplots(2,1)
    plt.suptitle('Raw data')
    plt.subplot(211)
    plt.plot(f,np.abs(S21),'+-')
    plt.subplot(212)
    plt.plot(f,np.unwrap(np.angle(S21)),'+-')
    
    
   # # 1st Remove electric delay ===================
    fr,Ql=port._fit_skewed_lorentzian_v2(f,S21)
    b,delay = port.get_delay(f,S21,fr,Ql,delay = 40, maxiter=int(10000)) # use _fit_circle
#    for delay in [47,48]:
    nS21 = port._remove_cable_delay(f,S21,-delay)#/(1+b*((f-fr)/fr))#+c*((f-fr)/fr)**2)
    plt.subplots()    
    plt.title('phase after delay remove')
    plt.plot(f,np.unwrap(np.angle(nS21)),'+')
    plt.subplots()
    plt.suptitle('S21 delay remove before after')
    plt.plot(S21.real,S21.imag,'.-',c='b')
    plt.plot(nS21.real,nS21.imag,'.-',c='r')
            
        
    if REFINE:
        num_iter = 5
        PLOT=False
    else:
        num_iter = 1
        PLOT=True
        
    for i in range(num_iter):

        if UCSB:
            nS21=1/nS21
            
        f_data=f[:]
        z_data=nS21[:] 
        xc,yc,r0 = port._fit_circle(z_data,refine_results=True)
        
        if PLOT:
            plt.subplots()
            plt.title("circle fit 1")
            plt.plot(np.real(z_data),np.imag(z_data),'+',xc+r0*np.cos(angles),yc+r0*np.sin(angles),xc,yc,'o')
            plt.subplots()
            plt.title('z_data phase')
            plt.plot(f,np.unwrap(np.angle(z_data)))
            
        #center the circle to obtain a and alpha
        zp_data=port._center(z_data,xc+1j*yc)
        phase = np.arctan2(zp_data.imag,zp_data.real)
        phase = np.unwrap(phase)
        if PLOT:
            plt.subplots()
            plt.title("circle after center")
            plt.plot(np.real(zp_data),np.imag(zp_data),'+',r0*np.cos(angles),r0*np.sin(angles))
            plt.subplots()
            plt.title('z_data phase after center')
            plt.plot(f,np.unwrap(np.angle(zp_data)))
        
        if UCSB:
            if phase[0]<phase[-1]:
                phase = -phase
                theta0_sign_reverse=True
            else:
                theta0_sign_reverse=False
                
        fr = f[np.where(np.abs(S21)==np.min(np.abs(S21)))[0][0]]
        theta0=0.5*(np.max(phase)+np.min(phase))
        theta0,Ql,fr=port._phase_fit(f_data,phase,theta0,Ql,fr)
        
        if UCSB: 
            Qi = Ql
            if theta0_sign_reverse:
                theta0=-theta0
                                
        print("first phase fit : theta0,Ql,fr :",theta0,Ql,fr)
        if PLOT:
            plt.subplots()
            plt.title('phase fit results vs data')
            plt.plot(f_data,phase,f_data,(phase_vs_freq([theta0, Ql, fr],f_data)))
        
        a = np.abs(xc+r0*np.cos(theta0-np.pi)+1j*(yc+r0*np.sin(theta0-np.pi)))
        alpha = np.angle(xc+r0*np.cos(theta0-np.pi)+1j*(yc+r0*np.sin(theta0-np.pi)))
        print('a, alpha:', a, alpha)
        
        if PLOT:
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
        
        if UCSB:
            z_data=z_data-1
    
        if Maryland:
            z_data=(1-z_data)
            
        xc,yc,r0 = port._fit_circle(z_data,refine_results=True)
        
        if PLOT:
            plt.subplots()
            plt.title('S21 after normalization')
            plt.plot(z_data.real,z_data.imag,'.-',xc+r0*np.cos(angles),yc+r0*np.sin(angles))
            
        phi= np.arctan2(yc,xc)
        print("phi: ", phi)
        z_data=z_data*np.exp(-1j*phi)
        
        if PLOT:
            plt.subplots()
            plt.title('plot canoncial posiiton of S21')
            plt.plot(z_data.real,z_data.imag,'.')
    #    plt.xlim(-1,1),plt.ylim(-1,1)
        
    #%%
        if Maryland:
            Qc = Ql/2/r0
            Qi = 1./(1./Ql-1./Qc)
            S21sim = a*np.exp(1j*alpha-1j*2*np.pi*f*delay)*(1-(Ql/Qc)*np.exp(1j*phi)/(1+2j*Ql*(f/fr-1)))
            
        if UCSB: 
            Qc = Qi/2/r0
            Ql=1/(1/Qc+1/Qi)
            S21sim_inverse=(1.+Qi/Qc*np.exp(1j*phi)/(1.+2.*1j*Qi*(f/fr-1.)))*(a*np.exp(1j*alpha)) 
            S21sim=S21sim_inverse
            S21sim=np.exp(-1j*2*np.pi*f*delay)/S21sim_inverse
            if PLOT:
                plt.subplots()
                plt.plot(S21sim.real,S21sim.imag,'.')
                S21sim=1/S21sim_inverse
                S21sim=np.exp(-1j*2*np.pi*f*delay)/S21sim_inverse
                plt.subplots()
                plt.plot(S21sim.real,S21sim.imag,'.')
                
        if REFINE:
            print(' Refining results:')
            for j in np.arange(num_iter):
                if Maryland:
                    popt, params_cov, infodict, errmsg, ier  =port._fit_entire_model_2(f,S21,fr,Qc,Ql,phi,delay=delay,a=a,alpha=alpha,b=b,maxiter=1000)
                    fr,Qc,Ql,phi,delay,a,alpha,b = popt
                    nS21 = port._remove_cable_delay(f,S21,-delay)#/(1+b*((f-fr)/fr))#+c*((f-fr)/fr)**2)
        
                if UCSB:
                    popt, params_cov, infodict, errmsg, ier  =port._fit_entire_model_3(f,S21,fr,Qc,Qi,phi,delay=delay,a=a,alpha=alpha,maxiter=1000)
                    fr,Qc,Qi,phi,delay,a,alpha = popt
                    nS21 = port._remove_cable_delay(f,S21,-delay)     
                    
        if PLOT or (REFINE and i==(num_iter-1)):
            fig,ax = plt.subplots(2,1)
            plt.suptitle('Raw data(blue dots) vs fitting(red solid line)')
            ax[0].plot(f,np.abs(S21),'.',c='b')
            ax[0].plot(f,np.abs(S21sim),'-',c='r')
            ax[1].plot(f,np.unwrap(np.angle(S21)),'.',c='b')
            ax[1].plot(f,np.unwrap(np.angle(S21sim)),'-',c='r')
            
        print('Qi\tQc\tQl\tfr\ta\talpha\tdelay\tphi\tb:')
        print('%d\t%d\t%d\t%.4f\t%.6f\t%.2f\t%.2f\t%.2f\t%.2f'%(Qi,Qc,Ql,fr,a,alpha,delay,phi,b))    
    
     
    
    