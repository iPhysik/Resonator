# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 10:22:29 2016

@author: Mint
"""
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spopt
from scipy import stats
from circuit import *

def plotcircle(f,S21,title=''):
    fig,ax = plt.subplots()
    ax.plot(S21.real,S21.imag,'.')
    plt.title(title)
    
def plotphase(f,S21):
    fig,ax = plt.subplots()
    ax.plot(f,np.unwrap(np.angle(S21)))
    
def phase_vs_freq(p,x):
    theta0, Ql, fr = p
    return theta0+2.*np.arctan(2.*Ql*(1.-x/fr))
    
def skewed_lorentzian(x,A2,A3,fra,Ql):
    return A2*(x-fra)+(Smax+A3*(x-fra))/np.sqrt(1.+4.*Ql**2*((x-fra)/fra)**2)

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
    fdir = 'D:\\Google Drive\\Projects\\Resonator\\InOx resonator measurement\\'
    fname = fdir+"170602-ADC_InOx04_f6.06-140322_slide16_AeroflexCh1=30dB.dat"
    f,S21 = loaddata(fname)
    port = notch_port(f,S21) 
    f=f[150:]
    S21=S21[150:]
    plt.plot(f,np.abs(S21))
   # # 1st Remove electric delay ===================
  
    delay,fr,Ql = port.get_delay(f,S21,delay = None, maxiter=int(500)) # use _fit_circle
    nS21 = port._remove_cable_delay(f,S21,-delay)
    plt.subplots()
    plt.title('phase after delay remove')
    plt.plot(f,np.unwrap(np.angle(nS21)),'+')

    plt.subplots()
    plt.suptitle('S21 delay remove before after')
    plt.plot(S21.real,S21.imag,'.',c='b')
    plt.plot(nS21.real,nS21.imag,'.-',c='r')
    # circle fit to extract alpha and a 
    f_data=f[120:-220]
    z_data=nS21[120:-220] # set data for circle fit, ignore the data away from resonance and distorted
    xc,yc,r0 = port._fit_circle(z_data,refine_results=True)
    plt.subplots()
    plt.title("circle fit 1")
    plt.plot(np.real(z_data),np.imag(z_data),'+',xc+r0*np.cos(angles),yc+r0*np.sin(angles))

    #center the circle to obtain a and alpha
    zp_data=port._center(z_data,xc+1j*yc)
    #%%
    theta0= np.average(np.unwrap(np.arctan2(zp_data.imag,zp_data.real))[np.where(np.abs(f_data-fr)<f[1]-f[0])])
    theta0,Ql,fr=port._phase_fit(f_data,zp_data,theta0,Ql,fr)
    print("first phase fit : theta0,Ql,fr :",theta0,Ql,fr)
    plt.subplots()
    plt.plot(f_data,np.angle(zp_data),f_data,phase_vs_freq([theta0, Ql, fr],f_data))
    a = np.abs(xc+r0*np.cos(theta0-np.pi)+1j*(yc+r0*np.sin(theta0-np.pi)))
    alpha = np.angle(xc+r0*np.cos(theta0-np.pi)+1j*(yc+r0*np.sin(theta0-np.pi)))
    plt.subplots()
    plt.title("cricle fit 2")
    plt.plot(np.real(z_data),np.imag(z_data),'+',xc+r0*np.cos(angles),yc+r0*np.sin(angles))
    plt.plot(a*np.cos(alpha),a*np.sin(alpha),'o',c='r')

    plt.subplots()
    plt.title('mag vs freq')
    plt.plot(f,np.abs(nS21),'+',fr,np.abs(xc+r0*np.cos(theta0)+1j*(yc+r0*np.sin(theta0))),'o')

    #%%
    # normalize
    nnS21 = nS21/(a*np.exp(1j*alpha)) 
    nnS21 = (1-nnS21)     # now the S21 is lorenztian times e^(i phi)
    z_data=z_data/(a*np.exp(1j*alpha)) 
    z_data=(1-z_data)
    
    xc,yc,r0 = port._fit_circle(z_data,refine_results=True)
    plt.subplots()
    plt.title('S21 after normalization')
    plt.plot(z_data.real,z_data.imag,'.-',xc+r0*np.cos(angles),yc+r0*np.sin(angles))
    
    phi= np.arctan2(yc,xc)
    print("phi: ", phi)
    nnS21 = nnS21*np.exp(-1j*phi)
    z_data=z_data*np.exp(-1j*phi)
    plt.subplots()
    plt.plot(z_data.real,z_data.imag)
    plt.xlim(-1,1),plt.ylim(-1,1)
    
    plt.subplots()
    plt.plot(np.angle(z_data))
#%%
    Qc = Ql/2/r0
    Qi = 1./(1./Ql-1./Qc)

    print("Qi %f, Qc %f, Ql %f,fr %f "% (Qi,Qc,Ql,fr))
    
