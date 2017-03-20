# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 10:22:29 2016

@author: Mint
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spopt
from scipy import stats
from circuit import *

def plotcircle(f,S21):
    fig,ax = plt.subplots()
    ax.plot(S21.real,S21.imag,'.')
    
def plotphase(f,S21):
    fig,ax = plt.subplots()
    ax.plot(f,np.unwrap(np.angle(S21)))
    
def phase_vs_freq(p,x,):
    theta0, Ql, fr = p
    return theta0+2.*np.arctan(2.*Ql*(1.-x/fr))

if __name__ == "__main__":

    plt.close("all")    
    
    #fname = "01042015_ADC_S21vsFreq_LowT_Dev2_Power=-121dBm"    
    fname = "4_11_2013_LCmeander2_S21vsFreq_Pmeas=-133dBm_T=20mK"   
    print fname
    data = np.loadtxt(fname)    
    f = data[:,0]
    print "frequency min %f and max %f" %(f.min(),f.max())
    mag = data[:,3]#/np.max(data[:,3])
    phase = -np.deg2rad(data[:,4])
    S21 = mag*np.cos(phase)+1j*mag*np.sin(phase)
    
    port = notch_port(f,S21)    
    
    # 1st Remove electric delay
#    guess_delay = port._guess_delay(f[:30],S21[:30])
#    print "guess delay %f" % guess_delay
    
#    for i in np.arange(1):
    delay,paras = port.get_delay(f,S21,delay = 0)
    #print "iter %d, delay = %f" %(i,delay)
    print "get_delay"
    print delay
    
#    delay = port._get_delay(f,S21,delay = guess_delay)
    nS21 = port._remove_cable_delay(f,S21,-delay)
    
    # Lorentzian fit to obtain fr and Ql as input for further simulation
    popt= port._fit_skewed_lorentzian(f,S21)
    #plotcircle(f,nS21)
    fr,Ql = popt[3],popt[4]
    
    # circle fit to prepare for a e^i\alpha calibration
    print "fr, Ql %f %f" % (fr,Ql)
    xc,yc,r0 = port._fit_circle(nS21)
    zc = np.complex(xc,yc) 
    z_data = port._center(nS21,zc)
#    plotcircle(f,z_data)
    
    theta0= phase[np.argmin(np.abs(z_data))]
    p0 = port._phase_fit(f,z_data,theta0,Ql,fr)
#    plt.plot(f,phase_vs_freq(p0,f),'.')
    print "phase fit paras"
    print p0
    fr = p0[2]
    Ql = p0[1]
   # plt.plot(f,np.unwrap(np.arctan2(z_data.imag,z_data.real)),'.',f,phase_vs_freq(p0,f),'.')    
    
    
    theta0 = p0[0] # phase correspond to resonance 
    print "theta0 after phase fit %f " % theta0
    #beta = port._periodic_boundary(theta0+np.pi,np.pi)
    beta = theta0+np.pi
    offrespoint = np.complex((xc+r0*np.cos(beta)),(yc+r0*np.sin(beta)))
    alpha = np.angle(offrespoint)
    a = np.absolute(offrespoint)
    print    "alpha and a  %f %f" % (alpha,a)
    
    
    # normalize S21
    nnS21 = nS21/(a*np.exp(1j*alpha)) 
    #plotcircle(f, nnS21)
    plt.subplots()
    plt.plot(nS21.real,nS21.imag,'.',offrespoint.real,offrespoint.imag,'.',c='r')
    plt.subplots()
    plt.plot(nnS21.real,nnS21.imag,'.')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    
    xc,yc,r0 = port._fit_circle(nnS21)   
    plt.subplots()
    plt.plot(nnS21.real,nnS21.imag,'.',xc,yc,'.')
    zc = np.complex(xc,yc)
    z_data = port._center(nnS21,zc)
    #plotcircle(f,z_data)    
        
    theta0 = np.pi-np.arcsin(yc/r0)
    p0= port._phase_fit(f,z_data,theta0,Ql,fr)            
    print p0
    plt.subplots()
    plt.plot(f,np.unwrap(np.angle(z_data))+2*np.pi,'.',f,phase_vs_freq(p0,f),'.')    
    
    # Quality factor calculation
    if (2*r0)>1:
        r0=0.5
        
    Qc = Ql/2/r0
    Qi = 1./(1./Ql-1./Qc)
    print "Qi %f, Qc %f, Ql %f,fr %f "% (Qi,Qc,Ql,fr)
    
    #transverse to canonical location    
    xc,yc = 1-xc,-yc
    #print xc,yc,r0
    theta = np.arctan2(yc,xc)
    print theta
    fS21 = (1-nnS21)/np.exp(1j*theta)
    plt.subplots()
    plt.plot(fS21.real,fS21.imag,'.')
    #%%
    xc,yc,r0 = port._fit_circle(fS21)
    #print xc,yc,r0
    S21sim = a*np.exp(1j*alpha-1j*2*np.pi*f*0)*(1-(Ql/Qc)*np.exp(1j*theta)/(1+2j*Ql*(f/fr-1)))
    plt.subplots()
    plt.plot(fS21.real,fS21.imag,'.',c='b')
    plt.plot(S21sim.real,S21sim.imag,'.',c='r')
    #%%
    fig,ax = plt.subplots(2,1)
    ax[0].plot(f,mag,'.',f,np.abs(S21sim),'.-')
    ax[1].plot(f,np.angle(S21),'.',f,np.angle(S21sim),'.-')
