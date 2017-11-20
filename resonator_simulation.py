
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 22:05:53 2017

@author: Mint
"""
import matplotlib.pyplot as plt
import numpy as np

def standard_type_res(f,fr,Ql,Qc,phi0):
    return (Ql/Qc)*np.exp(1j*phi0)/(1+2j*Ql*(f/fr-1))

def notch_type_res(f,fr,Ql,Qc,phi0):
    return 1-standard_type_res(f,fr,Ql,Qc,phi0)    
    
def notch_type_ucsb(f,fr,Ql,Qc,phi0):
    return 1/(1+Qi/Qc*np.exp(1j*phi0)/(1+1j*2*Qi*(f/fr-1)))
    
    
if __name__=='__main__':
    
    plt.close('all')
    fr=6.1209980401777422 # GHz
    Ql = 305959.16249575053
    absQc = 355069.09919743845
    phi0 = 0.24601917539768126
    a = 0.95268751584495814
    Qc_complex = absQc * np.exp(-1j*phi0)
    Qc = 1/np.real(1/Qc_complex)
    Qi = 1/(1/Ql-1/Qc)
    
    print('Ql=%d'%Ql,'Qc=%d'%Qc,'Qi=%d'%Qi)
    freq_step=fr/Ql/30 # frequency step is one tenth of fr/Ql
    f=np.linspace(fr-100*freq_step,fr+100*freq_step,5000) # frequency array
    
#    S21sim=standard_type_res(f,fr,Ql,Qc)
#    fig,ax = plt.subplots(2,1)
#    ax[0].set_title('standard type resonator')
#    ax[0].plot(f,20*np.log10(np.abs(S21sim)),'.-',c='r')
#    ax[0].plot(S21sim.real,S21sim.imag,'.')
    fig,ax = plt.subplots(1,1)
#    ax.set_ylabel('dB')
    S21sim=notch_type_res(f,fr,Ql,absQc,phi0)*a
    
#    mag = 20*np.log10(np.abs(S21sim))
    mag = np.angle(S21sim)
#    ax.set_title('notch type resonator')
    ax.plot(f,mag,'.-',c='r')
    ax.xaxis.set_ticks([6.121-40e-6,6.121,6.121+40e-6])
    ax.yaxis.set_ticks([-1.2,-np.pi/4,0,np.pi/4,1])
    ax.ticklabel_format(useOffset=False)
    
    array=np.zeros((f.size,4))
    array[:,0]=f
    array[:,1]=mag
    array[:,2]=np.abs(S21sim)
    array[:,3]=np.angle(S21sim)
    
#    fname='Martinis_sim.txt'
#    np.savetxt('D:\\Dropbox\\Drive\\Projects\\Resonator\\data\\others\\'+fname,array)