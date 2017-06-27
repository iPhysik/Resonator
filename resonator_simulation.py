# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 22:05:53 2017

@author: Mint
"""
import matplotlib.pyplot as plt
import numpy as np

def standard_type_res(f,fr,Ql,Qc):
    return (Ql/Qc)/(1+2j*Ql*(f/fr-1))

def notch_type_res(f,fr,Ql,Qc):
    return 1-standard_type_res(f,fr,Ql,Qc)    
    
if __name__=='__main__':
    
    plt.close('all')
    Qi = 668e3
    Qc = 100e3
    Ql = 1/(1/Qi+1/Qc)
    print('Ql=',Ql,'Qc=',Qc,'Qi=',Qi)
    fr=6 # GHz
    freq_step=fr/Ql/30 # frequency step is one tenth of fr/Ql
    f=np.linspace(fr-500*freq_step,fr+500*freq_step,5000) # frequency array
    
    S21sim=standard_type_res(f,fr,Ql,Qc)
    fig,ax = plt.subplots(2,1)
    ax[0].set_title('standard type resonator')
    ax[0].plot(f,20*np.log10(np.abs(S21sim)),'.-',c='r')
    ax[0].set_ylabel('dB')
    S21sim=notch_type_res(f,fr,Ql,Qc)
    ax[1].set_title('notch type resonator')
    ax[1].plot(f,20*np.log10(np.abs(S21sim)),'.-',c='r')
    ax[1].set_ylabel('dB')    