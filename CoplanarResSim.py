"""
This program is used for calculation of the length of quater wave or half wave resonator
Refer to Mazin and Gao's dissertation for details
"""
from math import *
import scipy.constants as const
from scipy.special import ellipk
import numpy as np
import  matplotlib.pyplot as plt
def L_k(w,Rsq,Tc,T): # Henry SI unti
    """
    kinetic inductance per unit length
    SI units
    w : width
    Rsq : sheet resistance
    Tc : critical Temeprature
    T : measurement temperature
    """
    return 1/w * Rsq*const.h/(2*pi**2 * 1.76*const.Boltzmann*Tc) * 1/tanh(1.76*Tc/(2*T))
#    return 0.18* const.hbar * Rsq/(const.Boltzmann*Tc)/w

def coplanar(w,s,epsilon_eff):
    """
    return capacitance and inductance (geometric) perunit length in SI unit
    """
    k0 = w / (w+2*s)
    k0_ = sqrt(1-k0**2)
    C = 4* const.epsilon_0 * epsilon_eff * ellipk(k0)/ellipk(k0_)
    L = const.mu_0 /4 *ellipk(k0_) / ellipk(k0)
    return C, L    
def phase_velocity(Ll,Cl):
    return 1/np.sqrt(Ll*Cl)
    
def WaveLength(freq, vp):
    """
        ref : Wallraff, Coplanar WG resonator
    """
    return vp/freq
#    return const.speed_of_light/freq*np.sqrt(2/(1+11.86))

def CouplingC(length,Ll,Cl,freq_Hz,Qint,g,Zr,option):
    """
    length of the resonator
    Ll, Cl, distributed value / unit length
    w0, angular frequency
    g : coupling coefficient
    """
    print("TL Res length %.4g um" % (length/1e-6))
    w0 = 2*np.pi*freq_Hz
    C = Cl * length/2
    L = 8*Ll*length/np.pi**2
    Qe = Qint/g # coupling quality factor
    R_loss = Qint/w0/C
    Z0 = 50 # input and output port impedance
    factor = 1 # coupling capacitance is over estimated in this version
    if option == 'Q': # quarter wave length
        Cc = sqrt(np.pi/2/Qe)/(Z0*w0)/factor
#        Cc = sqrt(Qe*2/w0/)
#        Cc=2*Qint/2/w0/Cl
    elif option == 'H' : # half wave length
        Cc = sqrt(pi/(4*Qe*w0**2*Z0*Zr))
    
    return Cc
    
def TLResonator(w,s,freq_Hz,Rsq,Tc,T):
    """
    Input geometry of the TL resonator
    w: TL width
    s: TL gap
    Output capacitance, inductance, characteristic impedance of the line, and wavelength
    """ 
    print("width = %.4g um\t gap = %.4g um, resonant freq_GHz %.3f" % (w/1.e-6,s/1e-6,freq_Hz/1e9))
    w0 = 2*pi*freq_Hz;
    epsilon_eff = (1+11.86)/2 # silicon and air
    
    Cl, Lg = coplanar(w,s,epsilon_eff)
    Lk = L_k(w,Rsq,Tc,T) # kinetic inductance per unit lenght
    Ll = Lk + Lg
    Zr = sqrt(Ll/Cl)
    vp = phase_velocity(Ll,Cl)
    lambda_ = WaveLength(freq_Hz,vp)
    print(" Geometric C: %.4g \n Geometric L: %.4g H\nKinetic L : %.4g H\n Ratio K/G %.2f \n" %(Cl, Lg,Lk,Lk/Lg))
    print("TL line impedance Zr = %.4g Ohm" % Zr)
    #print("wave length lambda_ = %.4g um" % (lambda_/1.e-6))
    return Cl,Ll,Zr,lambda_,Lg,Lk
    
if __name__=='__main__':
    # Qint: internal Q
    # Qe : coupling Q
    plt.close('all')
    Qint = 10e3
    T_K = 0.03 # base temperature
    Tc=1.75
    w_m= 10 * 1e-6 #m, width of coplanar line
    s_m = 6 * 1e-6 #m, gap of coplanar line
    g =1 # g = Qint/Qe
    # Aluminum feed line: 
    Cl,Ll,Zr,lambda_,Lg,Lk =TLResonator(w=w_m,s=s_m,freq_Hz=5.e9,Rsq=600,Tc=Tc,T=T_K) 
    Quarterwave,Halfwave = [False,True]
    Rsq_list=[650]
    
#    freq_Hz_list = np.array([5.5,6,6.5,7])*1e9 #um
    length = (np.array([558,507,462,429,399,372])) #um
    #lk = (1/(4*length*f))**2/Cl - Lg
    
    #plt.plot(lk,'o')

if True:    
    # calculating resonator length given resonant frequency
    if False:
        resonator_length_um=[]
        coupler_cap_pH=[]
        for Rsq in Rsq_list:
            print("\nRsq = %f\n" % Rsq)
            for freq_Hz in freq_Hz_list:
                Cl,Ll,Zr,lambda_,Lg,Lk= TLResonator(w_m,s_m,freq_Hz,Rsq,Tc=Tc,T=T_K)
                Ll = Ll/(1.06**2)
                vp = phase_velocity(Ll,Cl)
                lambda_ = vp/freq_Hz
                if Quarterwave:
                    length = lambda_/4.
                    Cc = CouplingC(length,Ll,Cl,freq_Hz,Qint,g,Zr,'Q')
                
                if Halfwave:
                    length = lambda_/2.
                    Cc = CouplingC(length,Ll,Cl,freq_Hz,Qint,g,Zr,'H')
                
                resonator_length_um.append(length*1e6)
                coupler_cap_pH.append(Cc*1e12)
            print('L/sq=%dpH'%(1e12*Lk*w_m))
            results = np.vstack((freq_Hz_list, resonator_length_um, coupler_cap_pH))
            results = np.transpose(results)
            print(results)
        
    # calculating resonant frequency given length
    else:
        for Rsq in Rsq_list:
            print("\nRsq = %f\n" % Rsq)
            Cl,Ll,Zr,lambda_,Lg,Lk = TLResonator(w_m,s_m,2e9,Rsq,Tc=1.75,T=T_K)
            vp = phase_velocity(Ll,Cl)
            resonator_length_um = (np.array([558,507,462,429,399,372])) #um
            wavelength = 4*resonator_length_um*1e-6 
            frequency_list=[]
            for length in wavelength:
                frequency = vp/length/1e9
                frequency_list.append(frequency)
                
            results = np.vstack((resonator_length_um,frequency_list))
            results = np.transpose(results)
            print(results)
            
    