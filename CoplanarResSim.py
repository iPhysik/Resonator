"""
This program is used for calculation of the length of quater wave or half wave resonator
Refer to Mazin and Gao's dissertation for details
"""
from math import *
import scipy.constants as const
from scipy.special import ellipk
import numpy as np

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

def WaveLength(freq, Ll, Cl):
    """
        ref : Wallraff, Coplanar WG resonator
    """
    return 1/(freq*sqrt(Ll*Cl))
#    return const.speed_of_light/freq*np.sqrt(2/(1+11.86))
def func_frequency(wavelength,Ll,Cl):
    return 1/wavelength/sqrt(Ll*Cl)

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
    
    f=-4*Z0*w0**2*Cc+freq_Hz
    print("Loaded resonance frequency GHz %.4f" %(f/1e9))
    print("L=%.3fnH\tC=%.3fpF\tR_loss=%.3f KOhm\tCc=%.3fpF" % (L*1e9,C*1e12,R_loss/1e3,Cc*1e12))
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
    lambda_ = WaveLength(freq_Hz,Ll,Cl)
    print(" Geometric L: %.4g H \n Kinetic L : %.4g H\n Ratio K/G %.2f \n" %(Lg,Lk,Lk/Lg))
    print("TL line impedance Zr = %.4g Ohm" % Zr)
    print("wave length lambda_ = %.4g um" % (lambda_/1.e-6))
    return Cl,Ll,Zr,lambda_
    
if __name__=='__main__':
    Qint = 10e3
#    Tc_K= 1.7
    T = 0.08 # base temperature
    freq_Hz = 3e9 #Hz
    w_m= 10 * 1e-6 #m, width of coplanar line
    s_m = 6 * 1e-6 #m, gap of coplanar line
    g =1
    # Aluminum feed line: 
    a=TLResonator(300e-6,180e-6,6e9,2500,1.7,0.2)        
    
    for Rsq in [2200]:
        print("\nRsq = %f\n" % Rsq)
       
        Cl,Ll,Zr,lambda_ = TLResonator(w_m,s_m,freq_Hz,Rsq,Tc=2.2,T=T)
        half_wavelength = [686,610,549,499,457,422,392,366]
        wavelength = np.array(half_wavelength)*1e-6*2
        frequency = func_frequency(wavelength,Ll,Cl)/1e9
        print('Lsq %.0fpH'%(w_m*Ll*1e12))
#    print(frequency)
        # quarter wave length 
     #   length = lambda_/4
      #  Cc_q = CouplingC(length,Ll,Cl,freq_Hz,Qint,g,Zr,'Q')
        
        
#        length = lambda_/2
#        Cc_h = CouplingC(length,Ll,Cl,freq_Hz,Qint,g,Zr,'H')
        
    #Cl,Ll,Zr,lambda_ = TLResonator(60e-6,10e-6,freq_Hz,250,Tc,T)
