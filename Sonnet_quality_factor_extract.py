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
import glob
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
import logging,time
from utilities import dBm2Watt

mpl.rcParams['axes.formatter.useoffset'] = False
#formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
#handler = logging.FileHandler(os.path.join(OUTPUT_DIR,'fit.log'))      
#handler.setFormatter(formatter)
#logger = logging.getLogger('fit_log')
#logger.setLevel(logging.DEBUG)
#logger.addHandler(handler)  
loggers = {}

#fdir_GoogleDrive = 'C:\Wenyuan\Google Drive'

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
    if f_col !=0:
        atten = data[:,0]
        return f, S21, atten    
    else :
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
    
def get_DUT_Input_power(Ch1atten,fr,DR_input_power=-20):
    try:
        fname = 'D:\\RU_GoogleDrive\\MW_Calibration\\Attenuation-DR_Input_line-180122.dat'
        data = np.loadtxt(fname)
    except FileNotFoundError:
        fname = os.path.join(os.getcwd(),'../MW_Calibration/Attenuation-DR_Input_line-180122.dat')
        data = np.loadtxt(fname)
        
    f = data[:,0]
    power = data[:,1]+DR_input_power
    DUT_input_power = np.interp(fr,f,power)-Ch1atten
    return DUT_input_power
    
def get_photons_in_resonator(power, fr, Ql, Qc, unit='dBm'):
    '''
    returns the average number of photons
    for a given power in units of W
    unit can be 'dBm' or 'watt'
    '''
    from scipy.constants import hbar

    if unit=='dBm':
        power = dBm2Watt(power)
#            fr = self.fitresults['fr']
#            k_c = fr/self.fitresults['Qc']
#            k_i = fr/self.fitresults['Qi']
#            return 4.*k_c/(2.*np.pi*hbar*fr*(k_c+k_i)**2) * power

        return 2/(hbar*(2*np.pi*fr*1e9)**2) * Ql**2 / Qc * power
    
         
def Cali_input_to_DUT_power():
    from scipy.interpolate import interp1d
    fdir = 'D:\\Dropbox\\_People__Wenyuan\\MW_Calibration\\01222018'
    fname = '180122-VNAamp_3Cables+Amp_40dBP1atten-195121.txt'
    data = np.loadtxt(os.path.join(fdir,fname),skiprows=2)
    f1 = data[:,0]/1e9
    power_amplifier = data[:,1]+40
    fname = '180122-VNAamp_3Cables+Amp_InputLine-172751.txt'
    data = np.loadtxt(os.path.join(fdir,fname),skiprows=2)
    f2 = data[:,0]/1e9
    power2 = data[:,1]
    f = np.linspace(1,14,1601)
    S21_1 = interp1d(f1,power_amplifier)
    S21_2 = interp1d(f2,power2)
    S21 = S21_2(f)-S21_1(f)-20
    plt.plot(f,S21)
    data = np.zeros((f.size,2))
    data[:,0]=f
    data[:,1]=S21
    
    with open(os.path.join(fdir,'InputPower2DUT.dat'),'w') as fh:
        fh.write('#freq\tPower\n')
        fh.write('#GHz\tdBm\n')
    with open(os.path.join(fdir,'InputPower2DUT.dat'),'ab') as fh:
        np.savetxt(fh,data,delimiter='\t')
#%%
def fit_quality_factor(temperature, fname,OUTPUT_DIR, Delay=None, f_range=None, MEAS = True, REFIT=False):
    logger = logging.getLogger('fit_log')
    logger.setLevel(logging.DEBUG)
    plt.close("all")    
    
    TEST,LORENTZIAN,FIT_DELAY = [False,True,True] # 
    fit_entire_with_delay = True # fit entire model that containts or not contain the electric delay
    if Delay ==None:
        delay, delay_err =[47.55,0.15] # if FIT_DELAY == False
    else: 
        delay, delay_err = Delay
        
    logger.info('init delay=%.2f +- %.2f'%(delay,delay_err))
    angles=np.linspace(0,2*np.pi,1001)

#    OUTPUT_DIR = 'D:\\Dropbox\\Drive\\Projects\\Resonator\\data\\180208AlOx06'
##    fitting_parameter_fdir = 
##    fitting_plot_fdir = 
#    fdir = 'X:\\wsLu\\RawData\\MW'
#    
#    data_file_list=glob.glob(os.path.join(fdir,'AlOx06_ADC_2.78GHzRes_CH1=%ddB_20180210_*.dat'%ch1atten))
#    if data_file_list==[]:
#        print('measurement performed at %ddB not exist.' % ch1atten)
#        return None
#    tf = np.array([])
#    tS21 = np.array([])
#    tatten = np.array([])
    logger.info('Start processing %s'%fname)
    if MEAS:
        output_dir = os.path.join(OUTPUT_DIR,os.path.basename(fname)[0:-4])
    else: 
        output_dir = os.path.join(OUTPUT_DIR,os.path.basename(fname)[0:-4]+'fit_simulation')
    logger.info('Output results to %s'%output_dir)
    try:
        os.mkdir(output_dir)
    except WindowsError:
        pass
            
    if MEAS:
        try:
            f,S21, atten_ = loaddata(fname, f_col=2,mag_col=3,phase_col=4,delimiter=' ',phase_unit='RAD') # phase_unit='DEG' or 'RAD'
        except ValueError:
            f,S21, atten_ = loaddata(fname, f_col=2,mag_col=3,phase_col=4,delimiter='\t',phase_unit='RAD')
        ch1atten = atten_[0]
        
    else:
        f,S21 = loaddata(fname,skiprows=10, f_col=0,mag_col=3,phase_col=4,delimiter=',',phase_unit='DEG')
        atten_=[0]

    atten = atten_[0]
    port = notch_port(f,S21)

    if f_range == None:
        f_min,f_max = [f[0],f[-1]] # set range for final fit, which is fit_entire_model
    else:
        f_min,f_max = f_range
    
    index_min = np.argmin(np.abs(f-f_min))
    index_max = np.argmin(np.abs(f-f_max))
    index = np.arange(index_min,index_max+1)
    
    
    f_data_origin=f[index]
    z_data_origin=S21[index]

    f_ = f_data_origin
    S21_ = z_data_origin    
    
    f_data=f_data_origin
    z_data=z_data_origin
    
    plt.subplots(2,1)
    plt.suptitle('Raw data')
    plt.subplot(211)
    plt.plot(f_data_origin,np.abs(f_data_origin),'+-')
    plt.plot(f_data,np.abs(z_data),'+-',c='r')
    plt.subplot(212)
    plt.plot(f_data_origin,np.unwrap(np.angle(z_data_origin)),'+-')
    plt.plot(f_data,np.unwrap(np.angle(z_data)),'+-',c='r')
    plt.savefig(os.path.join(output_dir,'raw data.png'))
    
    if FIT_DELAY:
        delay,delay_err = port._guess_delay(f_,S21_,f_min=f_min,f_max=f_max)
        logger.info('Linear delay fit results: Delay={},delay_err={}'.format(delay,delay_err))
        
        if fit_entire_with_delay:
            delay=port._fit_delay(f_data,z_data,delay,maxiter=1000)      
#   
    while True: 
        if LORENTZIAN: # Use fit_skewed_lorentzian to initialize parameters for further fitting 
            A1, A2, A3, A4, frcal, Ql=port._fit_skewed_lorentzian(f_data,z_data)
            plt.savefig(os.path.join(output_dir,'Lorentzian fit.png'))
            logger.info('Ql from Lorentzian %d, %.4fGHz'%(Ql,frcal))
            if Ql<0 or Ql>1e7:
                LORENTZIAN=False
            else:
                break
#                logger.info('Initial Ql from skewed lorentzian fit %d'%Ql)
        else:
            Ql = 10000
    #        frcal = f_data[np.argmin(np.abs(z_data))]
            frcal = f_data[np.argmin(np.abs(z_data))]
            logger.info('Manually initialize Ql and fr by eyeball. Ql=%d, fr=%.4fGHz:'%(Ql,frcal))
            break
        
    # remove delay
    z_data = z_data*np.exp(2.*1j*np.pi*delay*f_data)
    xc, yc, r0 = port._fit_circle(z_data,refine_results=True)

    plt.subplots()
    plt.plot(np.real(z_data),np.imag(z_data),'.-',c='b')
    plt.title('circle fit')
    plt.plot(xc+r0*np.cos(angles),yc+r0*np.sin(angles))
    plt.savefig(os.path.join(output_dir,'prelimilary circle fit.png'))
    
    if TEST== False:
        zc = np.complex(xc,yc)
        plt.subplots()
        plt.title('phase fit before after')
        _phase = np.unwrap(np.angle(port._center(z_data,zc)))
        plt.plot(f_data,(_phase),'.')
    #    theta = np.angle(port._center(z_data,zc))[np.argmin(np.abs(z_data))]
        theta = (_phase[0]+_phase[-1])/2
        #theta = -5
        logger.info('Init theta {}'.format(theta))
    
        theta, Ql, fr  = port._phase_fit(f_data,port._center(z_data,zc),theta,np.absolute(Ql),frcal)
        #fr = frcal
        logger.info('after phase fit %d %d %.2f '%(theta,Ql,fr))
        if Ql<0:
#            raise('Ql is less than zero')
            logger.info('Ql is less than zero')
            Ql = -Ql
        _phase = theta+2.*np.arctan(2.*Ql*(1.-f_data/fr))
        plt.plot(f_data, _phase)
        plt.savefig(os.path.join(output_dir,'prelimilary phase fit.png'))
        
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
        plt.savefig(os.path.join(output_dir,'off res point.png'))
        
        plt.subplots()
        plt.title('after normalization')
        plt.plot(z_data.real,z_data.imag,'.')
        plt.xlim(0,1.5)
        plt.ylim(-1.5,1,5)
        plt.savefig(os.path.join(output_dir,'after normalization.png'))
        
        if True:
            port.fitresults = port.circlefit(f_data,z_data,fr=fr,Ql=Ql,refine_results=True,calc_errors=True)
            fr,absQc,Ql,phi0 = [port.fitresults['fr'],
                                              port.fitresults['absQc'],
                                              port.fitresults['Ql'],
                                              port.fitresults['phi0']]
        
        results_entire_model = port.results_from_fit_entire_model(f_data_origin,z_data_origin, fr,absQc,Ql,phi0,delay,a,alpha,ftol=1e-5,xtol=1e-5,maxfev=1000,entire=fit_entire_with_delay)
        if results_entire_model == False:
            logger.debug('Fitting does not converge %s'%fname)
            return None
            
        if fit_entire_with_delay==False:
            if results_entire_model['delay_err'] ==0:
                results_entire_model['delay_err'] = delay_err
            else:
                raise ValueError('delay err is not zero when fit entire model without delay')
                
        fr,absQc,Qc,Ql,phi0,delay,a,alpha = [results_entire_model['fr'],
                                           results_entire_model['absQc'],
                                           results_entire_model['Qc'],
                                           results_entire_model['Ql'],
                                           results_entire_model['phi0'],
                                           results_entire_model['delay'],
                                           results_entire_model['a'],
                                           results_entire_model['alpha']
                   ]
    
        chisqr_test(port, results_entire_model,f_,S21_,f_min,f_max)
        
        z_data_sim = port._S21_notch(f_,fr,Ql,absQc,phi0,a,alpha,delay)
        
        matrix = np.zeros((z_data_sim.size,3))
        matrix[:,0] = f_
        matrix[:,1] = np.absolute(z_data_sim)
        matrix[:,2] = np.angle(z_data_sim)
        
        np.savetxt(os.path.join(output_dir,'simdata.dat'),matrix)
        
        plt.subplots(2,1)
        plt.subplot(2,1,1)
        plt.title('raw data vs fitted data, Mag')
        plt.plot(f_,np.abs(S21_),'o-')
        plt.plot(f_data_origin,np.abs(z_data_origin),'.-',c='g')
        plt.plot(f_,np.abs(z_data_sim),c='r')
        #plt.xlim([f[np.argmin(np.abs(S21))]-0.0003,f[np.argmin(np.abs(S21))]+0.0003])
        
        plt.subplot(2,1,2)
        plt.title('raw data vs fitted data, phase')
        plt.plot(f_,np.angle(S21_),'.')
        plt.plot(f_data_origin,np.angle(z_data_origin),'.',c='g')
        plt.plot(f_,np.angle(z_data_sim),c='r')
        plt.figtext(0.,0.01,'Qi=%dpm%d\tQc=%dpm%d\tfr=%.5gpm%.5g'%(results_entire_model['Qi'],
                                                   results_entire_model['Qi_err'],
                                                   results_entire_model['Qc'],
                                                   results_entire_model['Qc_err'],
                                                   results_entire_model['fr'],
                                                   results_entire_model['fr_err'],
                                                   ),size ='large',color='r')
        
        timestamp_var = '_'.join(fname.split('_')[-2:])
        if 'GHzSpec_2ndtonePower' in fname:
            basename = os.path.basename(fname)
            freq_spec, power_spec = [float(basename.split('_')[4][0:-7]),
                            float(basename.split('_')[5].split('=')[1][0:-3])]
            plt.savefig(os.path.join(output_dir,'T=%dmK_CH1=%ddB_fr=%s_%.5fGHzSpec_2ndtonePower=%ddBm_%s.png'%(temperature,ch1atten,basename.split('_')[2][0:-6],freq_spec,power_spec,timestamp_var)))
        else:
            plt.savefig(os.path.join(output_dir,'T=%dmK_CH1=%ddB_fr=%.2f_Fitting.png'%(temperature,atten,results_entire_model['fr'])))
        
      #  pprint(results_entire_model)
        
       # print('average number of photons in resonator at %ddBm input power'%inputpower,avg_photon_num)
    #    print('single photon limit power at input port(dBm)',port.get_single_photon_limit(results_entire_model['fr'], results_entire_model['Ql'],results_entire_model['Qc']))
        
        results = [temperature,
                   atten,
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
                    fname
    ]
        aresults = '\t'.join(map(str,results))
        labels = 'T\t\
                  atten\t\
                  fr\t\
                  fr_err\t\
                  Qi\t\
                  Qi_err\t\
                  Qc\t\
                  Qc_err\t\
                  Ql\t\
                  Ql_err\t\
                  avg_photon_num\t\
                  inputpower\t\
                  circulation_power\t\
                  single_photon_power_at_input\t\
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
          
        with open(os.path.join(output_dir,'results.dat'),'w') as fh:
            fh.write(labels+'\n')
            fh.writelines(aresults)
        return results_entire_model
        
if __name__ =='__main__':
    OUTPUT_DIR = 'D:\\Dropbox\\Drive\\Projects\\Resonator\\MKID\\0.5um resonator'
    fdir = OUTPUT_DIR
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
#    try:
#        os.remove(os.path.join(OUTPUT_DIR,'fit.log'))
#    except :
#        pass
    
    handler = logging.FileHandler(os.path.join(OUTPUT_DIR,'fit.log'))      
    handler.setFormatter(formatter)
    logger = logging.getLogger('fit_log')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler) 
    
    input_file_list = []
    fn = 'hw_GND_Al_Res_AlOx_Length=162um_GNDstrip=2_Lsq=812_ResW=0.5um_ResG=3um_coupler=81um.csv'
    input_file_list.extend(glob.glob(os.path.join(fdir,fn)))
    input_file_list.sort(key = os.path.getmtime)
    input_file_list.reverse()
    
    if input_file_list !=[]:
        for fname in input_file_list:
            if os.path.getsize(fname)>5000:
                print('Processing', fname)
                temperature = float(fname.split('_')[-3].split('=')[-1][0:-2])
                print('Temperature is', temperature, 'mK')
                results_entire_model = fit_quality_factor(temperature,fname,OUTPUT_DIR,
                                               Delay = [0, 0],
#                                               f_range=[9.68,9.705], 
                                           REFIT=False,
                                           MEAS = False)
