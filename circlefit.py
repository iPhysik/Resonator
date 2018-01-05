
import numpy as np
import scipy.optimize as spopt
from scipy import stats
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class circlefit(object):
    '''
    contains all the circlefit procedures
    see http://scitation.aip.org/content/aip/journal/rsi/86/2/10.1063/1.4907935
    arxiv version: http://arxiv.org/abs/1410.3365
    '''
    def _remove_cable_delay(self,f_data,z_data, delay):
        return z_data/np.exp(2j*np.pi*f_data*delay)

    def _center(self,z_data,zc):
        return z_data-zc
    
    def _dist(self,x):
        np.absolute(x,x)
        c = (x > np.pi).astype(np.int)
        return x+c*(-2.*x+2.*np.pi)  
        
    def _periodic_boundary(self,x,bound):
        return np.fmod(x,bound)-np.trunc(x/bound)*bound
    def _weight(self,x,y):
        xref = (x[0]+x[-1])/2
        yref = (y[0]+y[-1])/2
        res = np.sqrt((x-xref)**2+(y-yref)**2)
#        res = res/res.max()
        res = np.ones(x.size)
        return res
    
    def _phase_fit_wslope(self,f_data,z_data,theta0, Ql, fr, slope):
        phase = np.angle(z_data)
        def residuals(p,x,y):
            theta0, Ql, fr, slope = p
            err = self._dist(y - (theta0+2.*np.arctan(2.*Ql*(1.-x/fr))-slope*(x)))
            return err
        p0 = [theta0, Ql, fr, slope]
        p_final = spopt.leastsq(residuals,p0,args=(np.array(f_data),np.array(phase)))
        return p_final[0]
    
    def _phase_fit(self,f_data,z_data,theta0, Ql, fr):
        phase = np.angle(z_data)
#        phase = np.unwrap(phase)
        xdat=z_data.real
        ydat = z_data.imag
        def residuals_1(p,x,y,Ql):
            theta0, fr = p
            err = self._dist(y - (theta0+2.*np.arctan(2.*Ql*(1.-x/fr))))
            return err
        def residuals_2(p,x,y,theta0):
            Ql, fr = p
            err = self._dist(y - (theta0+2.*np.arctan(2.*Ql*(1.-x/fr))))
            return err
        def residuals_3(p,x,y,theta0,Ql):
            fr = p
            err = self._dist(y - (theta0+2.*np.arctan(2.*Ql*(1.-x/fr))))
            return err
        def residuals_4(p,x,y,theta0,fr):
            Ql = p
            err = self._dist(y - (theta0+2.*np.arctan(2.*Ql*(1.-x/fr))))
            return err
        def residuals_5(p,x,y):
            theta0, Ql, fr = p
            err = self._dist(y - (theta0+2.*np.arctan(2.*Ql*(1.-x/fr))))
            err = err * self._weight(xdat,ydat)
            return err
        
        p0 = [theta0, Ql, fr]
        p_final = spopt.leastsq(residuals_5,p0,args=(f_data,phase))
        return p_final[0]
    
    def _fit_skewed_lorentzian(self,f_data,z_data):
        amplitude = np.absolute(z_data)
        amplitude_sqr = amplitude**2
        A1a = np.minimum(amplitude_sqr[0],amplitude_sqr[-1])
        A3a = -np.max(amplitude_sqr)
        fra = f_data[np.argmin(amplitude_sqr)]
        f = f_data
        
        def residuals(p,x,y):
            A2, A4, Ql = p
            err = y -(A1a+A2*(x-fra)+(A3a+A4*(x-fra))/(1.+4.*Ql**2*((x-fra)/fra)**2))
            return err
        p0 = [0., 0., 2e3]
        p_final = spopt.leastsq(residuals,p0,args=(np.array(f_data),np.array(amplitude_sqr)))
        A2a, A4a, Qla = p_final[0]
        
#        plt.subplots()
#        plt.title('skewed lorentzian')
#        plt.plot(f,amplitude_sqr, '.', f,(A1a+A2a*(f-fra)+(A3a+A4a*(f-fra))/(1.+4.*Qla**2*((f-fra)/fra)**2)) )
        
        def residuals2(p,x,y):
            A1, A2, A3, A4, fr, Ql = p
            err = y -(A1+A2*(x-fr)+(A3+A4*(x-fr))/(1.+4.*Ql**2*((x-fr)/fr)**2))
            return err
        p0 = [A1a, A2a , A3a, A4a, fra, Qla]
        p_final = spopt.leastsq(residuals2,p0,args=(np.array(f_data),np.array(amplitude_sqr)))
        A1, A2, A3, A4, fr, Ql = p_final[0]
        #print p_final[0][5]
        
        plt.subplots()
        plt.title('skewed lorentzian')
        plt.plot(f,amplitude_sqr, '.', f,(A1+A2*(f-fr)+(A3+A4*(f-fr))/(1.+4.*Ql**2*((f-fr)/fr)**2)) )
        return p_final[0]
    
    def _fit_circle(self,z_data, refine_results=False):
        def calc_moments(z_data):
            xi = z_data.real
            xi_sqr = xi*xi
            yi = z_data.imag
            yi_sqr = yi*yi
            zi = xi_sqr+yi_sqr
            Nd = float(len(xi))
            xi_sum = xi.sum()
            yi_sum = yi.sum()
            zi_sum = zi.sum()
            xiyi_sum = (xi*yi).sum()
            xizi_sum = (xi*zi).sum()
            yizi_sum = (yi*zi).sum()
            return np.array([ [(zi*zi).sum(), xizi_sum, yizi_sum, zi_sum],  \
            [xizi_sum, xi_sqr.sum(), xiyi_sum, xi_sum], \
            [yizi_sum, xiyi_sum, yi_sqr.sum(), yi_sum], \
            [zi_sum, xi_sum, yi_sum, Nd] ])
    
        M = calc_moments(z_data)
    
        a0 = ((M[2][0]*M[3][2]-M[2][2]*M[3][0])*M[1][1]-M[1][2]*M[2][0]*M[3][1]-M[1][0]*M[2][1]*M[3][2]+M[1][0]*M[2][2]*M[3][1]+M[1][2]*M[2][1]*M[3][0])*M[0][3]+(M[0][2]*M[2][3]*M[3][0]-M[0][2]*M[2][0]*M[3][3]+M[0][0]*M[2][2]*M[3][3]-M[0][0]*M[2][3]*M[3][2])*M[1][1]+(M[0][1]*M[1][3]*M[3][0]-M[0][1]*M[1][0]*M[3][3]-M[0][0]*M[1][3]*M[3][1])*M[2][2]+(-M[0][1]*M[1][2]*M[2][3]-M[0][2]*M[1][3]*M[2][1])*M[3][0]+((M[2][3]*M[3][1]-M[2][1]*M[3][3])*M[1][2]+M[2][1]*M[3][2]*M[1][3])*M[0][0]+(M[1][0]*M[2][3]*M[3][2]+M[2][0]*(M[1][2]*M[3][3]-M[1][3]*M[3][2]))*M[0][1]+((M[2][1]*M[3][3]-M[2][3]*M[3][1])*M[1][0]+M[1][3]*M[2][0]*M[3][1])*M[0][2]
        a1 = (((M[3][0]-2.*M[2][2])*M[1][1]-M[1][0]*M[3][1]+M[2][2]*M[3][0]+2.*M[1][2]*M[2][1]-M[2][0]*M[3][2])*M[0][3]+(2.*M[2][0]*M[3][2]-M[0][0]*M[3][3]-2.*M[2][2]*M[3][0]+2.*M[0][2]*M[2][3])*M[1][1]+(-M[0][0]*M[3][3]+2.*M[0][1]*M[1][3]+2.*M[1][0]*M[3][1])*M[2][2]+(-M[0][1]*M[1][3]+2.*M[1][2]*M[2][1]-M[0][2]*M[2][3])*M[3][0]+(M[1][3]*M[3][1]+M[2][3]*M[3][2])*M[0][0]+(M[1][0]*M[3][3]-2.*M[1][2]*M[2][3])*M[0][1]+(M[2][0]*M[3][3]-2.*M[1][3]*M[2][1])*M[0][2]-2.*M[1][2]*M[2][0]*M[3][1]-2.*M[1][0]*M[2][1]*M[3][2])
        a2 = ((2.*M[1][1]-M[3][0]+2.*M[2][2])*M[0][3]+(2.*M[3][0]-4.*M[2][2])*M[1][1]-2.*M[2][0]*M[3][2]+2.*M[2][2]*M[3][0]+M[0][0]*M[3][3]+4.*M[1][2]*M[2][1]-2.*M[0][1]*M[1][3]-2.*M[1][0]*M[3][1]-2.*M[0][2]*M[2][3])
        a3 = (-2.*M[3][0]+4.*M[1][1]+4.*M[2][2]-2.*M[0][3])
        a4 = -4.
    
        def func(x):
            return a0+a1*x+a2*x*x+a3*x*x*x+a4*x*x*x*x
    
        def d_func(x):
            return a1+2*a2*x+3*a3*x*x+4*a4*x*x*x
    
        x0 = spopt.fsolve(func, 0., fprime=d_func)
    
        def solve_eq_sys(val,M):
            #prepare
            M[3][0] = M[3][0]+2*val
            M[0][3] = M[0][3]+2*val
            M[1][1] = M[1][1]-val
            M[2][2] = M[2][2]-val
            return np.linalg.svd(M)
    
        U,s,Vt = solve_eq_sys(x0[0],M)
    
        A_vec = Vt[np.argmin(s),:]
    
        xc = -A_vec[1]/(2.*A_vec[0])
        yc = -A_vec[2]/(2.*A_vec[0])
        # the term *sqrt term corrects for the constraint, because it may be altered due to numerical inaccuracies during calculation
        r0 = 1./(2.*np.absolute(A_vec[0]))*np.sqrt(A_vec[1]*A_vec[1]+A_vec[2]*A_vec[2]-4.*A_vec[0]*A_vec[3])
        if refine_results:
            print ("agebraic r0: " + str(r0))
            xc,yc,r0 = self._fit_circle_iter(z_data, xc, yc, r0)
            r0 = self._fit_circle_iter_radialweight(z_data, xc, yc, r0)
#            r0 = self._fit_circle_iter_radialweight(z_data, ((z_data[0]+z_data[-1])/2).real, ((z_data[0]+z_data[-1])/2).imag, r0)
            print ("iterative r0: " + str(r0))
        return xc, yc, r0

    def _guess_delay(self,f_data,z_data,f_min,f_max,PLOT=True):
        index_min = np.argmin(np.abs(f_data-f_min))
        index_max = np.argmin(np.abs(f_data-f_max))
        index = np.arange(index_min,index_max+1)
        
        phase2 = np.unwrap(np.angle(z_data))[index]
        f_data = f_data[index]
        gradient, intercept, r_value, p_value, std_err = stats.linregress(f_data,phase2)
        if PLOT:
            plt.subplots()
            plt.title('linear delay fit, slope={} pm {}'.format(gradient*(-1.)/(np.pi*2.),std_err*(1.)/(np.pi*2.)))
            plt.plot(f_data,phase2,'.')
            plt.plot(np.linspace(f_data.min(),f_data.max(),1000),gradient*np.linspace(f_data.min(),f_data.max(),1000)+intercept)
        gradient= gradient*(-1.)/(np.pi*2.)
        std_err = std_err*(1.)/(np.pi*2.)
        return gradient,std_err
    
    
    def _fit_delay(self,f_data,z_data,delay=0.,maxiter=0):
        def residuals(p,x,y):
            phasedelay = p
            z_data_temp = y*np.exp(1j*(2.*np.pi*phasedelay*x))
            xc,yc,r0 = self._fit_circle(z_data_temp)
            err = np.sqrt((z_data_temp.real-xc)**2+(z_data_temp.imag-yc)**2)-r0
            return err
        p_final = spopt.leastsq(residuals,delay,args=(f_data,z_data),maxfev=maxiter,ftol=1e-12,xtol=1e-12)
        return p_final[0][0]
    
    def _fit_delay_and_linear_var_in_S21(self,f_data,z_data,fr,delay=0,alpha=0,maxiter=0):
        """
        reference : supplementary material Appl. Phys. Lett. 106, 182601 (2015)
        """
        def residuals_1(p,x,y):
            alpha, phasedelay = p
            z_data_temp = y/(1+alpha*(x/fr-1))
            phasedelay= self._fit_delay(f_data,z_data,phasedelay,maxiter=200)
            z_data_temp = y*np.exp(1j*(2.*np.pi*phasedelay*x))
            xc,yc,r0 = self._fit_circle(z_data_temp)
            err = np.sqrt((z_data_temp.real-xc)**2+(z_data_temp.imag-yc)**2)-r0
            return err
        
        def residuals_2(p,x,y):
            alpha = p
            z_data_temp = y*np.exp(1j*(2.*np.pi*delay*x))/(1+alpha*(x/fr-1))
            xc,yc,r0 = self._fit_circle(z_data_temp)
            err = np.sqrt((z_data_temp.real-xc)**2+(z_data_temp.imag-yc)**2)-r0
            return err
        
        p_init = [alpha]
        p_final = spopt.leastsq(residuals_2,p_init,args=(f_data,z_data),maxfev=maxiter,ftol=1e-12,xtol=1e-12)
        alpha = p_final[0][0]

#        p_init = [alpha,delay]
#        p_final = spopt.leastsq(residuals_1,p_init,args=(f_data,z_data),maxfev=1000,ftol=1e-12,xtol=1e-12)
#        print(p_final)
        return p_final[0]
    
    def _fit_delay_alt_bigdata(self,f_data,z_data,delay=0.,maxiter=100):
        def residuals(p,x,y):
            phasedelay = p
            z_data_temp = 1j*2.*np.pi*phasedelay*x
            np.exp(z_data_temp,out=z_data_temp)
            np.multiply(y,z_data_temp,out=z_data_temp)
            #z_data_temp = y*np.exp(1j*(2.*np.pi*phasedelay*x))
            xc,yc,r0 = self._fit_circle(z_data_temp)
            err = np.sqrt((z_data_temp.real-xc)**2+(z_data_temp.imag-yc)**2)-r0
            return err
        p_final = spopt.leastsq(residuals,delay,args=(f_data,z_data),maxfev=maxiter,ftol=1e-12,xtol=1e-12)
        return p_final[0][0]
    
    def _fit_entire_model(self,f_data,z_data,fr,absQc,Ql,phi0,delay,a=1.,alpha=0.,ftol=1e-10,xtol=1e-10,maxfev=2000):
        '''
        fits the whole model: a*exp(i*alpha)*exp(-2*pi*i*f*delay) * [ 1 - {Ql/Qc*exp(i*phi0)} / {1+2*i*Ql*(f-fr)/fr} ]
        '''
        def residuals(p,x,y):
            fr,absQc,Ql,phi0,delay,a,alpha = p
            err = [np.absolute( y[i] - ( a*np.exp(np.complex(0,alpha))*np.exp(np.complex(0,-2.*np.pi*delay*x[i])) * ( 1 - (Ql/absQc*np.exp(np.complex(0,phi0)))/(np.complex(1,2*Ql*(x[i]-fr)/fr)) )  ) ) for i in range(len(x))]
            return np.array(err)

        p0 = [fr,absQc,Ql,phi0,delay,a,alpha]
        (popt, params_cov, infodict, errmsg, ier) = spopt.leastsq(residuals,p0,args=(np.array(f_data),np.array(z_data)),full_output=True,ftol=ftol,xtol=xtol,maxfev=maxfev) # if ftol or xtol are set too small error will return
#        print('params_cov', params_cov)
        len_ydata = len(np.array(f_data))
        if (len_ydata > len(p0)) and params_cov is not None:  #p_final[1] is cov_x data  #this caculation is from scipy curve_fit routine - no idea if this works correctly...
            s_sq = (residuals(popt, np.array(f_data),np.array(z_data))**2).sum()/(len_ydata-len(p0))
            pcov = params_cov * s_sq
        else:
            pcov = np.inf
        
        chi = residuals(p0,f_data,z_data)
        res = (chi**2).sum() # calculate residuals
        
        return popt, pcov, res, infodict, errmsg, ier
    
    def _fit_entire_model_except_delay(self,f_data,z_data,fr,absQc,Ql,phi0,a=1.,alpha=0.,ftol=1e-10,xtol=1e-10,maxfev=2000):
        '''
        fits the whole model: a*exp(i*alpha) * [ 1 - {Ql/Qc*exp(i*phi0)} / {1+2*i*Ql*(f-fr)/fr} ]
        '''
        def residuals(p,x,y):
            fr,absQc,Ql,phi0,a,alpha = p
            err = [np.absolute( y[i] - ( a*np.exp(np.complex(0,alpha)) * ( 1 - (Ql/absQc*np.exp(np.complex(0,phi0)))/(np.complex(1,2*Ql*(x[i]-fr)/fr)) )  ) ) for i in range(len(x))]
            return np.array(err)

        p0 = [fr,absQc,Ql,phi0,a,alpha]
        (popt, params_cov, infodict, errmsg, ier) = spopt.leastsq(residuals,p0,args=(np.array(f_data),np.array(z_data)),full_output=True,ftol=ftol,xtol=xtol,maxfev=maxfev) # if ftol or xtol are set too small error will return
#        print('params_cov', params_cov)
        len_ydata = len(np.array(f_data))
        if (len_ydata > len(p0)) and params_cov is not None:  #p_final[1] is cov_x data  #this caculation is from scipy curve_fit routine - no idea if this works correctly...
            s_sq = (residuals(popt, np.array(f_data),np.array(z_data))**2).sum()/(len_ydata-len(p0))
            pcov = params_cov * s_sq
        else:
            pcov = np.inf
        
        chi = residuals(p0,f_data,z_data)
        res = (chi**2).sum()
        
        return popt, pcov, res, infodict, errmsg, ier    
    
    def _optimizedelay(self,f_data,z_data,Ql,fr,maxiter=4):
        xc,yc,r0 = self._fit_circle(z_data)
        z_data = self._center(z_data,np.complex(xc,yc))
#        plt.subplots()
#        plt.title('optimize delay')
#        plt.plot(f_data,np.angle(z_data))
        theta, Ql, fr, slope = self._phase_fit_wslope(f_data,z_data,0.,Ql,fr,0.)
        delay = 0.
        for i in range(maxiter-1): #interate to get besser phase delay term
            delay = delay - slope/(2.*np.pi)
            z_data_corr = self._remove_cable_delay(f_data,z_data,delay)
            xc, yc, r0 = self._fit_circle(z_data_corr)
            z_data_corr2 = self._center(z_data_corr,np.complex(xc,yc))
            theta0, Ql, fr, slope = self._phase_fit_wslope(f_data,z_data_corr2,0.,Ql,fr,0.)
        delay = delay - slope/(2.*np.pi)  #start final interation
        return delay
    
    def _fit_circle_iter(self,z_data, xc, yc, rc):
        '''
        this is the radial weighting procedure
        it improves your fitting value for the radius = Ql/Qc
        use this to improve your fit in presence of heavy noise
        after having used the standard algebraic fir_circle() function
        the weight here is: W=1/sqrt((xc-xi)^2+(yc-yi)^2)
        this works, because the center of the circle is usually much less
        corrupted by noise than the radius
        '''
#        xdat = z_data.real
#        ydat = z_data.imag
#        def fitfunc(x,y,xc,yc):
#            return np.sqrt((x-xc)**2+(y-yc)**2)
#        def residuals(p,x,y):
#            xc,yc,r = p
#            temp = (r-fitfunc(x,y,xc,yc))
#            return temp
#        p0 = [xc,yc,rc]
#        p_final = spopt.leastsq(residuals,p0,args=(xdat,ydat))
#        xc,yc,rc = p_final[0]
#        return xc,yc,rc
        
        xdat = z_data.real
        ydat = z_data.imag
        def fitfunc(x,y,xc,yc):
            return np.sqrt((x-xc)**2+(y-yc)**2)
      
        def residuals(p,x,y):
            xc,yc,r = p
            temp = (r-fitfunc(x,y,xc,yc))*self._weight(xdat,ydat)
            return temp
        
        p0 = [xc,yc,rc]
        p_final = spopt.leastsq(residuals,p0,args=(xdat,ydat))
        xc,yc,rc = p_final[0]
        return xc,yc,rc    

    def _fit_circle_iter_radialweight(self,z_data, xc, yc, rc):
        '''
        this is the radial weighting procedure
        it improves your fitting value for the radius = Ql/Qc
        use this to improve your fit in presence of heavy noise
        after having used the standard algebraic fir_circle() function
        the weight here is: W=1/sqrt((xc-xi)^2+(yc-yi)^2)
        this works, because the center of the circle is usually much less
        corrupted by noise than the radius
        '''
        xdat = z_data.real
        ydat = z_data.imag
        def fitfunc(x,y):
            return np.sqrt((x-xc)**2+(y-yc)**2)
#        def weight(x,y):
#            try:
#                res = 1./np.sqrt((xc-x)**2+(yc-y)**2)
#            except:
#                res = 1.
#            return res
        def weight(x,y):
            xref = (x[0]+x[-1])/2
            yref = (y[0]+y[-1])/2
            res = ((x-xref)**2+(y-yref)**2)
            return res/res.max()
        def residuals(p,x,y):
            r = p[0]
            temp = (r-fitfunc(x,y))*self._weight(xdat,ydat)
            return temp
        p0 = [rc]
        p_final = spopt.leastsq(residuals,p0,args=(xdat,ydat))
        return p_final[0][0]
    def _get_cov(self,xdata,ydata, fitparams):
        """
        fitparams: fr,absQc,Ql,phi0, a, alpha, delay
        """
        pass
      
    
        
    def _get_errors(self,residual,xdata,ydata,fitparams):
        '''
        wrapper for get_cov, only gives the errors and chisquare
        '''
        chisqr, cov = self._get_cov(xdata,ydata,fitparams)
        if cov!=None:
            errors = np.sqrt(np.diagonal(cov))
        else:
            errors = None
        return chisqr, errors
    
    def _residuals_notch_full(self,p,x,y):
        fr,absQc,Ql,phi0,delay,a,alpha = p
        err = np.absolute( y - ( a*np.exp(np.complex(0,alpha))*np.exp(np.complex(0,-2.*np.pi*delay*x)) * ( 1 - (Ql/absQc*np.exp(np.complex(0,phi0)))/(np.complex(1,2*Ql*(x-fr)/float(fr))) )  ) )
        return err
    
    def _residuals_notch_ideal(self,p,x,y):
        fr,absQc,Ql,phi0 = p
        #if fr == 0: print p
        err = np.absolute( y - (  ( 1. - (Ql/float(absQc)*np.exp(1j*phi0))/(1+2j*Ql*(x-fr)/float(fr)) )  ) )
        #if np.isinf((np.complex(1,2*Ql*(x-fr)/float(fr))).imag):
         #   print np.complex(1,2*Ql*(x-fr)/float(fr))
          #  print "x: " + str(x)
           # print "Ql: " +str(Ql)
            #print "fr: " +str(fr)
        return err
    
    def _residuals_notch_ideal_complex(self,p,x,y):
        fr,absQc,Ql,phi0 = p
        #if fr == 0: print p
        err = y - (  ( 1. - (Ql/float(absQc)*np.exp(1j*phi0))/(1+2j*Ql*(x-fr)/float(fr)) )  )
        #if np.isinf((np.complex(1,2*Ql*(x-fr)/float(fr))).imag):
         #   print np.complex(1,2*Ql*(x-fr)/float(fr))
          #  print "x: " + str(x)
           # print "Ql: " +str(Ql)
            #print "fr: " +str(fr)
        return err
        
    def _residuals_directrefl(self,p,x,y):
        fr,Qc,Ql = p
        #if fr == 0: print p
        err = y - ( 2.*Ql/Qc - 1. + 2j*Ql*(fr-x)/fr ) / ( 1. - 2j*Ql*(fr-x)/fr )
        #if np.isinf((np.complex(1,2*Ql*(x-fr)/float(fr))).imag):
         #   print np.complex(1,2*Ql*(x-fr)/float(fr))
          #  print "x: " + str(x)
           # print "Ql: " +str(Ql)
            #print "fr: " +str(fr)
        return err
    
    def _residuals_transm_ideal(self,p,x,y):
        fr,Ql = p
        err = np.absolute( y -   ( 1./(np.complex(1,2*Ql*(x-fr)/float(fr))) )   )
        return err
    
    
    def _get_cov_fast_notch(self,xdata,ydata,fitparams): #enhanced by analytical derivatives
        #derivatives of notch_ideal model with respect to parameters
        def dS21_dQl(p,f):
            fr,absQc,Ql,phi0 = p
            return - (np.exp(1j*phi0) * fr**2) / (absQc * (fr+2j*Ql*f-2j*Ql*fr)**2 )
    
        def dS21_dQc(p,f):
            fr,absQc,Ql,phi0 = p
            return (np.exp(1j*phi0) * Ql*fr) / (2j*(f-fr)*absQc**2*Ql+absQc**2*fr )
    
        def dS21_dphi0(p,f):
            fr,absQc,Ql,phi0 = p
            return - (1j*Ql*fr*np.exp(1j*phi0) ) / (2j*(f-fr)*absQc*Ql+absQc*fr )
    
        def dS21_dfr(p,f):
            fr,absQc,Ql,phi0 = p
            return - (2j*Ql**2*f*np.exp(1j*phi0) ) / (absQc * (fr+2j*Ql*f-2j*Ql*fr)**2 )
    
        u = self._residuals_notch_ideal_complex(fitparams,xdata,ydata)
        chi = np.absolute(u)
        u = u/chi  # unit vector pointing in the correct direction for the derivative
    
        aa = dS21_dfr(fitparams,xdata)
        bb = dS21_dQc(fitparams,xdata)
        cc = dS21_dQl(fitparams,xdata)
        dd = dS21_dphi0(fitparams,xdata)
    
        Jt = np.array([aa.real*u.real+aa.imag*u.imag, bb.real*u.real+bb.imag*u.imag\
                , cc.real*u.real+cc.imag*u.imag, dd.real*u.real+dd.imag*u.imag  ])
        A = np.dot(Jt,np.transpose(Jt))
        chisqr = 1./float(len(xdata)-len(fitparams)) * (chi**2).sum()
        try:
            cov = np.linalg.inv(A)*chisqr
        except:
            cov = None
        return chisqr, cov
        
    def _get_cov_fast_directrefl(self,xdata,ydata,fitparams): #enhanced by analytical derivatives
        #derivatives of notch_ideal model with respect to parameters
        def dS21_dQl(p,f):
            fr,Qc,Ql = p
            return 2.*fr**2/( Qc*(2j*Ql*fr-2j*Ql*f+fr)**2 )
    
        def dS21_dQc(p,f):
            fr,Qc,Ql = p
            return 2.*Ql*fr/(2j*Qc**2*(f-fr)*Ql-Qc**2*fr)
    
        def dS21_dfr(p,f):
            fr,Qc,Ql = p
            return - 4j*Ql**2*f/(Qc*(2j*Ql*fr-2j*Ql*f+fr)**2)
    
        u = self._residuals_directrefl(fitparams,xdata,ydata)
        chi = np.absolute(u)
        u = u/chi  # unit vector pointing in the correct direction for the derivative
    
        aa = dS21_dfr(fitparams,xdata)
        bb = dS21_dQc(fitparams,xdata)
        cc = dS21_dQl(fitparams,xdata)
    
        Jt = np.array([aa.real*u.real+aa.imag*u.imag, bb.real*u.real+bb.imag*u.imag\
                , cc.real*u.real+cc.imag*u.imag  ])
        A = np.dot(Jt,np.transpose(Jt))
        chisqr = 1./float(len(xdata)-len(fitparams)) * (chi**2).sum()
        try:
            cov = np.linalg.inv(A)*chisqr
        except:
            cov = None
        return chisqr, cov