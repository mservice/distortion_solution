from astropy.table import Table
from astropy.table import join
from jlu.astrometry.align  import jay  
import numpy as np
from jlu.astrometry import high_order_class as high_order
import matplotlib.pyplot as plt
from scipy.interpolate import SmoothBivariateSpline as Spline
from scipy.interpolate import LSQBivariateSpline as SplineLSQ
from astropy.io import fits
import os
from astropy.modeling import models, fitting
import math
import pyfits
import datetime
from nirc2.reduce.dar import *
from jlu.util import statsIter
import pickle
from distortion_solution import match_trim
from scipy.stats import f as fdist
from scipy.integrate import quad



def fit_dist(pos_txt='april_pos.txt', order=6, n_iter_fit=10, nstring=None, lookup=True, poly_type=1, wtype=1):
    '''
    fits distortion with legendre polynomial
    weights by the errors added in quadrature for each point (both reerence and Nirc2 errors)
    wtype =1 is both reference and data positional errors
    wtype=2 is only the data errors as weights
    '''

    if nstring == None:
        nstring = 'Leg_'+str(order)+'_'
    tab = Table.read(pos_txt, format='ascii.fixed_width')
    

    #x, xerr, y, yerr, xr, xrerr, yr, yrerr, name, mag = sig_trim(tab['x'], tab['xerr'], tab['y'], tab['yerr'], tab['xr'], tab['xrerr'], tab['yr'], tab['yrerr'], tab['name'], tab['mag'])
    if wtype==1:
        weights = 1 / (tab['xerr']**2 + tab['yerr']**2 + tab['xrerr']**2 + tab['yrerr']**2)**0.5
    elif wtype == 2:
        weights = 1 / (tab['xerr']**2 + tab['yerr']**2)**0.5
    weights = weights / np.mean(weights)
    gbool = ( tab['xerr'] > 10**-10) * ( tab['yerr'] > 10**-10) * ( tab['xrerr'] > 10**-10) * ( tab['yrerr'] > 10**-10)
    print 'fitting ', np.sum(gbool), ' stars after zero error cut'
    t, dx, dy, sbool = calc_sig_dist_leg( tab['x'][gbool],  tab['y'][gbool],  tab['xr'][gbool] ,  tab['yr'][gbool], weights=weights[gbool], n_iter=n_iter_fit, order=order, ret_bool=True, poly_type=poly_type)
    if lookup:
        outx, outy = match_trim.leg2lookup(t)
        return t, outx, outy, dx, dy, sbool
    else:
        return t, dx, dy, gbool, sbool
    
  
def calc_sig_dist_leg(x, y,xref,yref, n_iter=3, sig_clip=3, order=3, weights=None, ret_bool=False, poly_type=1):


    if poly_type == 1:
        trans = high_order.LegTransform
    elif poly_type==2:
        trans = high_order.PolyTransform
    #now sigma clip
    sbool = np.ones(x.shape,dtype='bool')
    for i in range(n_iter):
        if weights != None:
            t1 = trans(x[sbool],y[sbool],xref[sbool]-x[sbool],yref[sbool]-y[sbool], order, weights=weights[sbool])
        else:
            t1 = trans(x[sbool],y[sbool],xref[sbool]-x[sbool],yref[sbool]-y[sbool], order)
        xout,yout = t1.evaluate(x,y)
        dx = xref - xout- x
        dy = yref - yout- y
        xcen = np.mean(dx[sbool])
        xsig = np.std(dx[sbool])
        ycen = np.mean(dy[sbool])
        ysig = np.std(dy[sbool])
        if i != n_iter-1: 
            sbool = (dx  > xcen - sig_clip * xsig)*(dx < xcen + sig_clip * xsig)*(dy > ycen - sig_clip * ysig)*(dy < ycen + sig_clip * ysig)
        
        print 'trimmed ',len(sbool)-np.sum(sbool), '  stars'
   
    if ret_bool:
        return t1 , dx,dy, sbool
    else:
        return t1, dx, dy


def fit_dist_spline(pos_txt='april_pos.txt',  n_iter_fit=10, nstring=None, s=200, lookup=True, wtype=1, bool_in=None):
    '''
    fits distortion with legendre polynomial
    does spatial sigma clipping before hand
    weights by the errors added in quadrature for each point (both reerence and Nirc2 errors)
    '''


    tab = Table.read(pos_txt, format='ascii.fixed_width')
    if  bool_in == None:
        bool_in = np.ones(len(tab['x']), dtype='bool')

    #x, xerr, y, yerr, xr, xrerr, yr, yrerr, name, mag = sig_trim(tab['x'], tab['xerr'], tab['y'], tab['yerr'], tab['xr'], tab['xrerr'], tab['yr'], tab['yrerr'], tab['name'], tab['mag'])
    if wtype == 1:
        weights = 1 / (tab['xerr']**2 + tab['yerr']**2 + tab['xrerr']**2 +  tab['yrerr']**2)**0.5
        wx = weights
        wy = weights
    elif wtype == 2:
        weights = 1 / (tab['xerr']**2 + tab['yerr']**2)**0.5
        wx = weights
        wy = weights
    elif wtype==3:
        weights= np.ones(len(tab['xerr']))
        wx = weights
        wy =  weights
            
    wx = wx / np.mean(wx)
    wy = wy / np.mean(wy)
        
    gbool = ( tab['xerr'] > 10**-10) * ( tab['yerr'] > 10**-10) * ( tab['xrerr'] > 10**-10) * ( tab['yrerr'] > 10**-10)
    gbool = gbool * bool_in
    print 'fitting ', np.sum(gbool), ' stars after zero error cut'
    tx,ty, dx, dy, sbool = calc_sig_dist_spline( tab['x'][gbool],  tab['y'][gbool],  tab['xr'][gbool] ,  tab['yr'][gbool], wx=wx[gbool], wy=wy[gbool], n_iter=n_iter_fit,smooth_fac=s, ret_bool=True)
    if lookup:
        outx, outy = spline2lookup(tx,ty)

    

        return tx,ty,outx,outy, dx, dy, gbool, sbool
    else:
        return tx,ty, dx, dy,gbool,sbool


def calc_sig_dist_spline(x, y,xref,yref, n_iter=5, sig_clip=3, plot=False, smooth_fac=10000, wx=None,wy=None,ret_bool=False):

   
    #now sigma clip
    sbool = np.ones(x.shape,dtype='bool')
    for i in range(n_iter):
        #import pdb; pdb.set_trace()
        tx = Spline(x[sbool],y[sbool],xref[sbool]-x[sbool], s=smooth_fac, w=wx[sbool])
        ty = Spline(x[sbool],y[sbool],yref[sbool]-y[sbool], s=smooth_fac, w=wy[sbool])
        xout= tx.ev(x,y)
        yout = ty.ev(x,y)
        dx = xref - xout- x
        dy = yref - yout- y
        xcen = np.mean(dx[sbool])
        xsig = np.std(dx[sbool])
        ycen = np.mean(dy[sbool])
        ysig = np.std(dy[sbool])
        sbool_temp = (dx  > xcen - sig_clip * xsig)*(dx < xcen + sig_clip * xsig)*(dy > ycen - sig_clip * ysig)*(dy < ycen + sig_clip * ysig)
        if i != n_iter-1:
            sbool = sbool_temp * sbool
        print 'trimmed ',len(sbool)-np.sum(sbool), '  stars'
        if plot:
            print 'number of residuals outside of -5,5', np.sum((dx>5)+(dx<-5)+(dy<-5)+(dy>5))
            plt.figure(35)
            plt.subplot(121)
            plt.hist(dx, bins=100, range=(-5,5))
            plt.title('X residual to fit')
            plt.subplot(122)
            plt.hist(dy, bins=100, range=(-5,5))
            plt.title('Y residual to fit')
            plt.show()
            

    if not ret_bool:
        return tx,ty,dx,dy
    else:
        return tx,ty,dx,dy, sbool


def fit_dist_splineLSQ(pos_txt='april_pos.txt',  n_iter_fit=1, nstring=None, nknots=6,order=3 ,lookup=True, wtype=1):
    '''
    fits distortion with legendre polynomial
    weights by the errors added in quadrature for each point (both reerence and Nirc2 errors)
    '''

    
    tab = Table.read(pos_txt, format='ascii.fixed_width')
    

    #x, xerr, y, yerr, xr, xrerr, yr, yrerr, name, mag = sig_trim(tab['x'], tab['xerr'], tab['y'], tab['yerr'], tab['xr'], tab['xrerr'], tab['yr'], tab['yrerr'], tab['name'], tab['mag'])

    if wtype ==1:
        weights = 1 / (tab['xerr']**2 + tab['yerr']**2 + tab['xrerr']**2 +  tab['yrerr']**2)**0.5
        w=weights/np.mean(weights)
    elif wtype==2:
        weights = 1 / (tab['xerr']**2 + tab['yerr']**2 )**0.5
        w=weights/np.mean(weights)

    gbool = ( tab['xerr'] > 10**-10) * ( tab['yerr'] > 10**-10) * ( tab['xrerr'] > 10**-10) * ( tab['yrerr'] > 10**-10)

    print 'fitting ', np.sum(gbool), ' stars after zero error cut'
    tx,ty, dx, dy, sbool = calc_sig_dist_splineLSQ( tab['x'][gbool],  tab['y'][gbool],  tab['xr'][gbool] ,  tab['yr'][gbool], w=w[gbool], n_iter=n_iter_fit, ret_bool=True, order=order, nknots=nknots)
    if lookup:
        outx, outy = spline2lookup(tx,ty)

    

        return tx,ty,outx,outy, dx, dy, gbool, sbool
    else:
        return tx,ty, dx, dy,gbool,sbool

def calc_sig_dist_splineLSQ(x, y,xref,yref, n_iter=5, sig_clip=3, plot=False, nknots=6, w=None,ret_bool=False, order=3):

   
    #now sigma clip
    sbool = np.ones(x.shape,dtype='bool')
    kx = np.linspace(np.min(x), np.max(x), num=nknots)
    ky = np.linspace(np.min(y), np.max(y), num=nknots)
    for i in range(n_iter):
        #import pdb; pdb.set_trace()
        tx = SplineLSQ(x[sbool],y[sbool],xref[sbool]-x[sbool], kx,ky, w=w[sbool],kx=order,ky=order )
        ty = SplineLSQ(x[sbool],y[sbool],yref[sbool]-y[sbool], kx,ky, w=w[sbool],kx=order,ky=order)
        xout= tx.ev(x,y)
        yout = ty.ev(x,y)
        dx = xref - xout- x
        dy = yref - yout- y
        xcen = np.mean(dx[sbool])
        xsig = np.std(dx[sbool])
        ycen = np.mean(dy[sbool])
        ysig = np.std(dy[sbool])
        sbool_temp = (dx  > xcen - sig_clip * xsig)*(dx < xcen + sig_clip * xsig)*(dy > ycen - sig_clip * ysig)*(dy < ycen + sig_clip * ysig)
        if i != n_iter-1:
            sbool = sbool_temp * sbool
        print 'trimmed ',len(sbool)-np.sum(sbool), '  stars'
        if plot:
            print 'number of residuals outside of -5,5', np.sum((dx>5)+(dx<-5)+(dy<-5)+(dy>5))
            plt.figure(35)
            plt.subplot(121)
            plt.hist(dx, bins=100, range=(-5,5))
            plt.title('X residual to fit')
            plt.subplot(122)
            plt.hist(dy, bins=100, range=(-5,5))
            plt.title('Y residual to fit')
            plt.show()
            

    if not ret_bool:
        return tx,ty,dx,dy
    else:
        return tx,ty,dx,dy, sbool


def search_smooth_spline(pos_txt='april_pos.txt'):
    '''
    searches through potential smoothing factors, makes plots
    '''

    tab = Table.read(pos_txt, format='ascii.fixed_width')
    

    x, xerr, y, yerr, xr, xrerr, yr, yrerr, name, mag = sig_trim(tab['x'], tab['xerr'], tab['y'], tab['yerr'], tab['xr'], tab['xrerr'], tab['yr'], tab['yrerr'], tab['name'], tab['mag'])
    weights = 1 / (xerr**2 + yerr**2 + xrerr**2 + yrerr**2)**0.5
    weights = weights / np.mean(weights)
    dx_p = np.ones(len(x))
    dy_p = np.ones(len(y))
    smooth_range = range(900,20010,100)
    #just remember size of residuals
    dx_rms = []
    dy_rms = []
    for i in smooth_range:
        spline_x, spline_y, dx, dy = calc_sig_dist_spline(x,y,xr,yr,n_iter=5,plot=False, smooth_fac=i, weights=weights)
        
        dx_p = dx
        dy_p = dy
        dx_rms.append(np.std(dx))
        dy_rms.append(np.std(dy))
        print 'smoothing factor of ', str(i)
        print 'Resid: xmin, xmax, ymin, ymax'
        print np.min(dx),np.max(dx),np.min(dy),np.min(dy)
        #if np.max(np.abs(dx))<100 and np.max(np.abs(dy))<100:
        #    outx,outy = spline2lookup(spline_x, spline_y)
        #    fits.writeto('spline_x_s'+str(i), outx)
        #    fits.writeto('spline_y_s'+str(i), outy)

    return smooth_range, dx_rms, dy_rms


def find_best_fit_spline(yeldax, yelday , pos_txt='sig_trimapril_pos.txt', wtype=3):
    '''
    go throigh orfer of legendre polynomial
    look at residuals / errors
    Fit with gaussian with sigma of 1
    measure chi squared
    print those number on a plot, call it a day
    '''

    #num_knots = range(2,10)
    num_knots = np.linspace(300,1200, num=20)
    #order_poly = 3

    tab = Table.read(pos_txt, format='ascii.fixed_width')
    tot_err = np.sqrt(tab['xerr']**2 + tab['yerr']**2 + (tab['xrerr']*5)**2 + (tab['yrerr']*5)**2)
    #tot_err = np.sqrt(tab['xerr']**2 + tab['yerr']**2)
    xerr = np.sqrt(tab['xerr']**2 + (tab['xrerr']*5)**2)
    yerr = np.sqrt( tab['yerr']**2 + (tab['yrerr']*5)**2)
    for i in num_knots:
        taprx, tapry,outx, outy, dx, dy, gbool, sbool = fit_dist_spline(pos_txt=pos_txt, s=i,  n_iter_fit=1, wtype=wtype)

        dxn = dx[sbool] / tot_err[gbool][sbool]
        dyn = dy[sbool] / tot_err[gbool][sbool]
        #import pdb; pdb.set_trace()

        xN , xbin_edge = np.histogram(dxn, bins=100, range=(-5,5))
        yN , ybin_edge = np.histogram(dyn, bins=100, range=(-5,5))
        #import pdb; pdb.set_trace()
        bcenx = np.zeros(len(xbin_edge)-1)
        bceny = np.zeros(len(xbin_edge)-1)
        for dd in range(len(xbin_edge)-1):
            bcenx[dd] = np.mean(xbin_edge[dd] + xbin_edge[dd+1])/2.0
            bceny[dd] = np.mean(ybin_edge[dd] + ybin_edge[dd+1])/2.0

        #import pdb; pdb.set_trace()


        match_trim.plot_dist( yeldax,outx, title2='Spline X Nknots:'+str(i)[:5], title1='Yelda', title3='Difference', title4='Difference', vmind=-.5, vmaxd=.5, outfile='Dist_sol_splineX_'+str(i)[:5]+'.png')
        match_trim.plot_dist( yelday,outy, title2='Spline Y Nknots:'+str(i)[:5], title1='Yelda', title3='Difference', title4='Difference', vmind=-.5, vmaxd=.5, outfile='Dist_sol_splineY_'+str(i)[:5]+'.png')

        plt.figure(5)
        plt.clf()
        match_trim.plot_lookup_diff(yeldax, yelday, outx, outy)
        plt.savefig('HST_fit_spline_smooth'+str(i)+'diff_yelda.png')
        iter_plots(dx, dy, pos_txt,'HST_fit_spline_smooth'+str(i))
            
        #fit_p  = fitting.LevMarLSQFitter()
        
        #gy = models.Gaussian1D(mean=0, stddev=1.0)
        #gy.mean.fixed =True
        #gy.stddev.fixed = True

        #gx = models.Gaussian1D(mean=0, stddev=1.0)
        #gx.mean.fixed =True
        #gx.stddev.fixed = True

        #mx = fit_p(gx , bcenx, xN)
        #my = fit_p(gy , bceny , yN)

        #chix = np.sum((mx(bcenx) - xN)**2/ mx(bcenx))
        #chiy = np.sum((my(bceny) - yN)**2/ my(bceny))

        plt.figure(3)
        plt.clf()
        plt.hist((yeldax-outx).flatten(), bins=100, alpha=.5, label='x')
        plt.hist((yelday-outy).flatten(), bins=100, alpha=.5, label='y')
        plt.text(-.75, 8000, 'std x:'+str(np.std((yeldax-outx).flatten())))
        plt.text(-.75, 10000, 'std y:'+str(np.std((yelday-outy).flatten())))
        plt.legend(loc='upper right')
        plt.title('Difference Yelda vs. April 2015 Spline')
        plt.xlabel('delta (pixels)')
        plt.ylabel('N')
        plt.savefig('hist_diff_from_yelda'+str(i)[:5]+'.png')


        #plt.figure(1)
        #plt.clf()
        #plt.scatter(bcenx, xN)
        #plt.plot(bcenx, mx(bcenx))
        #plt.text(np.min(bcenx)+1, np.max(xN)/2.0, r'$\chi^{2}$: '+str(chix)[:5])
        #plt.text(np.min(bcenx)+1, np.max(xN)/2.0-25, r'$\sigma$:'+str(mx.stddev.value)[:6])
        #plt.text(np.min(bcenx)+1, np.max(xN)/2.0-50,'S: '+str(i))
        #plt.title('X residual Nknots'+str(i)[:5])
        #plt.xlabel('residual / error')
        #plt.ylabel('N')
        #plt.savefig('Spline_x_resid_ord'+str(i)[:5]+'.png')

        #plt.figure(2)
        #plt.clf()
        #plt.hist(dyn, bins=100)
        #plt.scatter(bceny, yN)
        #plt.plot(bceny, my(bceny))
        #plt.text(np.min(bceny)+1, np.max(yN)/2.0, r'$\chi^{2}$: '+str(chiy)[:5])
        #plt.text(np.min(bceny)+1, np.max(yN)/2.0-25, r'$\sigma$:'+str(my.stddev.value)[:6])
        #plt.text(np.min(bceny)+1, np.max(yN)/2.0-50,'S: '+str(i))
        #plt.title('Y residual S:'+str(i)[:5])
        #plt.xlabel('residual / error')
        #plt.ylabel('N')
        #plt.savefig('Spline_y_resid_ord'+str(i)[:5]+'.png')
        #plt.show()
        
        

def find_best_fit(yeldax, yelday , plot_look=False, pos_txt='sig_trimapril_pos.txt', errtype=1, fitter=None):
    '''
    go throigh orfer of legendre polynomial
    look at residuals / errors
    Fit with gaussian with sigma of 1
    measure chi squared
    print those number on a plot, call it a day
    '''

    if fitter == None:
        fitter=high_order.LegTransform
        
    orders= range(3,10)

    tab = Table.read(pos_txt, format='ascii.fixed_width')
    if errtype ==1:
        tot_err = np.sqrt(tab['xerr']**2 + tab['yerr']**2 + (tab['xrerr']*5)**2 + (tab['yrerr']*5)**2)
    elif errtype==2:
        tot_err = np.sqrt(tab['xerr']**2 + tab['yerr']**2)
    for i in orders:
        tapr, dx, dy, gbool, sbool = fit_dist(pos_txt=pos_txt, order=i,n_iter_fit=1, lookup=False)

        dxn = dx[sbool] / tot_err[gbool][sbool]
        dyn = dy[sbool] / tot_err[gbool][sbool]

        xN , xbin_edge = np.histogram(dxn, bins=100, range=(-5,5))
        yN , ybin_edge = np.histogram(dyn, bins=100, range=(-5,5))
        #import pdb; pdb.set_trace()
        bcenx = np.zeros(len(xbin_edge)-1)
        bceny = np.zeros(len(xbin_edge)-1)
        for dd in range(len(xbin_edge)-1):
            bcenx[dd] = np.mean(xbin_edge[dd] + xbin_edge[dd+1])/2.0
            bceny[dd] = np.mean(ybin_edge[dd] + ybin_edge[dd+1])/2.0
        #import pdb; pdb.set_trace()
        fit_p  = fitting.LevMarLSQFitter()
        
        gy = models.Gaussian1D(mean=0, stddev=1.0)
        gy.mean.fixed =True
        #gy.stddev.fixed = True

        gx = models.Gaussian1D(mean=0, stddev=1.0)
        gx.mean.fixed =True
        #gx.stddev.fixed = True

        mx = fit_p(gx , bcenx, xN)
        my = fit_p(gy , bceny , yN)

        #import pdb; pdb.set_trace()
        chix = np.sum((mx(bcenx) - xN)**2/ mx(bcenx))
        chiy = np.sum((my(bceny) - yN)**2/ my(bceny))

        

        
        plt.figure(1)
        plt.clf()
        plt.scatter(bcenx, xN)
        plt.plot(bcenx, mx(bcenx))
        plt.text(np.min(bcenx)+1, np.max(xN)/2.0, r'$\chi^{2}$: '+str(chix)[:5])
        plt.text(np.min(bcenx)+1, np.max(xN)/2.0-20, r'$\sigma$:'+str(mx.stddev.value)[:6])
        #plt.text(np.min(bcenx)+2, np.max(xN)/2.0-30,'smooth factor: '+str(i))
        plt.title('X residual Leg order'+str(i))
        plt.xlabel('residual / error')
        plt.ylabel('N')
        plt.savefig('Leg_x_resid_ord'+str(i)+'.png')

        plt.figure(2)
        plt.clf()
        #plt.hist(dyn, bins=100)
        plt.scatter(bceny, yN)
        plt.plot(bceny, my(bceny))
        plt.text(np.min(bceny)+1, np.max(yN)/2.0, r'$\chi^{2}$: '+str(chiy)[:5])
        plt.text(np.min(bceny)+1, np.max(yN)/2.0-20, r'$\sigma$:'+str(my.stddev.value)[:6])
        #plt.text(np.min(bceny)+2, np.max(yN)/2.0-30,'smooth factor: '+str(i))
        plt.title('Y residual Leg order'+str(i))
        plt.xlabel('residual / error')
        plt.ylabel('N')
        plt.savefig('Leg_y_resid_ord'+str(i)+'.png')
        
        plt.figure(3)
        plt.clf()
        lx, ly = leg2lookup(tapr)
        plot_lookup_diff(lx, ly, yeldax, yelday)
        plt.title('Difference Legendre and Yelda')
        plt.savefig('Leg'+str(i)+'_resid_yelda.png')
        

def comp_yelda(yeldax, yelday , yelda_pos='yelda_pos.txt'):
    '''
    goes through successive orders of legendre polynomial and compare the results for fitting the yelda data to the final Yelada sidtortin map
    '''

    for i in range(3,8):
        t, outx, outy, dx, dy, sbooln = fit_dist(pos_txt=yelda_pos,order=i, n_iter_fit=1, wtype=2)

        plot_dist( yeldax,outx, title2='Leg:'+str(i), title1='Yelda', title3='Difference', title4='Difference', vmind=-.5, vmaxd=.5, outfile='yelda_plots/Dist_sol_X_'+str(i)+'.png')
        plot_dist( yelday,outy, title2='Leg:'+str(i), title1='Yelda', title3='Difference', title4='Difference', vmind=-.5, vmaxd=.5, outfile='yelda_plots/Dist_sol_Y_'+str(i)+'.png')

        plt.figure(1)
        plt.clf()
        plt.hist((yeldax-outx).flatten(), bins=100, alpha=.5, label='x')
        plt.hist((yelday-outy).flatten(), bins=100, alpha=.5, label='y')
        plt.xlabel('difference (pixels)')
        plt.ylabel('N')
        plt.title('Difference L'+str(i)+' and Yelda')
        plt.savefig('yelda_plots/residual_'+str(i)+'.png')

        ref = Table.read(yelda_pos, format='ascii.fixed_width')
        tot_err = np.sqrt(ref['xerr']**2 + ref['yerr']**2)
        dxn = dx / tot_err[sbooln]
        dyn = dy / tot_err[sbooln]

        xN , xbin_edge = np.histogram(dxn, bins=100, range=(-10,10))
        yN , ybin_edge = np.histogram(dyn, bins=100, range=(-10,10))
        #import pdb; pdb.set_trace()
        bcenx = np.zeros(len(xbin_edge)-1)
        bceny = np.zeros(len(xbin_edge)-1)
        for dd in range(len(xbin_edge)-1):
            bcenx[dd] = np.mean(xbin_edge[dd] + xbin_edge[dd+1])/2.0
            bceny[dd] = np.mean(ybin_edge[dd] + ybin_edge[dd+1])/2.0
        #import pdb; pdb.set_trace()
        fit_p  = fitting.LevMarLSQFitter()
        
        gy = models.Gaussian1D(mean=0, stddev=1.0)
        gy.mean.fixed =True
        #gy.stddev.fixed = True

        gx = models.Gaussian1D(mean=0, stddev=1.0)
        gx.mean.fixed =True
        #gx.stddev.fixed = True

        mx = fit_p(gx , bcenx, xN)
        my = fit_p(gy , bceny , yN)

        #import pdb; pdb.set_trace()
        chix = np.sum((mx(bcenx) - xN)**2/ mx(bcenx))
        chiy = np.sum((my(bceny) - yN)**2/ my(bceny))

        

        
        plt.figure(1)
        plt.clf()
        plt.scatter(bcenx, xN)
        plt.plot(bcenx, mx(bcenx))
        plt.text(np.min(bcenx)+1, np.max(xN)/2.0, r'$\chi^{2}$: '+str(chix)[:5])
        plt.text(np.min(bcenx)+1, np.max(xN)/2.0+10, r'$\sigma$:'+str(mx.stddev.value)[:6])
        #plt.text(np.min(bcenx)+2, np.max(xN)/2.0-30,'smooth factor: '+str(i))
        plt.title('X residual Leg order'+str(i))
        plt.xlabel('residual / error')
        plt.ylabel('N')
        plt.savefig('yelda_plots/Leg_x_resid_ord'+str(i)+'.png')

        plt.figure(2)
        plt.clf()
        #plt.hist(dyn, bins=100)
        plt.scatter(bceny, yN)
        plt.plot(bceny, my(bceny))
        plt.text(np.min(bceny)+1, np.max(yN)/2.0, r'$\chi^{2}$: '+str(chiy)[:5])
        plt.text(np.min(bceny)+1, np.max(yN)/2.0-+10, r'$\sigma$:'+str(my.stddev.value)[:6])
        #plt.text(np.min(bceny)+2, np.max(yN)/2.0-30,'smooth factor: '+str(i))
        plt.title('Y residual Leg order'+str(i))
        plt.xlabel('residual / error')
        plt.ylabel('N')
        plt.savefig('yelda_plots/Leg_y_resid_ord'+str(i)+'.png')
       

def iter_sol_leg(order, iter=5, initial_data = 'april_pos.txt', pref_app='', hst_file='../../../M53_F814W/F814_pix_err.dat.rot', plot=False):
    #wrapper to go through successive fits with Legendre polynomials, create new references and make a few plots
    
    #spatially sigma trim the positions
    match_trim.sig_trim_ref(initial_data)
    #now fit the distortion
    ref_base = 'Nref_leg'+str(order)
    data_base = 'pos_leg'+str(order)

    #for bootstrap, need to split the data in half, write new file, run everythin from that base
    tn, dx5n, dy5n, sbooln, b2= fit_dist(pos_txt='sig_trim'+initial_data,order=order, n_iter_fit=1, lookup=False)
    
    
    tab_match = Table.read('first_fits_m.lis', format='ascii.no_header')
    hst = Table.read(hst_file, format='ascii')

    #import pdb;pdb.set_trace()
    run_base = pref_app+'Nref_leg'+str(order)
    f= open(run_base+'hst.trans', 'w')
    pickle.dump(tn, f)
    for i in range(iter):
        #first need new reference (using the above distortion solution)
        match_trim.mk_reference_leg(tab_match['col2'],tab_match['col1'], hst, tn, outfile_pre=run_base+str(i))
        #now need new data table, based on the new reference
        match_trim.match_and_write2(reffile=run_base+str(i)+'.txt',outfile=data_base+str(i)+'.txt')
        #created a new reference, need to sigma trim it
        match_trim.sig_trim_ref(data_base+str(i)+'.txt')
        #now can fit distortion using this as the reference
        tn,lookupx, lookupy,  dxn, dyn, sbooln = fit_dist(pos_txt='sig_trim'+data_base+str(i)+'.txt',order=order, n_iter_fit=1, lookup=True, wtype=1)
        if plot:
            if i !=0:
                plt.figure(5)
                plt.clf()
                match_trim.plot_lookup_diff(lookup_x_prev, lookup_y_prev, lookupx, lookupy)
                plt.savefig(run_base+'iter_'+str(i)+'diff_sequence.png')
        np.save(open(run_base+'iter_'+str(i)+'lookupx.npy', 'w'),lookupx)
        np.save(open(run_base+'iter_'+str(i)+'lookupy.npy', 'w'),lookupy)
        
        lookup_x_prev = lookupx.copy()
        lookup_y_prev = lookupy.copy()
            

        #save tranform objects in case I want them later
        f= open(run_base+str(i)+'.trans', 'w')
        pickle.dump(tn, f)
        if plot:
            iter_plots(dxn, dyn, 'sig_trim'+data_base+str(i)+'.txt', run_base+'iter_'+str(i))
        

        #now we have a new distortion solution, so we return to step 1


def iter_sol_leg_boot(order, iter=5, initial_data = 'april_pos.txt', pref_app='', hst_file='../F814_pix_err.dat.rot', plot=False, boot_trials=100, double=False):
    #wrapper to go through successive fits with Legendre polynomials, create new references and make a few plots
    
    #now fit the distortion
    ref_base = 'Nref_leg'+str(order)
    data_base = 'pos_leg'+str(order)

    #for bootstrap, need to split the data in half, write new file, run everything from that base
   
    for bb in range(boot_trials):

        tab = Table.read(initial_data, format='ascii.fixed_width')
        rand_bool = np.random.choice(len(tab), size=int(len(tab)/2.0))
        #import pdb;pdb.set_trace()
        tmptab = tab[rand_bool]
        if double:
            newtab = join(tmptab, tmptab)
        else:
            newtab = tmptab
        newtab.write(str(bb)+initial_data, format='ascii.fixed_width')
        
        #spaitally sigma trim the positions
        match_trim.sig_trim_ref(str(bb)+initial_data)
    
        tn, dx5n, dy5n, sbooln, b2= fit_dist(pos_txt='sig_trim'+str(bb)+initial_data,order=order, n_iter_fit=1, lookup=False)
    
    
        tab_match = Table.read('first_fits_m.lis', format='ascii.no_header')
        hst = Table.read(hst_file, format='ascii')

        #import pdb;pdb.set_trace()
        run_base = pref_app+'Nref_leg'+str(order)
        f= open(run_base+'hst.trans', 'w')
        pickle.dump(tn, f)
        for i in range(iter):
            #first need new reference (using the above distortion solution)
            match_trim.mk_reference_leg(tab_match['col2'],tab_match['col1'], hst, tn, outfile_pre=run_base+str(i))
            #now need new data table, based on the new reference
            match_trim.match_and_write2(reffile=run_base+str(i)+'.txt',outfile=data_base+str(i)+'.txt')
            #now trim down to the stars that were randomly selected previosly
            tmptab = Table.read(data_base+str(i)+'.txt', format='ascii.fixed_width')
            newtab = tmptab[rand_bool]
            newtab.write(data_base+str(i)+'.txt', format='ascii.fixed_width')
            #created a new reference, need to sigma trim it
            match_trim.sig_trim_ref(data_base+str(i)+'.txt')
            #now can fit distortion using this as the reference
            tn,  dxn, dyn, sbooln, b2 = fit_dist(pos_txt='sig_trim'+data_base+str(i)+'.txt',order=order, n_iter_fit=1, lookup=False, wtype=1)
            if plot:
                if i !=0:
                    plt.figure(5)
                    plt.clf()
                    match_trim.plot_lookup_diff(lookup_x_prev, lookup_y_prev, lookupx, lookupy)
                    plt.savefig(run_base+'iter_'+str(i)+'diff_sequence.png')
            #np.save(open(run_base+'iter_'+str(bb)+'lookupx.npy', 'w'),lookupx)
            #np.save(open(run_base+'iter_'+str(bb)+'lookupy.npy', 'w'),lookupy)
            
            

        #save tranform objects in case I want them later
            f= open(run_base+'outsol'+str(bb)+'.trans', 'w')
            pickle.dump(tn, f)
            if plot:
                iter_plots(dxn, dyn, 'sig_trim'+data_base+str(bb)+'.txt', run_base+'iter_'+str(i))
                
        f= open(run_base+str(i)+'_'+str(bb)+'.trans', 'w')

def iter_plots( dx, dy, pos_txt, pref):
    #first want quiver plot of the residuals wrt data
    tab = Table.read(pos_txt, format='ascii.fixed_width')
    plt.figure(1)
    plt.clf()
    match_trim.mk_quiver_resid(tab['x'], tab['y'], dx, dy, title_s=pref)
    plt.savefig(pref+'quiver_resid.png')

    plt.figure(2)
    plt.clf()
    plt.hist(dx, bins=100, label='x', alpha=.5)
    plt.hist(dy, bins=100, label='y', alpha=.5)
    plt.text(-.75, 125, 'mean, sig x:'+str(np.mean(dx))[:6] + ' '+str(np.std(dx))[:6])
    plt.text(-.75,100, 'mean, sig y:'+str(np.mean(dy))[:6]+ ' '+str(np.std(dy))[:6])
    plt.xlabel('Residual (pixels)')
    plt.ylabel('N')
    plt.legend(loc='upper right')
    plt.savefig(pref+'hist_resid.png')



def calc_err(lis_trans='trans.lis'):
    '''
    calulates errors in transformation from bootstrap
    '''

    ttab = Table.read(lis_trans, format='ascii.no_header')
    lx = []
    ly = []
    for trans_p in ttab['col1']:
        t = pickle.load(open(trans_p, 'r'))
        outx, outy = match_trim.leg2lookup(t)

        lx.append(outx)
        ly.append(outy)

    lxn = np.array(lx)
    lyn = np.array(ly)

    distx = np.mean(lxn, axis=0)
    disty = np.mean(lyn, axis=0)

    err_x = np.std(lxn, axis=0)
    err_y = np.std(lyn, axis=0)

    np.save('dx.npy', distx)
    np.save('dy.npy', disty)
    np.save('lxn.npy', lxn)
    np.save('lyn.npy', lyn)
    np.save('dxerr.npy', err_x)
    np.save('dyerr.npy', err_y)

    return distx, disty, err_x, err_y

def plot_err(xerr_f='dxerr.npy', yerr_f='dyerr.npy'):
    '''
    plots the errors from the bootsrtap analysis
    '''
    xerr = np.load(xerr_f)
    yerr = np.load(yerr_f)

    llim = 0
    ulim = .3
    plt.figure(1)
    plt.clf()
    plt.title('Distortion Solution Uncertainty (X)')
    plt.xlabel('X (pix)')
    plt.ylabel('Y (pix)')
    plt.gray()
    plt.imshow(xerr, vmin=llim, vmax=ulim)
    cbar = plt.colorbar()
    cbar.set_label('X error (pix)', rotation=270, labelpad =+25)
    plt.axes().set_aspect('equal')

    plt.figure(2)
    plt.clf()
    plt.title('Distortion Solution Uncertainty (Y)')
    plt.xlabel('X (pix)')
    plt.ylabel('Y (pix)')
    plt.gray()
    plt.imshow(yerr, vmin=llim, vmax=ulim)
    cbar = plt.colorbar()
    cbar.set_label('Y error (pix)', rotation=270, labelpad =+25)
    plt.axes().set_aspect('equal')

    #make histogram of uncertainties
    plt.figure(3)
    plt.clf()
    plt.title('Distortion Solution Uncertainty')
    plt.xlabel('Error (pix)')
    plt.ylabel('N')
    plt.hist(xerr.flatten(), histtype='step', bins=30, range=(0,.25), label='x', lw=3, normed=True,  color='red')
    plt.hist(yerr.flatten(),histtype='step', bins=30, range=(0,.25), label='y', lw=3, normed=True, linestyle='dashed',color='blue')
    plt.legend(loc='upper right')
    #plt.axes().set_aspect('equal')

def calc_chisq(trans_f, pos_f, add_err=0):
    '''
    calculates the chi squared and reduced chi squared for the given data and transformation
    trans_f, str: filename of the pickled tranfomation object
    pos_f, str: filename of the sigma clipped data file that was used to generate the tranfoamtion
    add_Err is the additive error term added to the error when computing chi-square (default is 1 mas)
    '''
    t = pickle.load(open(trans_f, 'r'))
    tab = Table.read(pos_f, format='ascii.fixed_width')

    #compute model measuremtns from fit
    modx, mody = t.evaluate(tab['x'], tab['y'])
    datx = tab['xr'] - tab['x']
    daty = tab['yr'] - tab['y']

    errx = (tab['xerr']**2+tab['xrerr']**2+add_err**2)
    erry = (tab['yerr']**2+tab['yrerr']**2+add_err**2)
    chix = np.sum((modx-datx)**2 / errx)
    chiy = np.sum((mody-daty)**2 / erry)
    print 'reduced Chi square X: ', chix / len(t.px.parameters)
    print 'reduced Chi square Y: ', chiy / len(t.px.parameters)
    print 'Number of free parameters in X fit', len(t.px.parameters)

    return chix, chiy,  len(t.px.parameters), len(datx)

def calc_prob(trans_lis, pos_lis, add_err=0):
    '''
    '''

    chix = []
    chiy = []
    npar = []
    ndata = []
    
    for i in range(len(trans_lis)):
        cx, cy, npart, ndatat = calc_chisq(trans_lis[i], pos_lis[i], add_err=add_err)
        chix.append(cx)
        chiy.append(cy)
        npar.append(npart)
        ndata.append(ndatat)

    chix = np.array(chix)
    chiy = np.array(cjiy)
    ndata = np.array(ndata)
        
    probx = []
    proby = []
    for i in range(len(chix)-1):
        #now , need to integrate the f-distribution from ratio of chi-squares to infinity
        ratx = ((chix[i]) - (chix[i+1])) / (npar[i+1]-npar[i]) /  (chix[i+1]/(ndata[i]-npar[i+1]))
        raty = ((chiy[i]) - (chiy[i+1])) / (npar[i+1]-npar[i]) /  (chiy[i+1]/(ndata[i]-npar[i+1]))
    
        #raty = ((chiy[i]/nfree[i]) - (chiy[i+1]/nfree[i+1])) /  (chiy[i+1]/nfree[i+1])
        #import pdb;pdb.set_trace()
        tmpx = 1.0-quad(fdist.pdf , ratx, np.inf, args=(ndata[i]-npar[i], ndata[i+1]-npar[i+1]))[0]
        tmpy = 1.0-quad(fdist.pdf , raty, np.inf, args=(ndata[i]-npar[i], ndata[i+1]-npar[i+1]))[0]
        probx.append(tmpx)
        proby.append(tmpy)

    return probx, proby,  chix, chiy
        

    
        
        
    
    
    

    
