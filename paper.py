from . import match_trim, fit_dist
import pickle
import numpy as np
from astropy.table import Table
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def mkfig2():
    plt.figure(1, figsize=(6,6))
    match_trim.plot_pos_err(pos_txt ='/Users/service/Distortion_solution/starfinder/april/sig_trimpos_leg65.txt', title_s='2015-04-02')
    plt.savefig('pos_err_apr.png')
    plt.figure(2, figsize=(6,6))
    match_trim.plot_pos_err(pos_txt ='/Users/service/Distortion_solution/starfinder/may/sig_trimpos_leg65.txt', title_s='2015-05-05')
    plt.savefig('pos_err_may.png')

def mkfig3():
    
    plt.figure(1, figsize=(6,6))
    plt.clf()
    match_trim.mk_quiver_from_txt(pos_txt ='/Users/service/Distortion_solution/starfinder/april/april_pos.txt', color='red')
    match_trim.mk_quiver_from_txt(pos_txt ='/Users/service/Distortion_solution/starfinder/april/sig_trimapril_pos.txt', color='black', title_s='2015-04-02')
    plt.savefig('april_deltas_quiver.png')
    plt.clf()
    plt.figure(1, figsize=(6,6))
    plt.clf()
    match_trim.mk_quiver_from_txt(pos_txt ='/Users/service/Distortion_solution/starfinder/may/may_pos.txt', color='red')
    match_trim.mk_quiver_from_txt(pos_txt ='/Users/service/Distortion_solution/starfinder/may/sig_trimmay_pos.txt', color='black', title_s='2015-05-05')
    plt.savefig('may_deltas_quiver.png')


def mkfig4():
    apr_dir = '/Users/service/Distortion_solution/starfinder/april/'
    may_dir = '/Users/service/Distortion_solution/starfinder/may/'
    pos_hsta = [ 'sig_trimapril_pos.txt','sig_trimapril_pos.txt','sig_trimapril_pos.txt','sig_trimapril_pos.txt','sig_trimapril_pos.txt', 'sig_trimapril_pos.txt', 'sig_trimapril_pos.txt']
    pos_hstm = [ 'sig_trimmay_pos.txt','sig_trimmay_pos.txt','sig_trimmay_pos.txt','sig_trimmay_pos.txt','sig_trimmay_pos.txt', 'sig_trimmay_pos.txt', 'sig_trimmay_pos.txt']
    tran_hst = ['Nref_leg2hst.trans', 'Nref_leg3hst.trans', 'Nref_leg4hst.trans', 'Nref_leg5hst.trans', 'Nref_leg6hst.trans', 'Nref_leg7hst.trans', 'Nref_leg8hst.trans']
    pos_apr = []
    pos_may = []
    tran_apr = []
    tran_may = []
    for i in range(len(pos_hsta)):
        pos_apr.append(apr_dir + pos_hsta[i])
        pos_may.append(may_dir+pos_hstm[i])
        tran_apr.append(apr_dir+tran_hst[i])
        tran_may.append(may_dir+tran_hst[i])

    px, py, fxA, fyA, chix, chiy = fit_dist.calc_prob(tran_apr, pos_apr)
    px, py, fxM, fyM, chix, chiy = fit_dist.calc_prob(tran_may, pos_may)
    M = np.array([3,4,5,6,7,8])
    plt.figure(1, figsize=(6,6))
    plt.clf()
    plt.plot(M, fxA,'v', label='April X',  ms=8)
    plt.plot(M, fyA,'^', label='April Y',  ms=8)
    plt.plot(M, fxM, '<', label='May X', ms=8)
    plt.plot(M, fyM, '>', label='May Y', ms=8)
    plt.xlabel('M', fontsize=14)
    plt.ylabel(r'F value M - 1 $\rightarrow$ M', fontsize=14)
    plt.xlim((2,9))
    plt.legend(loc ='upper right', numpoints=1)
    plt.savefig('f_value_all.png')
    
        

def mkfig5():
    tran1 = pickle.load(open('/Users/service/Distortion_solution/starfinder/may/Nref_leg6hst.trans'))
    tran6 = pickle.load(open('/Users/service/Distortion_solution/starfinder/may/Nref_leg65.trans'))

    out6x , out6y = match_trim.leg2lookup(tran6)
    out1x, out1y  = match_trim.leg2lookup(tran1)
    plt.figure(1, figsize=(6,6))
    plt.clf()
    match_trim.plot_lookup_diff(out1x, out1y, out6x, out6y, title='Change from Iteration', scale=10, scale_size=.5)
    plt.savefig('may_iter_diff_quiver.png')

def mkfig6():
    dxerrM = np.load('/Users/service/Distortion_solution/Bootstrap_may/dxerr.npy')
    dyerrM = np.load('/Users/service/Distortion_solution/Bootstrap_may/dyerr.npy')

    dxerrA = np.load('/Users/service/Distortion_solution/Bootstrap_april/dxerr.npy')
    dyerrA = np.load('/Users/service/Distortion_solution/Bootstrap_april/dyerr.npy')

    llim = 0.01
    ulim = .3
    #t = np.linspace(llim, ulim, num=4)
    plt.figure(1, figsize=(6,6))
    plt.clf()
    plt.title('2015-05-05 Uncertainty (X)', fontsize=14)
    plt.xlabel('X (pix)', fontsize=14)
    plt.ylabel('Y (pix)', fontsize=14)
    plt.gray()
    plt.imshow(dxerrM, norm=LogNorm(vmin=llim, vmax=ulim))
    cbar = plt.colorbar()
    cbar.set_label('X error (pix)', rotation=270, labelpad =+25, fontsize=14)
    plt.axes().set_aspect('equal')

    plt.savefig('may_fit_err_X.png')

    


    plt.figure(1, figsize=(6,6))
    plt.clf()
    plt.title('2015-05-05 Uncertainty (Y)', fontsize=14)
    plt.xlabel('X (pix)', fontsize=14)
    plt.ylabel('Y (pix)', fontsize=14)
    plt.gray()
    plt.imshow(dyerrM, norm=LogNorm(vmin=llim, vmax=ulim))
    cbar = plt.colorbar()
    cbar.set_label('Y error (pix)', rotation=270, labelpad =+25, fontsize=14)
    plt.axes().set_aspect('equal')
    plt.savefig('may_fit_err_Y.png')

  
    plt.figure(1, figsize=(6,6))
    plt.clf()
    plt.title('2015-04-02 Uncertainty (X)', fontsize=14)
    plt.xlabel('X (pix)', fontsize=14)
    plt.ylabel('Y (pix)', fontsize=14)
    plt.gray()
    plt.imshow(dxerrA,  norm=LogNorm(vmin=llim, vmax=ulim))
    cbar = plt.colorbar()
    cbar.set_label('X error (pix)', rotation=270, labelpad =+25, fontsize=14)
    plt.axes().set_aspect('equal')
    plt.savefig('april_fit_err_X.png')

   
    plt.figure(1, figsize=(6,6))
    plt.clf()
    plt.title('2015-04-02 Uncertainty (Y)', fontsize=14)
    plt.xlabel('X (pix)', fontsize=14)
    plt.ylabel('Y (pix)', fontsize=14)
    plt.gray()
    plt.imshow(dyerrA,  norm=LogNorm(vmin=llim, vmax=ulim))
    cbar = plt.colorbar()
    cbar.set_label('Y error (pix)', rotation=270, labelpad =+25, fontsize=14)
    plt.axes().set_aspect('equal')
    plt.savefig('april_fit_err_Y.png')

    plt.figure(1, figsize=(6,6))
    plt.clf()
    plt.title('2015-04-02 Uncertainty', fontsize=14)
    plt.hist(dxerrA.flatten(), histtype='step', bins=30, range=(0,.25), label='x', lw=3, normed=True, color='red')
    plt.hist(dyerrA.flatten(), histtype='step', bins=30, range=(0,.25), label='y', lw=3, normed=True, color='blue', linestyle='dashed')
    plt.legend(loc='upper right')
    plt.axvline(0.1, color='black', lw=3)
    plt.xlabel('Uncertainty (pix)', fontsize=14)
    plt.savefig('april_fit_err_hist.png')

    plt.figure(1, figsize=(6,6))
    plt.clf()
    plt.title('2015-05-05 Uncertainty', fontsize=14)
    plt.hist(dxerrM.flatten(), histtype='step', bins=30, range=(0,.25), label='x', lw=3, normed=True, color='red')
    plt.hist(dyerrM.flatten(), histtype='step', bins=30, range=(0,.25), label='y', lw=3, normed=True, color='blue', linestyle='dashed')
    plt.legend(loc='upper right')
    plt.xlabel('Uncertainty (pix)', fontsize=14)
    
        
    plt.axvline(0.1, color='black', lw=3)
    plt.savefig('may_fit_err_hist.png')



def mkfig7():
    tran6M = pickle.load(open('/Users/service/Distortion_solution/starfinder/may/Nref_leg65.trans'))
    out6xM , out6yM = match_trim.leg2lookup(tran6M)
    tran6A = pickle.load(open('/Users/service/Distortion_solution/starfinder/april/Nref_leg65.trans'))
    out6xA , out6yA = match_trim.leg2lookup(tran6A)

    fitsx = fits.open('/Users/service/Distortion_solution/Yelda/nirc2_X_distortion.fits.gz')
    yeldax = fitsx[0].data
    fitsy = fits.open('/Users/service/Distortion_solution/Yelda/nirc2_Y_distortion.fits.gz')
    yelday = fitsy[0].data
    yelda_yerr = fits.open('/Users/service/Distortion_solution/Yelda/nirc2_Yerr_withResidual.fits.gz')
    yelda_yerr = yelda_yerr[0].data
    
    yelda_xerr = fits.open('/Users/service/Distortion_solution/Yelda/nirc2_Xerr_withResidual.fits.gz')
    yelda_xerr = yelda_xerr[0].data

    dxerrA = np.load('/Users/service/Distortion_solution/Bootstrap_april/dxerr.npy')
    dyerrA = np.load('/Users/service/Distortion_solution/Bootstrap_april/dyerr.npy')
    
    plt.figure(1, figsize=(6,6))
    match_trim.plot_lookup(out6xA, out6yA, title='2015-04-02')
    plt.savefig('apr_lookup_final.png')

    plt.figure(1, figsize=(6,6))
    match_trim.plot_lookup(out6xM, out6yM, title='2015-05-05')
    plt.savefig('may_lookup_final.png')

    plt.figure(1, figsize=(6,6))
    match_trim.plot_lookup(yeldax, yelday, title='Yelda et al. (2010)')
    plt.savefig('yelda_lookup_final.png')

    
    plt.figure(1, figsize=(6,6))
    match_trim.plot_lookup_diff(yeldax, yelday,out6xM, out6yM, title='Difference', scale=10, scale_size=.5, spacing=48)
    plt.savefig('apr-may_diff.png')

    plt.figure(1, figsize=(6,6))
    match_trim.plot_lookup_diff(yeldax, yelday,out6xA, out6yA, title='Difference', scale=10, scale_size=.5, spacing=48)
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.savefig('diff_yelda_april_quiver.png')
    deltx = yeldax - out6xA
    delty = yelday - out6yA

    dx_sig = deltx / (dxerrA**2 + 0.1**2 + yelda_xerr**2)**0.5
    dy_sig = delty / (dyerrA**2 + 0.1**2 + yelda_yerr**2)**0.5

    plt.figure(1, figsize=(6,6))
    plt.clf()
    plt.hist(dx_sig.flatten(), bins=20, range=(-2,2), lw=3,histtype='step', color='red', label='x', normed=True)
    plt.hist(dy_sig.flatten(), bins=20, range=(-2,2), lw=3,histtype='step', linestyle='dashed', color='blue', label='y', normed=True)
    plt.legend(loc='upper right')
    plt.xlabel(r'$\Delta / \sigma$', fontsize=14)
    plt.ylabel('N', fontsize=14)
    plt.savefig('diff_yelda_april_sighist.png')
    
    
