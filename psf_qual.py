import numpy as np
import radial_data
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from scipy.ndimage.measurements import  center_of_mass
from astropy.table import Table
from astropy.io import fits 
from Imaka import util
from jlu.util import psf
 
def measure_var(fits_f, path_to_fits='./', path_to_stf='/Users/service/Distortion_solution/starfinder/may/'):
    '''
     measures the FWHM of brightest num_stars stars in the image
    '''

    fits_lis = Table.read(path_to_fits+fits_f, format='ascii.no_header')['col1']
    stf_lis = [] 
    for i in range(len(fits_lis)):
        stf_lis.append(fits_lis[i].replace('.fits', '_0.8_stf.lis'))

    y = []
    x = []
    fwhm = []
    fwhmx = []
    fwhmy = []
    ecc = []
    e_ang = []
    
    for i in range(len(fits_lis)):
        _fwhm, _fwhmx, _fwhmy, _ecc, _e_ang, _MuX, _Muy, _x, _y = get_fwhm_psf(path_to_stf+stf_lis[i], fits_lis[i])
        fwhm.append(_fwhm)
        x.append(_x)
        y.append(_y)
        ecc.append(_ecc)
        e_ang.append(_e_ang)
        fwhmy.append(_fwhmy)
        fwhmx.append(_fwhmx)

    return np.array(fwhm), np.array(x), np.array(y), np.array(fwhmx), np.array(fwhmy), np.array(ecc), np.array(e_ang)

    

def get_fwhm_psf(psffile='cent_loc.txt', im_file='ave.fits', psf_size=12, plot_rad=False, nstars=10, sep_r=6, sep_mag=1.5, size=12):
    '''
    '''
    im = fits.open(im_file)[0].data
    loc_tab = Table.read(psffile, format='ascii.no_header')
    x = loc_tab['col4']
    y = loc_tab['col5']
    mag = loc_tab['col2']
    out_filt = np.ones(len(x),dtype='bool')
    out_filt = out_filt * (x > sep_r) * (y > sep_r) 
    for i in range(len(x)):
        _filt = np.ones(len(x),dtype='bool')
        _filt[i] = False
        mfilt = np.abs((mag[i]-mag)) > sep_mag
        _filt = _filt * mfilt 
        _sep  = ((x[i] - x[_filt])**2 + (y[i] - y[_filt])**2)**0.5
        if np.min(_sep) < sep_r:
            out_filt[i] = False
    x = x[out_filt][:10]
    y = y[out_filt][:10]
    
    fwhmx = []
    fwhmy = []
    height = []
    MuX = []
    MuY = []
    fwhm = []
    ecc = []
    e_ang = []
    
    for i in range(len(x)):
        #import pdb;pdb.set_trace()
        #rad = util.bin_rad(im, y[i], x[i], psf_size, plot=True)
        _tmp_im = im[y[i] -size/2+1:y[i]+size/2+1, x[i]- size/2+1:x[i]+size/2+1]
        #import pdb;pdb.set_trace()
        _Height, _MuX, _MuY, _FWHMX, _FWHMY, _FWHM, _E, _EA = psf.moments(_tmp_im)
        fwhmx.append(_FWHMX)
        fwhmy.append(_FWHMY)
        fwhm.append(_FWHM)
        ecc.append(_E)
        e_ang.append(_EA)
        MuX.append(_MuX)
        MuY.append(_MuY)
        height.append(_Height)
        
        
    plt.show()
    return fwhm, fwhmx, fwhmy, ecc, e_ang, MuX, MuY, x, y

def fwhm_from_rad(rad):
    '''
    '''
    #import pdb;pdb.set_trace()
    hmax = np.max(rad.mean) * 0.5
    print hmax
    #fwhm = np.max(rad.r[rad.mean > hmax]) * 2
    fwhm = np.interp(hmax*-1, rad.mean*-1, rad.r) * 2
    return fwhm

def get_2dgauss_psf(psffile='cent_loc.txt', im_file='ave.fits', psf_size=30):
    '''
    '''
    im = fits.open(im_file)[0].data
    loc_tab = Table.read(psffile, format='ascii.no_header')[:10]
    x = loc_tab['col4']
    y = loc_tab['col5']
    xstd = []
    ystd = []
    
    for i in range(len(x)):
        #import pdb;pdb.set_trace()
        _xstd, _ystd = fit2dgauss(im[y[i]-psf_size/2.0:y[i]+psf_size/2.0, x[i]-psf_size/2.0:x[i]+psf_size/2.0])
        #gauss_models.append(m_im)
        xstd.append(_xstd)
        ystd.append(_ystd)
    return xstd, ystd



def fit2dgauss(im, std_guess=5, plot=False ):
    '''
    fits a  2d gaussian to the image provided
    '''


    fit_g = fitting.LevMarLSQFitter()
    g_init = models.Gaussian2D(x_mean=im.shape[0]/2.0, y_mean=im.shape[1]/2.0, x_stddev=std_guess, y_stddev=std_guess)
    coo = np.meshgrid(range(im.shape[0]), range(im.shape[1]))
    g = fit_g(g_init, coo[0], coo[1], im)

    model_im = g(coo[0], coo[1])

    if plot:
        plt.figure(1)
        plt.clf()
        plt.imshow(im - model_im)
        plt.colorbar()
        plt.gray()
        plt.title('Residual')
        plt.axes().set_aspect('equal')
        import pdb;pdb.set_trace()
        plt.show()
    return np.abs(g.x_stddev[0]), np.abs(g.y_stddev[0])
    
    
def calc_ecc(x,y):

    ecc = []
    for i in range(len(x)):
        if x[i] > y[i]:
            _fac = y[i]**2 / x[i]**2
        else:
            _fac = x[i]**2 / y[i]**2
        ecc.append(np.sqrt(1 - _fac))

    return np.array(ecc)
