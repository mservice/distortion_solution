from astropy.table import Table
import numpy as np
from astropy.io import fits 


def write_tab(strehl_file='strehl_source.txt', frame_file = 'first_fits_m.lis', date='april', off_f='offsets.dat'):
    '''
    reads in some files, looks for the data
    writes out the table I want for the data quality table for the paper
    '''

    off_tab = Table.read(off_f, format='ascii.fixed_width')
    fits_tab = Table.read(frame_file, format='ascii.no_header')
    strehl_tab = Table.read(strehl_file, format='ascii')

    #first create list of the fits files associated with the lis_files
    lis_list = []
    nstars = []
    nimg = []
    sig_stars = []
    pa = []
    texp = []
    coadds = []
    offx = []
    offy = []
    
    for i in range(len(fits_tab['col1'])):
        
        #lis_list.append(fits_tab['col1'][i][17:23]+'_0.8_stf.lis')
        mat_tab = Table.read(fits_tab['col2'][i], format='fits')
        nstars.append(len(mat_tab['col0']))
        nimg.append(np.max(mat_tab['col6']))
        err = (np.mean(mat_tab['col2'])+ np.mean(mat_tab['col4'])) / 2.0
        sig_stars.append(err)
        tmpfits = fits.open(fits_tab['col1'][i])
        pa.append(tmpfits[0].header['ROTPOSN']-0.7)
        #import pdb;pdb.set_trace()
        texp.append(tmpfits[0].header['ITIME'])
        coadds.append(tmpfits[0].header['COADDS'])
        offx.append(off_tab['offx'][i])
        offy.append(off_tab['offy'][i])
            
        
    fwhm = []
    strehl = []
    mjd = []
    for i in range(len(fits_tab['col1'])):
        #first find correct index for strehl file
        #import pdb;pdb.set_trace()
        b_in = np.argmax(strehl_tab['col1'] == fits_tab['col1'][i][17:28].replace('d',''))
        strehl.append(strehl_tab['col2'][b_in])
        fwhm.append(strehl_tab['col4'][b_in])
        mjd.append(strehl_tab['col5'][b_in])

    #out_tab = Table(data=[mjd, pa, texp, coadds, fwhm, strehl, nstars, sig_stars], names=['MJD (UT)', 'P.A.', r'$t_{exp}$', 'Co-adds','Strehl', 'FWHM' r'$N_{stars}$', r'$\sigma_{pos}'])
    out_tab = Table(data=[mjd, pa,offx, offy,nimg, texp, coadds, fwhm, strehl, nstars, sig_stars])
    out_tab.write('data.tex', format='ascii.latex')
        
        

        

        
        
