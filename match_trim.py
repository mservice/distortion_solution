from astropy.table import Table
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

def match2hst(lis_f, hst_tab_ref, hst_tab_red):
    '''
    lis_f is the name of the text file catalog of nirc2 positi
    matches the lis file from Starfinder with the HST reference
    uses the fact that the first star is in the HST Reference (by name) to trim down the Stars from HST for matching
    Note: the HST coordiantes for this are not DAR corrected
     I use a matching raius of 2 Nirc2 pixels (20 mas)
    
    '''

    Looked_twice=False
    stf = Table.read(lis_f, format='ascii')
    refname = stf['col1'][0]

    hst = hst_tab_red

    ref_in = np.argmax(hst['Name']==refname)
    xcen = hst_tab_ref['Xarc'][ref_in]
    ycen = hst_tab_ref['Yarc'][ref_in]
    hpix_scale = .05  #arcseconds/pixel
    
    #want a 20 arcsecond box, to make sure we get all stars that could match
    pix_r = 10 / hpix_scale
    hst_bool = (hst['Xarc'] < xcen + pix_r)*(hst['Xarc'] > xcen  - pix_r)*(hst['Yarc'] < ycen + pix_r)*(hst['Yarc'] > ycen - pix_r)
    #hst_bool = np.ones(hst['Xarc'].shape, dtype=bool)

    N, x1m, y1m, m1m, x2m, y2m, m2m = jay.miracle_match_briteN(stf['col4'], stf['col5'], stf['col2'], hst['Xarc'][hst_bool], hst['Yarc'][hst_bool], hst['Mag'][hst_bool], 60)
    if N < 5:
        looked_twice=True
        N, x1m, y1m, m1m, x2m, y2m, m2m = search_for_mat(hst,stf)
        
        if N == None:
            N, x1m, y1m, m1m, x2m, y2m, m2m = search_for_mat(hst,stf,fac=2)
    
    dm = np.average(m1m-m2m)
    
    #now use matched coordiantes to tranform the sftarfinder lists into hst frame.
    #then match again, only on coordiantes?  keep it tight to avoid fake matches, also mind the potential for color terms between K and HST

   
    stfx = stf['col4']
    stfy = stf['col5']
        
    cx,cy = high_order.four_param(x2m, y2m, x1m,y1m)
    xnew = cx[0] + cx[1]*hst['Xarc'] + cx[2]*hst['Yarc']
    ynew = cy[0] + cy[1]*hst['Xarc'] + cy[2]*hst['Yarc']
    #now tranform hst into Nirc2 frame
    mbool = np.zeros(stf['col4'].shape,dtype='bool')
    hstin = []
    #import pdb; pdb.set_trace()

    idx1 , idx2 , dm, dr = jay.match(stf['col4'], stf['col5'] ,stf['col2'], xnew, ynew, hst['Mag'], 4)

    #if we matched less than 20% of the stars, try matching again because something almost certainly went wrong

    print 'found ',len(idx1),' matches out of  ',len(stf['col4'])
    #assert len(idx1) > 5
    #now we have more matches
    #what I need for the distortion solution is the delta to HST
    #and I also need the original pixel location on the chip
    #return those things, let everything go
    #consider savinf star name from hst list, to give chance to eliminate stars with bad measurements (Saturated, etc.)

    return   stf['col4'][idx1], stf['col5'][idx1], xnew[idx2], ynew[idx2], hst['Name'][idx2], np.ones(len(hst), dtype='bool'), idx2, stf['col2'][idx1], hst['Mag'][idx2], hst['xerr'][idx2], hst['yerr'][idx2]


def match2hst_err(lis_f, ref, fits_file, ecut=.002, ref_scale=1.0,ref_angle=0 ,ap_dar=True, rev_x=False):
    '''
    lis_f is the name of the fits table containing the Nirc2 coordiantes form a single pointing
    ref is the "distortion free" reference (either Nirc2 or HST)
    ecut (arcseconds) is the error cut that must be met for stars to be included in the 4 parameter tranfomtion between reference and lis_f
    rev_x is bool, if True multioply xref by -1.0.  this is done to maje +x to the west for the DAR correction
    
    ref_scale is scale if refernce catalog in arcsconds/pixel
    '''

    if ref_angle != 0:
        xinit , yinit = rot_lis(ref['Xarc'], ref['Yarc'], ref_angle)
        
    else:
        xinit = ref['Xarc']
        yinit = ref['Yarc']
        
    stf = Table.read(lis_f, format='fits')
    
    stf_pix_scale =.01
    
    idhst = match_by_name(stf['col0'], ref['Name'])

    if ap_dar:
        #this applies DAR to space coordinates, to make thme comparable to the Nirc2 distorted frames
        if rev_x:
            #import pdb;pdb.set_trace()
            xhst, yhst = applyDAR_coo(fits_file,xinit*ref_scale*-1.0,yinit*ref_scale)
            xhst = xhst * -1.0
        else:
            xhst, yhst = applyDAR_coo(fits_file,xinit*ref_scale,yinit*ref_scale)
    else:
        xhst = xinit * ref_scale
        yhst = yinit * ref_scale
        
    #do error cut, then recalulcate 4 parameter tranforamtion of HST
    hst_ebool = (ref['xerr'][idhst] < ecut / ref_scale )*(ref['yerr'][idhst] < ecut / ref_scale)
    stf_ebool = (stf['col2'] < ecut / stf_pix_scale)*(stf['col4'] < ecut / stf_pix_scale)
    tbool = hst_ebool * stf_ebool
    

    cx,cy = high_order.four_param(xhst[idhst][tbool], yhst[idhst][tbool], stf['col1'][tbool], stf['col3'][tbool])
    xnew = cx[0] + cx[1]*xhst + cx[2]*yhst
    ynew = cy[0] + cy[1]*xhst + cy[2]*yhst

    idx1 , idx2 , dm, dr = jay.match(stf['col1'], stf['col3'] ,stf['col5'], xnew, ynew, ref['Mag'], 3)                            
        
            
    print 'found ',len(idx1),' matches out of  ',len(stf['col4'])
    
    return   stf['col1'][idx1], stf['col3'][idx1], xnew[idx2], ynew[idx2], ref['Name'][idx2], stf['col5'][idx1], ref['Mag'][idx2], (ref['xerr'][idx2]*ref_scale)/stf_pix_scale, (ref['yerr'][idx2]*ref_scale)/stf_pix_scale, stf['col2'][idx1], stf['col4'][idx1]


def match2hst_err_ret_hst(lis_f,fits_file ,hst,tx, ty, ecut=.001, spline=False):
    '''
    matches the lis_f is the stacked measurements from frames at a single pointing
    matches stars by name to the HST reference
    corrects for distortion based on tx,ty
    then corrects for DAR based on the information in the header of fits_file
    then tranforams the lis_f catalogs to the HST frame using a four parameter tranformation on stars with positional errors < ecut (arcseconds)
    returns coordinates in  arcseconds in the HST frame 
    called by mk_reference
    DOES NOT FIND NEW MATCHES
    '''

    stf = Table.read(lis_f, format='fits')
    hpix_scale = .05
    stf_pix_scale =.01
    
    idhst = match_by_name(stf['col0'], hst['Name'])

    xorig = stf['col1']
    yorig = stf['col3']
    #first apply distortion solution, then apply DAR to get to space ref frame
    if not spline:
        dx, dy =tx.evaluate(xorig, yorig)
        xin = dx + xorig
        yin = dy + yorig
    else:
        xin = tx.ev(xorig, yorig) + xorig
        yin = ty.ev(xorig, yorig) + yorig

    (pa, darCoeffL, darCoeffQ) = nirc2dar(fits_file)
  

    
    sina = math.sin(pa)
    cosa = math.cos(pa)

    xnew2 =  xin * cosa + yin * sina
    ynew2 = -xin * sina + yin * cosa

    # Apply DAR correction along the y axis
    xnew3 = xnew2
    ynew3 = ynew2*(1 + darCoeffL) + ynew2*np.abs(ynew2)*darCoeffQ

    # Rotate coordinates counter-clockwise by PA back to original
    xstf = xnew3 * cosa - ynew3 * sina
    ystf = xnew3 * sina + ynew3 * cosa
     
    
    hst_ebool = (hst['xerr'][idhst] < ecut / hpix_scale )*(hst['yerr'][idhst] < ecut / hpix_scale)
    stf_ebool = (stf['col2'] < ecut / stf_pix_scale)*(stf['col4'] < ecut / stf_pix_scale)
    tbool = hst_ebool * stf_ebool


    #cx,cy = high_order.four_param(xin[tbool], yin[tbool],xhst[idhst][tbool], yhst[idhst][tbool])
    cx,cy = high_order.four_param(xstf[tbool], ystf[tbool],hst['Xarc'][idhst][tbool], hst['Yarc'][idhst][tbool])
    xnew = (cx[0] + cx[1]*xstf + cx[2]*ystf)*hpix_scale
    ynew = (cy[0] + cy[1]*xstf + cy[2]*ystf)*hpix_scale

    #idx1 , idx2 , dm, dr = jay.match(stf['col1'], stf['col3'] ,stf['col5'], xnew, ynew, hst['Mag'], 8)                            
        
            
    
    return   xnew, ynew, stf['col0'], hst['Mag'][idhst], stf['col9']

def mk_reference_spline(lis_files, fits_files,  hst_tab_ref, tx, ty, outfile='Nspline_ref.txt'):
    '''
    generates a new refrence list that combines all of the input stars.
    this allows a solution based soley on NIRC2 data  to be computed
    '''
    
    xdict = {}
    ydict = {}
    mag_dict = {}

    for i, ff in enumerate(lis_files):
        xn, yn , name, mag, magN = match2hst_err_ret_hst(ff,fits_files[i] , hst_tab_ref ,tx, ty, spline=True)

        for i, n in enumerate(name):
            if n in xdict.keys():
                xdict[n].append(xn[i])
                ydict[n].append(yn[i])
                mag_dict[n].append(magN[i])

            else:
                xdict[n] = [xn[i]]
                ydict[n] = [yn[i]]
                mag_dict[n] = [magN[i]]


    x = []
    y = []
    xerr = []
    yerr = []
    name = []
    N = []
    mag = []
    
    for n in xdict.keys():
        x.append(np.mean(xdict[n]))
        xerr.append(np.std(xdict[n]))
        y.append(np.mean(ydict[n]))
        yerr.append(np.std(ydict[n]))
        name.append(n)
        mag.append(np.mean(mag_dict[n]))
        N.append(len(xdict[n]))

    x = np.array(x)
    xerr = np.array(xerr)
    y = np.array(y)
    yerr = np.array(yerr)
    name = np.array(name)
    N = np.array(N)
    mag = np.array(mag)
    
    #now write final reference in fits and ascii form
    tabout = Table(data=[name, x, xerr, y, yerr, mag, N])
    tabout.write(outfile, format='ascii.fixed_width')
    #tabout.write('NIRC2_reference.fits', format='fits')
    
def mk_reference_leg(lis_files,  dar_lis, hst_tab_ref, t, outfile_pre='NIRC2_leg_reference'):
    '''
    generates a new refrence list that combines all of the input stars.
    this allows a solution based soley on NIRC2 data  to be computed
    '''
    
    xdict = {}
    ydict = {}
    mag_dict = {}
    #lis_match_fits = Table.read('match_fits.lis', format='ascii.no_header')['col1']

    for i, ff in enumerate(lis_files):
        xn, yn , name, mag,magN = match2hst_err_ret_hst(ff,  dar_lis[i] , hst_tab_ref,t,t , spline=False)

        for i, n in enumerate(name):
            if n in xdict.keys():
                xdict[n].append(xn[i])
                ydict[n].append(yn[i])
                mag_dict[n].append(magN[i])

            else:
                xdict[n] = [xn[i]]
                ydict[n] = [yn[i]]
                mag_dict[n] = [magN[i]]


    x = []
    y = []
    xerr = []
    yerr = []
    name = []
    N = []
    mag = []
    
    for n in xdict.keys():
        x.append(np.mean(xdict[n]))
        xerr.append(np.std(xdict[n]))
        y.append(np.mean(ydict[n]))
        yerr.append(np.std(ydict[n]))
        name.append(n)
        mag.append(np.mean(mag_dict[n]))
        N.append(len(xdict[n]))

    x = np.array(x)
    xerr = np.array(xerr)
    y = np.array(y)
    yerr = np.array(yerr)
    name = np.array(name)
    N = np.array(N)
    mag = np.array(mag)
    
    #now write final reference in fits and ascii form
    tabout = Table(data=[name, x, xerr, y, yerr, mag, N])
    tabout.write(outfile_pre+'.txt', format='ascii.fixed_width')
    #tabout.write(outfile_pre+'.fits', format='fits')

        
        



def precision_stack(lis_files, hst_tab_ref, hst_tab_red, n_iter=3, order=3, plot_hist=False, pix_range=4, long_in_set=None, tab_format='fits'):
    '''
    go through all the starfinder , match into hst, find the deltas in the keck frame
    keep track of names of stars
    Need to clculate precisions, which requires stacking positions from epochs that are very close to each other on the sky
    TO do this, first try I will just match all frames to the longest list
    THen find all frames that have >20 matches ( from triangle), and also have average dr < pix_range pixels
    Then stack the matched stars.
    THen find precision, also make plots of M vs. error, whci means that the I need to hold onto the magnitudes and match them to each other
    Note, errors are retunrned in units of arcseconds!!!!!
    
    '''

    xstf = []
    ystf = []
    xref = []
    yref = []
    rms_x = []
    rms_y = []
    nstf = []
    mstf =[]
    #hst_tab = Table.read(hst_file, format='ascii')
    long_in = 0
    long_length=0
    lis_used = []
    
    for n in range(n_iter):
        xstf = []
        ystf = []
        xref = []
        yref = []
        hxerr = []
        hyerr = []
        for i, lis_i in enumerate(lis_files):
            x,y,xr,yr,name, hst_bool, m_idx, mag,hst_mag,  hst_xerr, hst_yerr = match2hst(lis_i, hst_tab_ref, hst_tab_red)

            if len(x) > 5:
                xstf.append(x)
                ystf.append(y)
                xref.append(hst_tab_red['Xarc'][hst_bool][m_idx])
                yref.append(hst_tab_red['Yarc'][hst_bool][m_idx])
                hxerr.append(hst_xerr)
                hyerr.append(hst_yerr)
                mstf.append(mag)
                nstf.append(name)
                lis_used.append(lis_i)
                if len(x)>long_length:
                    long_in=i


    #we have a reference now, need to match into it

    if long_in_set != None:
        long_in = long_in_set
    #create matching dictionary
    lis_bool= np.ones(len(lis_used), dtype='bool')

    fnum = 0
    fits_frames_index = []
    mname = []
    while np.sum(lis_bool) > 0:
        #import pdb;pdb.set_trace()
        m_dict = {}
        x_dict ={}
        y_dict ={}
        hste_dict = {}
        hst_pos_dict = {}
        #import pdb; pdb.set_trace()
        if lis_bool[fnum]:
            lis_bool[fnum] = False
            for i in range(len(xstf[fnum])):
                x_dict[nstf[fnum][i]] = [xstf[fnum][i]]
                y_dict[nstf[fnum][i]] = [ystf[fnum][i]]
                m_dict[nstf[fnum][i]] = [mstf[fnum][i]]
                hst_pos_dict[nstf[fnum][i]] = (xref[fnum][i],yref[fnum][i])
                hste_dict[nstf[fnum][i]] = (hxerr[fnum][i], hyerr[fnum][i])

            for ii in range(1,20):
                #go through next 20 frames to make sure that we get the next frames we are looking for, should only be 4 frames for our observing frame
                #import pdb;pdb.set_trace()
                if ii+fnum < len(xstf):
                    N, x1m, y1m, m1m, x2m, y2m, m2m = jay.miracle_match_briteN(xstf[fnum], ystf[fnum], mstf[fnum], xstf[ii+fnum], ystf[ii+fnum],mstf[ii+fnum], 60)
    
                    if N > 9:
                        print N, '  matches found'
                        #tranform coordinates for final matching
                        cx,cy = high_order.four_param(x2m, y2m, x1m,y1m)
                        #import pdb;pdb.set_trace()
                        if np.mean(x2m-x1m)**2+np.mean(y2m-y1m)**2 < pix_range**2:
                            lis_bool[ii+fnum] = False
                            xnew = cx[0] + cx[1]*xstf[ii+fnum] + cx[2]*ystf[ii+fnum]
                            ynew = cy[0] + cy[1]*xstf[ii+fnum] + cy[2]*ystf[ii+fnum]
                
                            idx1 , idx2 , dm, dr = jay.match(xstf[fnum], ystf[fnum], mstf[fnum], xnew, ynew, mstf[ii+fnum], 2)

                            #now I need to go through the names of the matched stars, and add coordinates for stars that are in the refernce list
                            for kk in range(len(idx2)):
                                if nstf[ii+fnum][idx2][kk] in x_dict.keys():
                                    x_dict[nstf[ii+fnum][idx2][kk]].append(xnew[idx2][kk])
                                    y_dict[nstf[ii+fnum][idx2][kk]].append(ynew[idx2][kk])
                                    m_dict[nstf[ii+fnum][idx2][kk]].append(mstf[ii+fnum][idx2][kk])

        #now calc precisions for all lists and write a series of files !
            print 'base of catalog ' ,fnum
            print 'number of matched measurements for an arbitrary star ', len(x_dict[x_dict.keys()[0]]) 
            xerr = []
            yerr = []
            x = []
            y = []
            mag = []
            N = []
            name_f = []
            xref1 = []
            yref1 = []
            xrerr = []
            yrerr = []
            
            #fits_frames_index.append(fnum)
            for i in x_dict.keys():
                if len(x_dict[i]) > 1:
                    
                    xerr.append(np.std(np.array(x_dict[i])))
                    x.append(np.mean(x_dict[i]))
                    yerr.append(np.std(np.array(y_dict[i])))
                    y.append(np.mean(y_dict[i]))
                    mag.append(np.mean(m_dict[i]))
                    N.append(len(x_dict[i]))
                    name_f.append(i)
                    xref1.append(hst_pos_dict[i][0])
                    yref1.append(hst_pos_dict[i][1])
                    xrerr.append(hste_dict[i][0])
                    yrerr.append(hste_dict[i][0])


            if x != []:
                #tab = Table(data=[name_f, x, xerr, y, yerr, mag, N], names = ['Name', 'x', 'xerr','y','yerr','mag','N'])
                tab = Table(data=[name_f, x, xerr, y, yerr, mag, N])
            
        
                mname.append('match_'+str(fnum)+'.fits')
                fits_frames_index.append(fnum)
                tab.write('match_'+str(fnum)+'.fits', format='fits')
                tab.write('match_'+str(fnum)+'.txt', format='ascii.fixed_width')
        fnum+=1

        
    out_fits = []
    lis_used = np.array(lis_used)
    for s in lis_used[fits_frames_index]:
        out_fits.append('../../Data/april/'+str(s).replace('_0.8_stf.lis', '.fits.gz'))


    tab2 = Table(data=[out_fits,mname ])
    tab2.write('first_fits_m.lis', format='ascii.no_header')
    return xerr, yerr, mag, N


        

    
def collate_pos(lis_files, hst_tab_ref ,DAR_fits,ref_scale, ref_angle,plot_hist=False, ap_dar=True, rev_x=False):
    '''
    arguements:
    lis_files: list of fileanmes that are the NIRC2 position catalogs
    hst_tab_ref: catalog of stellar positoins from hst in astopy table form
    DAR_fits: list of fits files which coorespond to the lis_files catalogs.  These are used to apply DAR to the HST catalogs
    ref_scale: scale of the refereence( HST) catlogs in arcseconds/pixel
    ref_angle: rotation angle of HST catalog from N being up and East to the West
    keyword args:
    ap_dar: Bool, if True, DAR is applied, else it is ignored.
    
    go through all the starfinder , match into hst, find the deltas in the keck frame
    then return this information to write single catalog of x, y, xref, yref ...
    NOTE THAT REFERENCE ERRORS ARE ALL RETURNED IN NIRC2 PIXEL SCALE 
    '''

    xstf = []
    ystf = []
    xref = []
    yref = []
    rms_x = []
    rms_y = []
    nam_out = []
    #hst_tab = Table.read(hst_file, format='ascii')
    frame_num = []
    frame_name = []
    
    t = None
    
    xstf = []
    ystf = []
    xref = []
    yref = []
    hmag = []
    smag = []
    xerr = []
    yerr = []
    hxerr = []
    hyerr = []
        
    for index, i in enumerate(lis_files):
           
        x,y,xr,yr, name, s_mag, fmag,hst_xerr, hst_yerr, stf_xerr, stf_yerr = match2hst_err(i, hst_tab_ref, DAR_fits[index],ref_angle=ref_angle,ref_scale=ref_scale, ap_dar=ap_dar, rev_x=rev_x)
        for ii in range(len(x)):
            xstf.append(x[ii])
            ystf.append(y[ii])
            xref.append(xr[ii])
            yref.append(yr[ii])
            hmag.append(fmag[ii])
            smag.append(s_mag[ii])
            xerr.append(stf_xerr[ii])
            yerr.append(stf_yerr[ii])
            hxerr.append(hst_xerr[ii])
            hyerr.append(hst_yerr[ii])
            nam_out.append(name[ii])
            frame_num.append(index)
            frame_name.append(i)


    
    return np.array(xstf),np.array(xerr), np.array(ystf),np.array(yerr), np.array(xref),np.array(hxerr), np.array(yref),np.array(hyerr), t ,hmag, nam_out, frame_num, frame_name

def leg2lookup(t, plot=False):

    xrange = np.linspace(0,1023,num=1024)
    yrange = xrange
    outx = np.zeros((len(xrange),len(yrange)))
    outy = np.zeros((len(xrange),len(yrange)))
    
    for i in range(len(xrange)):
        xin = np.zeros(yrange.shape) + xrange[i]
        xn,yn = t.evaluate(xin, yrange)
        for y in yrange:
            
            outx[:,i] = xn
            outy[:,i] = yn

    if plot:
        plt.gray()
        plt.figure(1)
        plt.imshow(outx)
        plt.colorbar()
        plt.title('X distortion')
        plt.figure(2)
        plt.imshow(outy)
        plt.title('Y distortion')
        plt.colorbar()
        plt.show()
    return outx,outy
            
    


def search_for_mat(hst,stf,fac=1, num=50):
    hst_pix = .05 # arcseconds/pixel
    fov = 10 / hst_pix * fac
    bins_x = int((np.max(hst['Xarc'])-np.min(hst['Xarc']))/fov)
    bins_y = int((np.max(hst['Yarc'])-np.min(hst['Yarc']))/fov)
    xr = np.linspace(np.min(hst['Xarc']),np.max(hst['Xarc']),num=bins_x)
    yr = np.linspace(np.min(hst['Yarc']),np.max(hst['Yarc']),num=bins_y)

    for i in range(len(xr)-1):
        for j in range(len(yr)-1):
            hst_bool =  (hst['Xarc']>xr[i])*(hst['Xarc'] < xr[i+1]) *(hst['Yarc']>yr[j])*(hst['Yarc']<yr[j+1])
            N, x1m, y1m, m1m, x2m, y2m, m2m = jay.miracle_match_briteN(stf['col4'], stf['col5'], stf['col2'], hst['Xarc'][hst_bool], hst['Yarc'][hst_bool], hst['Mag'][hst_bool], num)
            if N > 10:
                #import pdb;pdb.set_trace()
                #print 'found ptential match'
                return N,x1m,y1m,m1m,x2m,y2m,m2m
                


def smooth_spline(x,y,xref,yref,plot=False):


    #import pdb; pdb.set_trace()
    dumx = xref - x
    dumy = yref - y 
    splinex = Spline(x,y,dumx, s=50)
    spliney = Spline(x,y,dumy, s=50)

    xrange = np.linspace(0,1024,num=1024)
    yrange = xrange
    outx = np.zeros((len(xrange),len(yrange)))
    outy = np.zeros((len(xrange),len(yrange)))
    
    for i in range(len(xrange)):
        xin = np.zeros(yrange.shape) + xrange[i]
        xn = splinex.ev(xin,yrange)
        yn = spliney.ev(xin, yrange)
        for y in yrange:
            
            outx[i,:] = xn
            outy[i,:] = yn

    if plot:
        plt.gray()
        plt.figure(1)
        plt.imshow(outx)
        plt.colorbar()
        plt.title('X distortion')
        plt.figure(2)
        plt.imshow(outy)
        plt.title('Y distort2ion')
        plt.colorbar()
        plt.show()

    return outx,outy

def spline2lookup(splinex,spliney):
    xrange = np.linspace(0,1023,num=1024)
    #import pdb;pdb.set_trace()
    yrange = xrange
    outx = np.zeros((len(xrange),len(yrange)))
    outy = np.zeros((len(xrange),len(yrange)))

    for i in range(len(xrange)):
        xin = np.zeros(yrange.shape) + xrange[i]
        xn = splinex.ev(xin,yrange)
        yn = spliney.ev(xin, yrange)
        for y in yrange:
            
            outx[:,i] = xn
            outy[:,i] = yn

    return outx,outy
def calc_dist(xapr,yapr,xaprref,yaprref,xmay,ymay,xmayref,ymayref, yeldax,yelday):
    #first try fitting one set of data with progressively higher order polynomials
    
    t1 = high_order.LegTransform(xapr,yapr,xaprref-xapr,yaprref-yapr, 3)
    t2 = high_order.LegTransform(xapr,yapr,xaprref-xapr,yaprref-yapr, 4)
    t3 = high_order.LegTransform(xapr,yapr,xaprref-xapr,yaprref-yapr, 5)
    outx1,outy1 = plot_dist(t1,plot=False)
    outx2,outy2 = plot_dist(t2,plot=False)
    outx3,outy3 = plot_dist(t3,plot=False)

    plt.clf()
    plt.gray()
    plt.figure(1)
    plt.clf()
    plt.subplot(121)
    plt.imshow(yeldax)
    plt.colorbar()
    plt.title('Yelda X distortion')
    plt.subplot(122)
    plt.imshow(yelday)
    plt.colorbar()
    plt.title('Yelda Y distortion')
    plt.savefig('Yelda_distortion')
    
    plt.figure(2)
    plt.clf()
    plt.subplot(121)
    plt.imshow(outx1.T)
    plt.colorbar()
    plt.title('X April Leg 3rd order')
    plt.subplot(122)
    plt.imshow(outy1.T)
    plt.title('Y April Leg 3rd order')
    plt.colorbar()
    plt.savefig('April_distortion_leg_3rd')

    plt.figure(3)
    plt.clf()
    plt.subplot(121)
    plt.imshow(outx2.T)
    plt.colorbar()
    plt.title('X April Leg 4th order')
    plt.subplot(122)
    plt.imshow(outy2.T)
    plt.title('Y April Leg 4th order')
    plt.colorbar()
    plt.savefig('April_distortion_leg_4th')

    plt.figure(4)
    plt.clf()
    plt.subplot(121)
    plt.imshow(outx3.T)
    plt.colorbar()
    plt.title('X April Leg 5th order')
    plt.subplot(122)
    plt.imshow(outy3.T)
    plt.title('Y April Leg 5th order')
    plt.colorbar()
    plt.savefig('April_distortion_leg_5th')
    
    
    
    
    
    #now plot residuals
    plt.figure(5)
    plt.clf()
    plt.subplot(121)
    plt.title('X (yelda-Leg 3rd)')
    offset = np.average(yeldax-outx1.T)
    plt.imshow(yeldax-outx1.T - offset)
    plt.colorbar()
    plt.subplot(122)
    plt.title('Y (yelda-Leg 3rd)')
    offset = np.average(yelday-outy1.T)
    plt.imshow(yelday-outy1.T - offset)
    plt.colorbar()
    plt.savefig('yelda-apr_3rd_order')

    plt.figure(6)
    plt.clf()
    plt.subplot(121)
    plt.title('X (Leg 3rd - Leg 4th)')
    
    plt.imshow(outx1.T-outx2.T)
    plt.colorbar()
    plt.subplot(122)
    plt.title('Y (Leg 3rd - Leg 4th')
    
    plt.imshow(outy1.T-outy2.T-offset)
    plt.colorbar()
    plt.savefig('apr_leg_3rd-apr_Leg_4th')


    plt.figure(7)
    plt.clf()
    plt.subplot(121)
    plt.title('X (Leg 3rd - Leg 5th)')
    plt.imshow(outx1.T-outx3.T)
    plt.colorbar()
    plt.subplot(122)
    plt.title('Y (Leg 3rd - Leg 5th')
    plt.imshow(outy1.T-outy3.T)
    plt.colorbar()
    plt.savefig('apr_leg_3rd-apr_Leg_5th')
    


def calc_sig_dist(x, y,xref,yref, n_iter=3, sig_clip=3, order=3, weights=None, ret_bool=False, poly_type=1):


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
    
def mk_quiver(x,y,dx,dy, title_s='April', scale=50):
    
    q = plt.quiver(x, y, dx, dy, scale = scale)
    qk = plt.quiverkey(q,1050, 1050, 3 , '3 pixel', coordinates='data', color='red')
    plt.xlim(-100,1125)
    plt.ylim(-100,1124)
    plt.title(title_s)
    plt.show()    
    


def sig_trim(x,xerr,y,yerr,xref,xreferr,yref,yreferr,names,mag, sig_fac=3 , num_section=9):
    dx = xref - x
    dy = yref - y

    xout = []
    yout = []
    xrout = []
    yrout = []
    xeout = []
    yeout = []
    xreout =[]
    yreout = []
    nout = []
    mout = []
    
    bins = np.linspace(0,1024, num=num_section)

    for i in range(len(bins)-1):
        for j in range(len(bins)-1):
            sbool = (x > bins[i])*(x<bins[i+1])*(y>bins[j])*(y<bins[j+1])
            print 'numer of stars in this section are ', np.sum(sbool)
            ave_x, sig_x, nclipx = statsIter.mean_std_clip(dx[sbool], clipsig=3.0, return_nclip=True)
            ave_y, sig_y, nclipy = statsIter.mean_std_clip(dy[sbool], clipsig=3.0, return_nclip=True)
            good_bool = (dx < ave_x + sig_fac * sig_x)*(dx > ave_x - sig_fac * sig_x)*(dy < ave_y + sig_fac *sig_x)*(dy > ave_y - sig_fac * sig_x) * sbool
            #import pdb; pdb.set_trace()
            for ii in range(np.sum(good_bool)):
                xout.append(x[good_bool][ii])
                yout.append(y[good_bool][ii])
                xrout.append(xref[good_bool][ii])
                yrout.append(yref[good_bool][ii])
                xeout.append(xerr[good_bool][ii])
                yeout.append(yerr[good_bool][ii])
                xreout.append(xreferr[good_bool][ii])
                yreout.append(yreferr[good_bool][ii])
                nout.append(names[good_bool][ii])
                mout.append(mag[good_bool][ii])
                


    print 'umber of stars cliiped ', len(x) - len(xout)
    return  np.array(xout),np.array(xeout) ,np.array(yout), np.array(yeout), np.array(xrout), np.array(xreout), np.array(yrout), np.array(yreout), np.array(nout) , np.array(mout)
    

def search_smooth_spline(pos_txt='april_pos.txt'):
    '''
    searches through potential uses of
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


def match_by_name(n1,n2, ret_both=False):
    '''
    returns indexes for n2 that match to n1
    assumes that all of n1 are in n2
    n1 and n2 should be arrays of strings
    '''

    id1 = []
    id2 = []
    for i in range(len(n1)):
        match = False
        j = 0
        while not match:

            
            if n1[i] == n2[j]:
                match=True
                id2.append(j)
                id1.append(i)
            elif len(n2)-1 == j:
                #give up
                match = True
            else:
                j+=1



    if ret_both:
        return id1, id2
    else:
        return id2

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
    t, dx, dy, sbool = calc_sig_dist( tab['x'][gbool],  tab['y'][gbool],  tab['xr'][gbool] ,  tab['yr'][gbool], weights=weights[gbool], n_iter=n_iter_fit, order=order, ret_bool=True, poly_type=poly_type)
    if lookup:
        outx, outy = leg2lookup(t)
        return t, outx, outy, dx, dy, sbool
    else:
        return t, dx, dy, gbool, sbool
    
    #now fit, but need to be careful with errors, as the ref errors are still in HST pixel scale 

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


def fit_dist_splineLSQ(pos_txt='april_pos.txt',  n_iter_fit=1, nstring=None, nknots=6,order=3 ,lookup=True, wtype=1):
    '''
    fits distortion with legendre polynomial
    does spatial sigma clipping before hand
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

    
def stack_from_pos(pos_txt, t):
    tab = Table.read(pos_txt, format='ascii.fixed_width')
    xnew , ynew = t.evaluate(tab['x'], tab['y'])

    xdict = {}
    ydict = {}
    for i, name in enumerate(tab['name']):
        if name in xdict.keys():
            xdict[name].append(xnew[i])
            ydict[name].append(ynew[i])

        else:
            xdict[name] = [xnew[i]]
            ydict[name] = [ynew[i]]

    return xdict, ydict
    
def match_and_write(outfile='april_pos.txt', fits_lis='first_fits_m.lis'):
    
    hst_ref =  Table.read('../../../M53_F814W/F814_pix_err.dat', format='ascii')
    #lis_apr = Table.read('lis.lis', format='ascii.no_header')
    lis_fits = Table.read(fits_lis, format='ascii.no_header')
    #DAR_fits = Table.read(fits_lis, format='ascii.no_header')

    #angle is orientation of the HST frame, from header in this data (M53) is -102.5 degrees

    x,xerr,y,yerr,xref,xreferr,yref,yreferr,t,hmag, name,frame_num, frame_name = collate_pos(lis_fits['col2'], hst_ref, lis_fits['col1'], .05 , -102.5 , rev_x=True, ap_dar=True)
    outtab = Table(data=[name, hmag, x, xerr, y, yerr, xref , xreferr, yref , yreferr, frame_num, frame_name], names=['name','mag', 'x', 'xerr', 'y', 'yerr', 'xr', 'xrerr', 'yr', 'yrerr', 'frame_num', 'frame_name'])

    outtab.write(outfile, format='ascii.fixed_width')

def match_and_write2(reffile='NIRC2_leg_reference.txt',fits_lis='first_fits_m.lis', outfile='april_pos_leg_r2.txt', ap_dar=True):
    
    hst_ref =  Table.read(reffile, format='ascii.fixed_width')
    #lis_apr = Table.read('lis.lis', format='ascii.no_header')
    lis_fits = Table.read(fits_lis, format='ascii.no_header')

    #DAR_fits = Table.read(fits_lis, format='ascii.no_header')

    x,xerr,y,yerr,xref,xreferr,yref,yreferr,t,hmag, name, frame_num, frame_name = collate_pos(lis_fits['col2'], hst_ref,lis_fits['col1'], 1.0, 0)
    outtab = Table(data=[name, hmag, x, xerr, y, yerr, xref , xreferr, yref , yreferr, frame_num, frame_name], names=['name','mag', 'x', 'xerr', 'y', 'yerr', 'xr', 'xrerr', 'yr', 'yrerr', 'frame_num', 'frame_name'])
    

    outtab.write(outfile, format='ascii.fixed_width')
def plot_dist(out1, out2, title1='Legendre', title2='Spline', title3='Difference', title4='Difference', vmind=-.5, vmaxd=.5, outfile=None, show=False):
    plt.figure(1)
    plt.clf()
    plt.gray()
    plt.subplot(131)
    vmax = np.max(out1)
    vmin = np.min(out1)
    plt.imshow(out1, vmax=vmax, vmin=vmin)
    plt.colorbar()
    plt.title(title1)

    plt.subplot(132)
    plt.imshow(out2, vmax=vmax, vmin=vmin)
    plt.colorbar()
    plt.title(title2)

    plt.subplot(133)
    plt.title(title3)
    #plt.imshow(out1-out2, vmin=vmind, vmax=vmaxd)
    plt.imshow(out1-out2)
    plt.colorbar()

    if outfile != None:
        plt.savefig(outfile)
    
    plt.figure(2)
    plt.hist((out1-out2).flatten(), bins=100)
    plt.title(title4)
    if show:
        plt.show()

def plot_lookup_diff(l1x,l1y, l2x,l2y, spacing=24):
    #plt.figure(10)
    plt.clf()
    indices = range(0,1024,spacing)
    coos = np.meshgrid(indices, indices)
    dx = np.zeros(coos[0].shape)
    dy = np.zeros(coos[0].shape)
    
    for i in range(len(indices)):
        for j in range(len(indices)):
            dx[i,j] = l1x[indices[i],indices[j]] - l2x[indices[i],indices[j]]
            dy[i,j] = l1y[indices[i],indices[j]] - l2y[indices[i],indices[j]]
    
    coos = np.meshgrid(indices, indices)
    q = plt.quiver(coos[0], coos[1], dx, dy, scale = 10)
    qk = plt.quiverkey(q,1050, 1050, .5 , '.5 pixel', coordinates='data', color='red')
    
def plot_lookup(lx,ly, spacing=24, scale=10, scale_size=.5):
    #plt.figure(10)
    plt.clf()
    indices = range(0,1024,spacing)
    coos = np.meshgrid(indices, indices)
    dx = np.zeros(coos[0].shape)
    dy = np.zeros(coos[0].shape)
    
    for i in range(len(indices)):
        for j in range(len(indices)):
            dx[i,j] = lx[indices[i],indices[j]] 
            dy[i,j] = ly[indices[i],indices[j]] 
    
    coos = np.meshgrid(indices, indices)
    q = plt.quiver(coos[0], coos[1], dx, dy, scale = scale)
    qk = plt.quiverkey(q,1050, 1050, scale_size , str(scale_size)+' pixels', coordinates='data', color='red')
    
def mk_quiver_from_txt(pos_txt='april_pos.txt', title_s='April'):

    tab = Table.read(pos_txt, format='ascii.fixed_width')
    #x, xerr, y, yerr, xr, xrerr, yr, yrerr, name, mag = sig_trim(tab['x'], tab['xerr'], tab['y'], tab['yerr'], tab['xr'], tab['xrerr'], tab['yr'], tab['yrerr'], tab['name'], tab['mag'])
    x  = tab['x']
    y = tab['y']
    xr = tab['xr']
    yr = tab['yr']
    dx = xr - x
    dy = yr -y 
    q = plt.quiver(x, y, dx, dy, scale = 50)
    qk = plt.quiverkey(q,1050, 1050, 2 , '2 pixel', coordinates='data', color='red')
    plt.xlim(-100,1125)
    plt.ylim(-100,1124)
    plt.title(title_s)
    plt.show()    

def mk_quiver_from_txt_dist(t, pos_txt='april_pos.txt', title_s='April Distortion Corrected', scale=50, scale_size=.5):

    tab = Table.read(pos_txt, format='ascii.fixed_width')
    #x, xerr, y, yerr, xr, xrerr, yr, yrerr, name, mag = sig_trim(tab['x'], tab['xerr'], tab['y'], tab['yerr'], tab['xr'], tab['xrerr'], tab['yr'], tab['yrerr'], tab['name'], tab['mag'])
    dx, dy = t.evaluate(tab['x'],tab['y'])
    x = tab['x'] + dx
    y = tab['y'] + dy
    
    #import pdb;pdb.set_trace()
    xr = tab['xr']
    yr = tab['yr']
    
    dx = xr - x
    dy = yr -y 
    q = plt.quiver(x, y, dx, dy, scale = scale)
    qk = plt.quiverkey(q,1050, 1050, scale_size , str(scale_size)+' pixel', coordinates='data', color='red')
    plt.xlim(-100,1125)
    plt.ylim(-100,1124)
    plt.title(title_s)
    plt.show()  

def mk_quiver_resid(x,y,dx,dy, scale=50, scale_size=.5, title_s='', color='black'):

    #plt.figure(1)
    #plt.clf()
    q = plt.quiver(x, y, dx, dy, scale = scale, color=color)
    qk = plt.quiverkey(q,1050, 1050, scale_size , str(scale_size)+' pixel', coordinates='data', color='red')
    plt.xlim(-100,1125)
    plt.ylim(-100,1124)
    plt.title(title_s)
    plt.show()
     


    
def find_best_fit_spline(yeldax, yelday , pos_txt='sig_trimapril_pos.txt', wtype=3):
    '''
    go throigh orfer of legendre polynomial
    look at residuals / errors
    Fit with gaussian with sigma of 1
    measure chi squared
    print those number on a plot, call it a day
    '''

    #num_knots = range(2,10)
    num_knots = np.linspace(600,1100, num=20)
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


        plot_dist( yeldax,outx, title2='Spline X Nknots:'+str(i)[:5], title1='Yelda', title3='Difference', title4='Difference', vmind=-.5, vmaxd=.5, outfile='Dist_sol_splineX_'+str(i)[:5]+'.png')
        plot_dist( yelday,outy, title2='Spline Y Nknots:'+str(i)[:5], title1='Yelda', title3='Difference', title4='Difference', vmind=-.5, vmaxd=.5, outfile='Dist_sol_splineY_'+str(i)[:5]+'.png')
            
        fit_p  = fitting.LevMarLSQFitter()
        
        gy = models.Gaussian1D(mean=0, stddev=1.0)
        gy.mean.fixed =True
        #gy.stddev.fixed = True

        gx = models.Gaussian1D(mean=0, stddev=1.0)
        gx.mean.fixed =True
        #gx.stddev.fixed = True

        mx = fit_p(gx , bcenx, xN)
        my = fit_p(gy , bceny , yN)

        chix = np.sum((mx(bcenx) - xN)**2/ mx(bcenx))
        chiy = np.sum((my(bceny) - yN)**2/ my(bceny))

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

        plt.figure(4)
        plt.clf()
        plot_lookup_diff(outx, outy, yeldax, yelday)
        plt.title('Difference Legendre and Yelda')
        plt.savefig('Spline'+str(i)[:5]+'_resid_yelda.png')

        plt.figure(1)
        plt.clf()
        plt.scatter(bcenx, xN)
        plt.plot(bcenx, mx(bcenx))
        plt.text(np.min(bcenx)+1, np.max(xN)/2.0, r'$\chi^{2}$: '+str(chix)[:5])
        plt.text(np.min(bcenx)+1, np.max(xN)/2.0-25, r'$\sigma$:'+str(mx.stddev.value)[:6])
        plt.text(np.min(bcenx)+1, np.max(xN)/2.0-50,'S: '+str(i))
        plt.title('X residual Nknots'+str(i)[:5])
        plt.xlabel('residual / error')
        plt.ylabel('N')
        plt.savefig('Spline_x_resid_ord'+str(i)[:5]+'.png')

        plt.figure(2)
        plt.clf()
        #plt.hist(dyn, bins=100)
        plt.scatter(bceny, yN)
        plt.plot(bceny, my(bceny))
        plt.text(np.min(bceny)+1, np.max(yN)/2.0, r'$\chi^{2}$: '+str(chiy)[:5])
        plt.text(np.min(bceny)+1, np.max(yN)/2.0-25, r'$\sigma$:'+str(my.stddev.value)[:6])
        plt.text(np.min(bceny)+1, np.max(yN)/2.0-50,'S: '+str(i))
        plt.title('Y residual S:'+str(i)[:5])
        plt.xlabel('residual / error')
        plt.ylabel('N')
        plt.savefig('Spline_y_resid_ord'+str(i)[:5]+'.png')
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
        

def sig_trim_ref(pos_txt):

    tab = Table.read(pos_txt, format='ascii.fixed_width')
    x, xerr, y, yerr, xr, xrerr, yr, yrerr, name, mag = sig_trim(tab['x'], tab['xerr'], tab['y'], tab['yerr'], tab['xr'], tab['xrerr'], tab['yr'], tab['yrerr'], tab['name'], tab['mag'], )

    
    outtab = Table(data=[name, mag, x, xerr, y, yerr, xr, xrerr, yr, yrerr], names=['name','mag', 'x', 'xerr', 'y', 'yerr', 'xr', 'xrerr', 'yr', 'yrerr'])
    outtab.write('sig_trim'+pos_txt, format='ascii.fixed_width')



def applyDAR_coo(fits, x_h, y_h):
    """
    Input a starlist in x=RA (+x = west) and y=Dec (arcseconds) taken from
    space and introduce differential atmospheric refraction (DAR). The amount
    of DAR that is applied depends on the header information in the input fits
    file. The resulting output starlist should contain what was observed
    after the starlight passed through the atmosphere, but before the
    starlight passed through the telescope. Only achromatic DAR is 
    applied in this code.

    The output file has the name <fitsFile>_acs.lis and is saved to the
    current directory.

    Added funcionallity from nirc2.util.dar dunction:
    orient ang assumes an orientation of the input HST starlist
    hst_pix_scale simply converts the hst input ref values to arcseconds
    """
    
    # Get header info
    hdr = pyfits.getheader(fits)

    
    effWave = hdr['EFFWAVE']
    elevation = hdr['EL']
    lamda = hdr['CENWAVE']
    airmass = hdr['AIRMASS']
    parang = hdr['PARANG']

    date = hdr['DATE-OBS'].split('-')
    year = int(date[0])
    month = int(date[1])
    day = int(date[2])

    utc = hdr['UTC'].split(':')
    hour = int(utc[0])
    minute = int(utc[1])
    second = int(math.floor(float(utc[2])))

    utc = datetime.datetime(year, month, day, hour, minute, second)
    utc2hst = datetime.timedelta(hours=-10)
    hst = utc + utc2hst

    (refA, refB) = keckDARcoeffs(effWave, hst.year, hst.month, hst.day,
                                 hst.hour, hst.minute)

    tanz = math.tan(math.radians(90.0 - elevation))
    tmp = 1.0 + tanz**2
    darCoeffL = tmp * (refA + 3.0 * refB * tanz**2)
    darCoeffQ = -tmp * (refA*tanz +
                            3.0 * refB * (tanz + 2.0*tanz**3))

    #import pdb; pdb.set_trace()
    # Convert DAR coefficients for use with arcseconds
    darCoeffL *= 1.0
    darCoeffQ *= 1.0 / 206265.0
    
    # Lets determine the zenith and horizon unit vectors for
    # this image. The angle we need is simply the parallactic
    # (or vertical) angle since ACS images are North Up already.
    pa = math.radians(parang)

    ##########
    #
    # Read in the starlist
    #
    ##########
    #_list = Table.read(spaceStarlist, format='ascii')
    #cols = _list.columns.keys()
    #names = [_list[ss][0].strip() for ss in range(len(_list))]
    #mag = _list[cols[1]]
    #date = _list[cols[2]]
    #x = _list[cols[3]] # RA in arcsec
    #y = _list[cols[4]]
    #xe = _list[cols[5]]
    #ye = _list[cols[6]]
   
    
    x = x_h
    y = y_h
    # Magnify everything in the y (zenith) direction. Do it relative to
    # the first star. Even though dR depends on dzObs (ground observed dz),
    # it is a small mistake and results in less than a 10 micro-arcsec
    # change in dR.
    dx = x - x[0]
    dy = y - y[0]

    # Rotate coordinates CW so that the zenith angle is at +ynew
    sina = math.sin(pa)
    cosa = math.cos(pa)
    xnew1 = dx * cosa + dy * sina
    ynew1 = -dx * sina + dy * cosa

    # Apply DAR
    xnew2 = xnew1
    ynew2 = ynew1 * (1.0 - darCoeffL) - ynew1 * np.abs(ynew1) * darCoeffQ

  

    # Rotate coordinates CCW back to original angle
    xnew3 = xnew2 * cosa - ynew2 * sina
    ynew3 = xnew2 * sina + ynew2 * cosa

    xnew = xnew3 + x[0]
    ynew = ynew3 + y[0]

    
    return xnew, ynew


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
       


def plot_err_on_mean(ref_txt, mcut=10, run=1):

        nref = Table.read(ref_txt, format='ascii.fixed_width')
        plt.figure(1)
        plt.clf()
        eX =  nref['col2']/np.sqrt(nref['col6'])
        eY =  nref['col4']/np.sqrt(nref['col6'])

        mbool = nref['col5'] < mcut   
        plt.semilogy(nref['col5'], eX, 'o', label='x')
        plt.semilogy(nref['col5'], eY, 'o', label='y')
        plt.legend(loc='upper right')
        plt.title('Error on Mean of Stack Pass '+str(run))
        plt.text(np.min(nref['col5'])+.5, 10**-2, 'mean errorX:'+str(np.mean(eX[mbool])))
        plt.text(np.min(nref['col5'])+.5, 5*10**-2, 'mean errorY:'+str(np.mean(eY[mbool])))
        plt.axvline(mcut)
        plt.xlabel('Mag')
        plt.ylabel('Error (arcseconds')


def iter_sol_leg(iter=5):
    #assumes intiial
    initial_data = 'april_pos.txt'
    #spaitally sigma trim the positions
    sig_trim_ref(initial_data)
    #now fit the distortion, I choose 5th order legendre polynomials
    ref_base = 'Nref_leg5'
    data_base = 'pos_leg5'
    t, dx5n, dy5n, sbooln, b2= fit_dist(pos_txt='sig_trim'+initial_data,order=5, n_iter_fit=1, lookup=False)
    tab_match = Table.read('first_fits_m.lis', format='ascii.no_header')
    hst = Table.read('../../../M53_F814W/F814_pix_err.dat', format='ascii')

    run_base = 'Nref_leg5'
    for i in range(iter):
        #first need new reference (using the above distortion solution)
        mk_reference_leg(tab_match['col2'],tab_match['col1'], hst, t, outfile_pre=run_base+str(i))
        #now need new data table, based on the new reference
        match_and_write2(reffile=run_base+str(i)+'.txt',outfile=data_base+str(i)+'.txt')
        #created a new reference, need to sigma trim it
        sig_trim_ref(data_base+str(i)+'.txt')
        #now can fit distortion using this as the reference
        t, dx5n, dy5n, sbooln, b2 = fit_dist(pos_txt='sig_trim'+data_base+str(i)+'.txt',order=5, n_iter_fit=1, lookup=False)

        #save tranform objects in case I want them later
        f= open(run_base+str(i)+'.trans', 'w')
        pickle.dump(t, f)

        #now we have a new distortion solution, so we return to step 1



def writefits2txt(lis_txt):
    for i in range(len(lis_txt)):
        t1 = Table.read(lis_txt[i], format='ascii.fixed_width')
        t1.write(lis_txt[i].replace('.txt', '.fits'), format='fits')




def rot_lis(x, y, angle):
    '''
    rotates x and y by the angle givem
    angle should be in degrees
    '''
    ang = np.deg2rad(angle)
    xnew = x * np.cos(ang) - y * np.sin(ang)
    ynew = x * np.sin(ang) + y * np.cos(ang)
    return xnew, ynew
