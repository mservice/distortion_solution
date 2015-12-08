from astropy.table import Table
from jlu.astrometry.align import align
from jlu.astrometry.align import jay
from jlu.astrometry import high_order_class as high_order
import pickle
import matplotlib.pyplot as plt
import numpy as np
from nirc2.reduce.dar import nirc2dar
import math
import matplotlib.animation as manimation
from visvis.vvmovie.images2gif import writeGif
from scipy.misc import imread
from distortion_solution import match_trim


def go(plot=True, trans_file='Nref_leg69.trans'):
    lis_stf = Table.read('lis.lis', format='ascii.no_header')
    lisd = Table.read('lis_dn.lis', format='ascii.no_header')
    darlis = Table.read('darn.lis', format='ascii.no_header')
    t = pickle.load(open(trans_file, 'r'))
    x, y, dx, dy = comp_starlists(lis_stf['col1'], lisd['col1'], t, darlis['col1'], plot=plot)

    #now make some plots
    plt.figure(1)
    plt.clf()
    plt.hist(dx, bins=100, label='x', alpha=.5)
    plt.hist(dy, bins=100, label='y', alpha=.5)
    print 'mean, std x:',np.mean(dx), np.std(dx)
    print 'mean, std y:',np.mean(dy), np.std(dy)
    plt.title('Difference between Correcting after STF and before')
    plt.ylabel('N')
    plt.xlabel('difference (pixels)')
    plt.legend(loc='upper left')

    plt.figure(2)
    plt.clf()
    q = plt.quiver(x, y, dx, dy, scale=4)
    qk = plt.quiverkey(q, 1050, 1050, 0.5, '0.5 pixels', color='red', coordinates='data')

    #import pdb;pdb.set_trace()
    plt.show()

    return x, y, dx, dy
    


def comp_starlists(lis_lis_stf, lis_lis_d, t, dar_lis, rad_tol=1, plot=True, movie=True):
    '''
    arguments
    lis_lis_stf is  list of starfinder catalogs that were distortion corrected before running starfinder
    lis_lis_d are the catalogs from runnign starfinder on the distorted images
    the two lists are assumed to be from the same frame, but not prematched
    t is the distoriton tranfomrion to apply to the catalogs in lis_lis_d
    if plot=True, it also plots every 10 frames
    '''
    

    x = []
    y = []
    dx = []
    dy = []

    xl = []
    yl = []
    dxl = []
    dyl = []
    
    for i in range(len(lis_lis_stf)):

        tabd = Table.read(lis_lis_d[i],format='ascii.no_header')
        tabs = Table.read(lis_lis_stf[i], format='ascii.no_header')
       
        #first correct the distortion
        xdiff, ydiff = t.evaluate(tabd['col4'], tabd['col5'])
        xdc = tabd['col4'] + xdiff
        ydc = tabd['col5'] + ydiff

        #now apply the DAR correction
        (pa, darCoeffL, darCoeffQ) = nirc2dar(dar_lis[i])
  

    
        sina = math.sin(pa)
        cosa = math.cos(pa)

        xnew2 =  xdc * cosa + ydc * sina
        ynew2 = -xdc * sina + ydc * cosa

        # Apply DAR correction along the y axis
        xnew3 = xnew2
        ynew3 = ynew2*(1 + darCoeffL) + ynew2*np.abs(ynew2)*darCoeffQ

        # Rotate coordinates counter-clockwise by PA back to original
        xdar = xnew3 * cosa - ynew3 * sina
        ydar = xnew3 * sina + ynew3 * cosa

        #import pdb; pdb.set_trace()
        
        #now do matching
        
        idx1, idx2, dr, dm = align.match(xdar, ydar, tabd['col2'], tabs['col4'], tabs['col5'], tabs['col2'], rad_tol)
        if len(idx1) < 3:
            import pdb; pdb.set_trace()
        
        #now we have matches, add them to the list?

        for ii in range(len(idx1)):
            x.append(xdar[idx1[ii]])
            y.append(ydar[idx1[ii]])
            dx.append(tabs['col4'][idx2[ii]]-xdar[idx1[ii]])
            dy.append(tabs['col5'][idx2[ii]]-ydar[idx1[ii]])
        xl.append(xdar[idx1])
        yl.append(ydar[idx1])
        dxl.append(tabs['col4'][idx2]-xdar[idx1])
        dyl.append(tabs['col5'][idx2]-ydar[idx1])

        if plot and 1.0*i % 10 == 0:
            plt.figure(i)
            plt.clf()
            q = plt.quiver(xdar[idx1], ydar[idx1], tabs['col4'][idx2]-xdar[idx1], tabs['col5'][idx2] - ydar[idx1], scale=4)
            qk = plt.quiverkey(q, 1050, 200, .5, '0.5 pixel', color='red', coordinates='data')
            plt.title('Difference for lis '+lis_lis_stf[i])
            #import pdb;pdb.set_trace()
            plt.show()
            


    if movie:
        mk_movie(xl, yl, dxl, dyl, mname='diff_stf_dist_mean_sub.mp4', scale=1, scale_size=.1)
        mk_movie(xl, yl, dxl, dyl, mname='diff_stf_dist.mp4', sub_ave=False)
    return np.array(x), np.array(y), np.array(dx), np.array(dy)




def mk_movie(xl, yl,dxl, dyl, mname='diff_stf_dist.mp4', mag=None, scale=5, scale_size=.5 , sub_ave=True ):
    
   
    
    rootim = 'im'

    if sub_ave:
        #dxl = np.array(dxl)
        #dyl = np.array(dyl)
        dxln = []
        dyln = []
        for i in range(len(dxl)):
            dxln.append(dxl[i] - np.mean(dxl[i]))
            dyln.append(dyl[i] - np.mean(dyl[i]))
        dxl = dxln[:]
        dyl = dyln[:]
        
   
    names = []
    
    for i in range(len(xl)):
        plt.clf()
       
        q = plt.quiver(xl[i],yl[i],dxl[i],dyl[i], scale=scale)
        plt.xlim(-200,1200)
        plt.ylim(-200,1200)
        qk = plt.quiverkey(q,1050, 300, scale_size, str(scale_size)+' pixels', coordinates='data', color='red')
        plt.title('Frame {0:2d}'.format(i)) 
        fname = rootim + str(i) +'.png'
        
        #calc mean resid in x and y
        xresid = np.mean(np.abs(dxl[i]))
        yresid = np.mean(np.abs(dyl[i]))
        plt.text(1050, 400, 'x:'+str(xresid)[:4]+' pix')
        plt.text(1050, 450, 'y:'+str(yresid)[:4]+' pix')
        names.append(fname)
        plt.savefig(fname)

    ims = []
    
    for i in names:
        ims.append(imread(i))

    #writeGif(gifname, ims)
    gifim = []

    an_ims = []
    plt.clf()
    fig = plt.figure()
    plt.clf()
    for i in range(len(ims)):
        im = plt.imshow(np.flipud(ims[i]))
        an_ims.append([im])

    ani = manimation.ArtistAnimation(fig, an_ims, interval=300, blit=True)
    ani.save(mname)
        

def stack_stf(lis_stf, ref_lis='/Users/service/Distortion_solution/starfinder/april/Nref_leg69.txt', plot=True):
    '''
    Parameters
    -----------
    lis_stf: string
        filename of text file that contains names of stf*.lis files.  These files will be stacked
    ref_lis: string
        name of the catalog to use as the reference for matching
    plot: Bool
        if True, creates and saves plots of the errors in the stacking.
    '''

    lis_tab = Table.read(lis_stf, format='ascii.no_header')
    ref_tab = Table.read(ref_lis, format='ascii.fixed_width')

    #use stars from reference, but only those with at least 10 detections
    nrefbool = ref_tab['N'] > 10
    xref = ref_tab['Xarc'][nrefbool]
    yref = ref_tab['Yarc'][nrefbool]
    mref = ref_tab['Mag'][nrefbool]

    xall, yall, mall = create_ref_from_lis(xref, yref, mref, lis_str=lis_tab['col1'], dr_tol=.01)

    #now create averages not includeing the reference for stars with at least 20 measurements
    Ninv = np.sum(xall[:,1:].mask, axis=1)
    N = xall[:,1:].shape[1] - Ninv
    nbool = N > 20

    xavg = np.mean(xall[:,1:],axis=1)
    xerr = np.std(xall[:,1:], axis=1)
    
    yavg = np.mean(yall[:,1:],axis=1)
    yerr = np.std(yall[:,1:], axis=1)
    
    mavg = np.mean(mall[:,1:],axis=1)
    merr = np.std(mall[:,1:], axis=1)

    xeom = xerr / np.sqrt(N-1)
    yeom = yerr / np.sqrt(N-1)

    print 'Number of stars with at least 20 detections:', np.sum(nbool)
    print 'Mean Error of the Mean X (N>20, arcseconds):', np.mean(xeom[nbool])
    print 'Mean standard deviation in stack X (N>20, arcseconds):',np.mean(xerr[nbool])
    print 'Mean standard deviation in stack X (N>20, brighter than 11, arcseconds):',np.mean(xerr[nbool*(mavg<11)])
    print 'Mean Error of the Mean Y (N>20, arcseconds):', np.mean(yeom[nbool])
    print 'Mean standard deviation in stack Y (N>20, arcseconds):',np.mean(yerr[nbool])
    print 'Mean standard deviation in stack Y (N>20, brighter than 11, arcseconds):',np.mean(yerr[nbool*(mavg<11)])
    
    #make plots of EOM
    plt.figure(1)
    plt.clf()
    plt.semilogy(mavg[nbool], xeom[nbool],'o', label='x')
    plt.semilogy(mavg[nbool], yeom[nbool],'o', label='y')
    
    plt.title('Error of the Mean')
    plt.xlabel('Magnitude')
    plt.ylabel('Error (arcseconds)')
    plt.axvline(11)
    plt.text(8,.0002,'Error X:'+str(np.mean(xeom[nbool*(mavg<11)])*10**3)[:4]+'mas')
    plt.text(8,.0003,'Error Y:'+str(np.mean(yeom[nbool*(mavg<11)])*10**3)[:4]+'mas')
    plt.legend(loc='upper left')
    
    return xall, yall, mall

def align(xrefin, yrefin, mrefin, lis_f='lis.lis', trans=high_order.four_paramNW, dr_tol=.02, lis_str=None, trans_lis=None, req_match=5, params_lis=[], weights=None):
    """
    Parameters
    ----------
    xrefin : numpy array
        array of x positions in the reference epoch
    yrefin : numpy array
        array of x positions in the reference epoch
    mrefin : numpy array
        array of x positions in the reference epoch
    lis_f : string
        one column text file containing a list of filenames of the starfinder starlists
    dr_tol: float
        maximum radial distance when looking for matches
    trans_model : transformation model object (class)
        The transformation model class that will be instantiated to find the best-fit
        transformation parameters between each list and the reference list.
    lis_str : list (optional)
        list of file names
    trans_lis : list of transformation model objects (optional)
        list of transformation model objects (these should have already been
        instantiated and have transformation parameters defined).
    param_lis: (optional) list of arrays with same shape as catalogs being matched
         List of parameters to be matched along with the psotions and magnitudes.
         form is [[Name_cat_1, Name_cat_2,...],[param_2_cat_1, param_2_cat_2, ....]]
         Here Name_cat_1 must be arrays, or at least carry a Numpy data type
    
    req_match: integer number of required matches of the input catalog to the total reference

    See Also
    --------
    high_order
    
    """

    if lis_str == None:
        lis_tab = Table.read(lis_f, format='ascii.no_header')
        lis_str =lis_tab['col1']
    

       
    #if we are on the first
    xrm = xrefin
    yrm = yrefin
    mrm = mrefin
    #need to make shape explicityly Nstars x 1
    xref = np.zeros((len(xrm), 1))
    yref = np.zeros((len(yrm), 1))
    mref = np.zeros((len(mrm), 1))
                        
    xref[:,0] = xrm[:]
    yref[:,0] = yrm[:]
    mref[:,0] = mrm[:]

    ind_mat_cats = []
    ind_mat_ref = []
  
    
    for i in range(len(lis_str)):
        cat = Table.read(lis_str[i], format='ascii.no_header')

        #first triangle match into reference frame
        #import pdb;pdb.set_trace()
        if trans_lis==None:
            N, x1m, y1m, m1m, x2m, y2m, m2m = jay.miracle_match_briteN(cat['col4'], cat['col5'], cat['col2'], xrefin, yrefin, mrefin, 50)
            assert len(x1m) > req_match

            t = trans(x1m, y1m ,x2m, y2m)

        else:
            t = trans_lis[i]
        #tranform catalog coordiantes, then match again

        xt,yt = t.evaluate(cat['col4'], cat['col5'])
        #now match again
        #import pdb;pdb.set_trace()
        idx1 , idx2 , dm, dr = align.match(xt,yt ,cat['col2'],  xrm.data, yrm.data, mrm.data, dr_tol)
        print 'Number of matches:',len(idx1)
        #assert len(idx1) > req_match

        # Update the total number of stars.
        #    Remember that xrm is modified in each loop.
        #    New stars are added to xrm as they are found in each successive list.
        #    idx1 - the matches between xrm and the current list.
        # Nnew = len(xrm) + all new sources not yet in the reference list
        Nnew = len(xrm) + len(xt) - len(idx1)

        # Initiale to a junk value (will be used for masking later).
        xrefn = np.zeros((Nnew, i+2)) - 100000
        yrefn = np.zeros((Nnew, i+2)) - 100000
        mrefn = np.zeros((Nnew, i+2)) - 100000

        # The index of the first new star to be added.
        in_mid = len(xrm)
        
        # Keep all of the old data (already in the reference list).
        # This is a 2D array with the first column is the reference epoch measurement.
        # All subsequent columns (up to in_mid) are the measurements in the lists
        # we have already processed.
        xrefn[:in_mid,:-1] = xref[:]
        yrefn[:in_mid,:-1] = yref[:]
        mrefn[:in_mid,:-1] = mref[:]

        # Put in the new matched stars for this list.
        xrefn[:in_mid,-1][idx2] = xt[idx1]
        yrefn[:in_mid,-1][idx2] = yt[idx1]
        mrefn[:in_mid,-1][idx2] = cat['col2'][idx1]

        # Identify everything that are in the current list; but don't exist in the reference.
        # Remember this reference is changing every iteration. So this effectively means
        # we haven't found this star in the reference or in any previous list. Totally new star.
        cbool = np.ones(len(xt), dtype='bool')
        cbool[idx1] = False


        # Now need to add positions of previously unidentified stars. Add this to the lower
        # right-hand section of the array.
        
        xrefn[in_mid:,-1] = xt[cbool]
        yrefn[in_mid:,-1] = yt[cbool]
        mrefn[in_mid:,-1] = cat['col2'][cbool]

        # Finally create masked arrays then calulate new means for next round of alignment
        mask = xrefn < -99999
        xrefn = np.ma.masked_less(xrefn, -99999)
        yrefn = np.ma.masked_less(yrefn, -99999)
        mrefn = np.ma.masked_less(mrefn, -99999)

        # Calculate the mean. Note we know this inappropriately includes the reference epoch.
        # But we need this to preserve stars only found in the reference epoch for now.
        # This will be re-calculated before we return anything.
        xrm = np.mean(xrefn, axis=1)
        yrm = np.mean(yrefn, axis=1)
        mrm = np.mean(mrefn, axis=1)

        
        
        # Update xref, yref, mref (our big 2D arrays) with the new list added and new stars added.
        xref = xrefn[:]
        yref = yrefn[:]
        mref = mrefn[:]

        #save the matched indices for creating other matched arrays at the end
        #first grab the indexes of the matched stars
        ind_mat_cats.append(idx1.tolist())
        ind_mat_ref.append(idx2.tolist())
        #now grab everything else that wasn't matched
        for ii in range(len(cbool)):
            if cbool[ii]:
                ind_mat_cats[-1].append(ii)
        #now finish the indexes for the reference array
        for ii in range(np.sum(cbool)):
            ind_mat_ref[-1].append(ii+in_mid-1)
  
                
                
                

    xrefout = xref[1:,:]
    yrefout = yref[1:,:]
    mrefout = mref[1:,:]

    #return large 2d array of each quantity
    #Note that the reference measuremtns are still included (first column)
    #QUESTION: Do I return the index arrays or 2-d arrays????
    #for now, try to create 2-d arrays
    #also note, data types must be supported by numpy arrays, or this will fail.
    if params_lis != []:
        out_param_lis = []
        #loop through parameters
        for i in range(len(params_lis)):
            #create array for the parameters with the same shape as the positions
            #make it the correct type for the given parameter
            #intitiating it as zeros, but the data will be masked
            tmp = np.zeros(xrefout.shape, dtype=params_lis[i][0].dtype)
            
            tmp_arr = np.ma.array(tmp, mask=xrefout.mask)
            #loop through the catalogs, and put the given parameter data into the matched location in the array
            for catnum in range(len(params_lis[i])):
                tmp_arr[ind_mat_ref[catnum], catnum] = params_lis[i][catnum][ind_mat_cats[catnum]]
            out_param_lis.append(tmp_arr)
    #out_param_lis is a list of the given parameters (e.g. errors or star names)
    #it is matched to the 2-d arrays of positions


    #return matched 2-d arrays.  These DO NOT include the reference.
    return xref, yref, mref, out_param_lis

            

def test_param_with_mat():
    lis_f = 'lis.lis'
    lis_tab = Table.read(lis_f, format='ascii.no_header')
    reference = '../starfinder/april/Nref_leg69.txt'

    ref_tab = Table.read(reference, format='ascii.fixed_width')
    xin = ref_tab['Xarc'][ref_tab['N'] > 10]
    yin = ref_tab['Yarc'][ref_tab['N'] > 10]
    magin = ref_tab['Mag'][ref_tab['N'] > 10]

    #go ahead and create this ugly parameter list
    param_lis = [[],[]]
    for i in range(len(lis_tab['col1'])):
        #read in the catalog
        tab_cat = Table.read(lis_tab['col1'][i], format='ascii.no_header')
        param_lis[0].append(tab_cat['col1'])
        param_lis[1].append(tab_cat['col7'])

    xref, yref, mref, out_param = create_ref_from_lis(xin, yin, magin, lis_f='lis.lis', params_lis=param_lis, weights=None)
    import pdb;pdb.set_trace()
    
        
        
            
            
def find_trans_from_match_by_name(ref_tab):
    '''
    abuse the fact that I found a mtach once to match by name then calualte a tranfomration
    Then grab the maximum N from tat match*.fits file
    then, I can use those as initial guesses to align the frames, and find MORE MATCHES!!!!!!!!!!!!!!!
    '''

    lis_all = Table.read('first_fits_m.lis', format='ascii.no_header')
    lis_mat = lis_all['col2']
    dar_lis = lis_all['col1']
    idlist = []
    translis = []

    for i in range(len(lis_mat)):
        tabm = Table.read(lis_mat[i], format='fits')
        idstf, idhst = match_trim.match_by_name( tabm['col0'], ref_tab['Name'], ret_both=True)


        #create the tranfoarmtion objext
        t = high_order.four_paramNW(tabm['col1'][idstf], tabm['col3'][idstf], ref_tab['Xarc'][idhst], ref_tab['Yarc'][idhst])
        #now create the new name
        idstr = dar_lis[i].split('.fits')[0][-4:]
        idnum = float(idstr)

    
        for kk in range(np.max(tabm['col6'])):
            tnum = idnum+kk
            pre_str = 'c'
            pos_str = '_0.8_stf.lis'
            strout = pre_str + str(int(tnum)).zfill(4) + pos_str
            #import pdb; pdb.set_trace()
            print strout
            idlist.append(strout)
            translis.append(t)


    return idlist, translis
            
        
            
            
        
        
