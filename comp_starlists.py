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


def go(plot=True):
    lis_stf = Table.read('lis.lis', format='ascii.no_header')
    lisd = Table.read('lis_dn.lis', format='ascii.no_header')
    darlis = Table.read('darn.lis', format='ascii.no_header')
    t = pickle.load(open('Nref_leg62.trans', 'r'))
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
        mk_movie(xl, yl, dxl, dyl, mname='diff_stf_dist_mean_sub.mp4')
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
        



def create_ref_from_lis(xrefin, yrefin, mrefin, lis_f='lis.lis', trans=high_order.four_paramNW, dr_tol=.02, lis_str=None, trans_lis=None):

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


    
    for i in range(len(lis_str)):
        cat = Table.read(lis_str[i], format='ascii.no_header')

        #first triangle match into reference frame
        #import pdb;pdb.set_trace()
        if trans_lis==None:
            N, x1m, y1m, m1m, x2m, y2m, m2m = jay.miracle_match_briteN(cat['col4'], cat['col5'], cat['col2'], xrefin, yrefin, mrefin, 50)

            t = trans(x1m, y2m ,x2m, y2m)

        else:
            t = trans_lis[i]
        #tranform catalog coordiantes, then match again

        xt,yt = t.evaluate(cat['col4'], cat['col5'])
        #now match again
        #import pdb;pdb.set_trace()
        idx1 , idx2 , dm, dr = align.match(xt,yt ,cat['col2'],  xrm.data, yrm.data, mrm.data, dr_tol)
        print len(idx1)

        #now create new reference
        xrefn = np.zeros((len(xrm)+len(xt)-len(idx1),i+2)) - 100000
        yrefn = np.zeros((len(xrm)+len(xt)-len(idx1),i+2)) - 100000
        mrefn = np.zeros((len(xrm)+len(xt)-len(idx1),i+2)) - 100000

        in_mid = len(xrm)
        #keep all of the old data
        xrefn[:in_mid,:-1] = xref[:]
        yrefn[:in_mid,:-1] = yref[:]
        mrefn[:in_mid,:-1] = mref[:]

        #put in the new matched stars
        xrefn[:in_mid,-1][idx2] = xt[idx1]
        yrefn[:in_mid,-1][idx2] = yt[idx1]
        mrefn[:in_mid,-1][idx2] = cat['col2'][idx1]
        cbool = np.ones(len(xt), dtype='bool')
        cbool[idx1] = False

        #now need to add positions of previously unmatched stars
        instart = len(xrm)
        xrefn[in_mid:,-1] = xt[cbool]
        yrefn[in_mid:,-1] = yt[cbool]
        mrefn[in_mid:,-1] = cat['col2'][cbool]

        #finally create masked arrays then calulate new means for next round of alignment
        mask = xrefn < -99999
        xrefn = np.ma.masked_less(xrefn, -99999)
        yrefn = np.ma.masked_less(yrefn, -99999)
        mrefn = np.ma.masked_less(mrefn, -99999)

        
        xrm = np.mean(xrefn, axis=1)
        yrm = np.mean(yrefn, axis=1)
        mrm = np.mean(mrefn, axis=1)

        #import pdb; pdb.set_trace()
        xref = xrefn[:]
        yref = yrefn[:]
        mref = mrefn[:]



    return xrefn, yrefn, mrefn, np.mean(xrefn[:,1:],axis=1) , np.mean(yrefn[:,1:],axis=1), np.mean(mrefn[:,1:],axis=1), np.std(xrefn[:,1:], axis=1), np.std(yrefn[:,1:], axis=1), np.std(mrefn[:,1:], axis=1), np.sum(xrefn.mask, axis=1)

            
            
            
            
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
            
        
            
            
        
        
