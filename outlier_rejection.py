from jlu.util import statsIter
import numpy as np

def sig_trim(x,xerr,y,yerr,xref,xreferr,yref,yreferr,names,mag, sig_fac=3 , num_section=9):
    '''
    Performs spatial sigma trimming of the difference between catalog and reference
    sig_fac is the number of sigma clipped
    num_section is the number of sections that each axis is split into, for example num_section=9 means that a 9x9 grid is used
    other arguements are 1-d array-like

    returns trimmed matched arrays.  
    '''
    
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
    
    bins = np.linspace(0,1024, num=num_section + 1 )

    for i in range(len(bins)-1):
        for j in range(len(bins)-1):
            #first find all the stars that are in the current bin
            sbool = (x > bins[i])*(x<bins[i+1])*(y>bins[j])*(y<bins[j+1])
            print 'number of stars in this section are ', np.sum(sbool)
            #find the mean delta and sigma in x and y, using iterative sigma clipping
            ave_x, sig_x, nclipx = statsIter.mean_std_clip(dx[sbool], clipsig=3.0, return_nclip=True)
            ave_y, sig_y, nclipy = statsIter.mean_std_clip(dy[sbool], clipsig=3.0, return_nclip=True)
            #creates boolean 
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
                


    print 'number of stars cliiped ', len(x) - len(xout)
    return  np.array(xout),np.array(xeout) ,np.array(yout), np.array(yeout), np.array(xrout), np.array(xreout), np.array(yrout), np.array(yreout), np.array(nout) , np.array(mout)
