import glob,os,sys,numpy as np, matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from scipy import signal
from scipy.optimize import differential_evolution
import warnings
import astropy.io.fits as fits
import pandas as pd
import cartopy.crs as ccrs
from matplotlib import gridspec
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D


#####################################################################
#cutout_dict = {'20190607_165948':[375 ,445,1050,1140],'20190607_165013':[375 ,445,1140,1230],
#               '20190607_164006':[260 ,330,1050,1140],'20190607_163035':[260 ,330,1150,1240],
#               '20190607_174741':[380 ,450,540,630], '20190607_172832':[380 ,450,65,155], 
#               '20190607_171800':[265,335,165,255], '20190607_170915':[265,335,65,155], '20190607_161033':[430,500,1340,1430], '20190607_155108':[310,380,1810,1900]}

#375 ,445,1050,1140 #for 20190607_165948 #for  #375 ,445,1140,1230 for 20190607_165013 #260 ,330,1050,1140 for 20190607_164006 #260 ,330,1150,1240 for 20190607_163035 #380 ,450,540,630 for 20190607_174741 #380 ,450,65,155 for 20190607_172832 #for 172832 #265,335,165,255  for 171800 #265,335,65,155   for 170915

# scatype_dict = {{'20190607_165948':'B12','20190607_165013':'B12',
#               '20190607_164006':'B10','20190607_163035':'B10',
#               '20190607_174741':'A12', '20190607_172832':'A12', 
#               '20190607_171800':'A10', '20190607_170915':'A10', '20190607_161033':'C10', '20190607_155108':'C12'}}
######################################################################

def get_sca_box(fits_structure, ext, htyps,timecode, cutout_dict,scatype_dict, plot=False):

    g = fits_structure[ext].data
    typ = htyps[ext]
    tag = ''
    sca_ch = scatype_dict[timecode]
    
    if 'cross' in typ:
        tag = 'cross'
        strow,enrow, stcol,encol = cutout_dict[timecode][0:4]
        cut = g[strow:enrow, stcol:encol]
        boxsizex,boxsizey = encol-stcol, enrow-strow
        xm,ym=int((encol-stcol)/2),int((enrow-strow)/2)
        cy,cx = ym,xm

    elif 'along' in typ: 
        tag = 'along'
        strow,enrow, stcol,encol = cutout_dict[timecode][0:4]
        cut = g[strow:enrow, stcol:encol]
        xm,ym=int((encol-stcol)/2),int((enrow-strow)/2)
        cy,cx = ym,xm
    
    if plot: plt.imshow(cut, origin='lower')
    return tag,sca_ch,cx,cy,cut

###########################################################
def bgsubtract(openned_fits, htyps, raw_image_cutout,timecode,puck_extension, starting_extension, ending_extension,cutout_dict, scatype_dict, plot= False, verbose= False):
    
    # To Do: add a checkpoint to check the sizes of bg and raw image cutouts
    
    bg_cross = 0
    bg_along = 0
    f = openned_fits 
    cut = raw_image_cutout
    avgdark = np.zeros(np.shape(cut))#np.zeros((50,50))
    cnt = 0
    # stext,enext = 1,10
    stext,enext = starting_extension, ending_extension
    
    for n in range(stext,enext+1):  # the +1 is so the last extension is also included
        ext = n
        tag,sca_ch,cx,cy,bgcut = get_sca_box(f,ext,htyps, timecode, cutout_dict, scatype_dict)
        linear_bgcutout = linearize_sca(bgcut,timecode,puck_extension,sca_ch)

        avgdark += linear_bgcutout
        cnt += 1
        if verbose: 
            print(tag,n,bgcut.shape,np.average(avgdark))

    avgdark=avgdark/cnt

    if 'cross' in tag:
        bg_cross = avgdark
    elif 'along' in tag:
        bg_along = avgdark

    if verbose: print(tag+'-track master Background average:',np.average(avgdark))
    
    bgsubcut = raw_image_cutout - avgdark
    if plot:
        plt.imshow(bgsubcut, origin='lower')

    return avgdark, bgsubcut

###########################################################
def flatfield(openned_fits, htyps, background_subtracted_imagecutout, avgdark,timecode,puck_extension, startingflat_extension, endingflat_extension,cutout_dict,scatype_dict, plot=False, verbose=False):
    
    f = openned_fits 
    cut = background_subtracted_imagecutout
    flat_cross = 0
    flat_along = 0

    avgflat = np.zeros(np.shape(cut))
    cnt = 0

    stext,enext = startingflat_extension, endingflat_extension#10,19  # extensions for flat fields [square target]

    for n in range(stext,enext+1): #the '+1' is so the last extension is also included
        ext = n
        tag,sca_ch,cx,cy,ftcut = get_sca_box(f,ext,htyps, timecode, cutout_dict, scatype_dict)
        linear_scacutout = linearize_sca(ftcut,timecode,puck_extension,sca_ch)

        avgflat += linear_scacutout-avgdark ##### ---> subtracting the average background from the flat frame
        cnt += 1
        if verbose: print(tag,n,ftcut.shape,np.average(avgflat))

    avgflat = avgflat/cnt

    if 'cross' in tag:
        flat_cross = avgflat
    elif 'along' in tag:
        flat_along = avgflat

    if verbose: print(tag+'-track Average Background-subtracted master flat:',np.average(avgflat))
    flatcutout =  background_subtracted_imagecutout/avgflat

    if plot: 
        plt.imshow(flatcutout, origin='lower')

    return avgflat, flatcutout
###########################################################

def getmaxima(flatfielded_cutoutimage, plot=False):
    
    flatbgcutout= flatfielded_cutoutimage
    histrows = np.sum(flatbgcutout,1)
    maxrow = np.where(histrows == np.max(histrows))[0]
    histcols = np.sum(flatbgcutout,0)
    maxcol = np.where(histcols == np.max(histcols))[0]
    maxima_cross, maxima_along = maxcol[0], maxrow[0]
    if plot:
        ccut = flatbgcutout 
        ax1 = plt.subplot2grid((3, 3), (0, 2))
        ax1.axis('off')
        # plot some example image
        ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
        ax2.imshow(ccut, origin='lower')
        ax2.axvline(maxcol,ls='--',color='r')
        ax2.axhline(maxrow,ls='--',color='orange')
        # ax2.axis('off')
        # use 'barh' for horizontal bar plot
        horzSum = np.sum(ccut,1)
        ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2, sharey=ax2)
        ax3.barh(range(len(horzSum)), horzSum,color = 'navy')
        ax3.axhline(maxrow,ls='--',color='orange')
        # ax3.axis('off')
        vertSum = np.sum(ccut, 0)
        ax4 = plt.subplot2grid((3, 3), (0, 0), colspan=2, sharex=ax2)
        ax4.bar(range(len(vertSum)), vertSum,)
        ax4.axvline(maxcol,ls='--',color='r')
        # ax4.axis('off')
        #print('maximum count of the row conts in the cutout is in row & column:',maxrow[0], maxcol[0])
        plt.subplots_adjust(wspace=0, hspace=0.2)
        plt.show()

    return maxima_cross, maxima_along
###########################################################
def histimshow(inputarr):
        ccut = inputarr 
        ax1 = plt.subplot2grid((3, 3), (0, 2))
        ax1.axis('off')
        # plot some example image
        ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
        ax2.imshow(ccut, origin='lower')
        #ax2.axvline(maxcol,ls='--',color='r')
        #ax2.axhline(maxrow,ls='--',color='orange')
        # ax2.axis('off')
        # use 'barh' for horizontal bar plot
        horzSum = np.sum(ccut,1)
        ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2, sharey=ax2)
        ax3.barh(range(len(horzSum)), horzSum,color = 'navy')
        #ax3.axhline(maxrow,ls='--',color='orange')
        # ax3.axis('off')
        vertSum = np.sum(ccut, 0)
        ax4 = plt.subplot2grid((3, 3), (0, 0), colspan=2, sharex=ax2)
        ax4.bar(range(len(vertSum)), vertSum,)
        #ax4.axvline(maxcol,ls='--',color='r')
        # ax4.axis('off')
        #print('maximum count of the row conts in the cutout is in row & column:',maxrow[0], maxcol[0])
        plt.subplots_adjust(wspace=0, hspace=0.2)
        plt.show()#
	#plt.close()
##########################################################
def linearize_sca(raw_scacutout,timecode,puck_extension,sca_ch, plot=False):

    from scipy import interpolate
    path2file = '/home/sarah/Desktop/TIRS_Prelaunch/'
    cnt = 1
    if 'C' not in sca_ch:   
    
	    if 'A' in sca_ch:
	        #print('Linearization for SCA A is applied')
	        scalinear_datafile =  'tirs2_tvac1_HotOp_linearization_fpeA_vos1p1.csv'
	    elif 'B' in sca_ch: 
	    	#print('Linearization for SCA B is applied')
	    	scalinear_datafile = 'tirs2_tvac1_HotOp_linearization_fpeB_vos1p1.csv'
	    
	    df = pd.read_csv(path2file+scalinear_datafile)
	    func_nonlinear = interpolate.interp1d(df.A.values, df.B.values, kind='linear')
	    linear_scacutout = func_nonlinear(raw_scacutout)
	    
    elif 'C' in sca_ch:
    	print('No linearization is applied')
    	linear_scacutout = raw_scacutout
        
    if plot: 
        fig, ((ax1,ax2)) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.hist(raw_scacutout.reshape(-1),histtype='step', linewidth=1, facecolor='c',  hatch='/', edgecolor='red',fill=False,label='Raw DNs '+timecode)
        ax1.hist(linear_scacutout.reshape(-1),histtype='step', linewidth=1, facecolor='c', edgecolor='blue',fill=True,alpha=0.5,label='Linearized DNs'+'-ext='+str(puck_extension))
        ax2.plot(df["A"], df["B"])
        ax2.set_xlabel('Raw DN')
        ax2.set_ylabel('Linealized DN')
        ax1.legend()
        ax1.set_xlabel('DN')
        ax1.set_ylabel('Pixel #')
        ax1.set_yscale('log')
        plt.savefig('sca_linearized_'+timecode+'_'+sca_ch+'_ext'+str(puck_extension)+'.png')
        
    return linear_scacutout

############################################################

def get_sca_readycutout(openned_fits, htyps,timecode, puck_extension, dark_stex, dark_enext, flat_stext, flat_enext,cutout_dict,scatype_dict, plot=False, verbose=False):
    
    tag,sca_ch,cx,cy,raw_scacutout = get_sca_box(openned_fits,puck_extension,htyps,timecode, cutout_dict, scatype_dict)
    linear_scacutout = linearize_sca(raw_scacutout,timecode,puck_extension,sca_ch)
    avgdark, bgsubcut = bgsubtract(openned_fits, htyps, raw_scacutout,timecode,puck_extension, dark_stex, dark_enext,cutout_dict,scatype_dict, plot=plot)
    
    avgflat, flatbgcutout = flatfield(openned_fits, htyps, bgsubcut, avgdark,timecode,puck_extension, flat_stext, flat_enext,cutout_dict,scatype_dict)
    (openned_fits, htyps, bgsubcut, avgdark,timecode,puck_extension, flat_stext, flat_enext,cutout_dict,scatype_dict)
    maxima_cross, maxima_along = getmaxima(flatbgcutout)
    
    if verbose: print('background corrected, flat fielded cutout from extension {} is ready!'.format(puck_extension))
    
    
    return maxima_along, maxima_cross, flatbgcutout
###########################################################

#-------------------------------------------------------------------

def lfunc4pars(x_vals, a, b, c, d):
    return d + a/(1+np.exp((x_vals-b)/c))
#-------------------------------------------------------------------
def sumOfSquaredError_lab4p_func(test_X, test_Y):
    def sumOfSquaredError_lab4p(parameterTuple):
            warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
            val = lfunc4pars(test_X, *parameterTuple)
            return np.sum((test_Y - val) ** 2.0)
        
    return sumOfSquaredError_lab4p
#-------------------------------------------------------------------    
def sumOfSquaredError_lab3p_func(test_X, test_Y):
    def sumOfSquaredError_lab3p(parameterTuple):
            warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
            val = lfunc3pars(test_X, *parameterTuple)
            return np.sum((test_Y - val) ** 2.0)
        
    return sumOfSquaredError_lab3p
#-------------------------------------------------------------------
def generate_Initial_lab4Parameters(test_X, test_Y):
        # min and max used for bounds
        maxX = max(test_X)
        minX = min(test_X)
        maxY = max(test_Y)
        minY = min(test_Y)
        maxXY = max(maxX, maxY)

        parameterBounds = []
        parameterBounds.append([10000, 30000]) # seach bounds for a= b
        parameterBounds.append([10, 30]) # seach bounds for b = e
        parameterBounds.append([0.0001, 10]) # c = s
        parameterBounds.append([10000, 30000]) # seach bounds for d
        
        sumOfSquaredError_lab4p = sumOfSquaredError_lab4p_func(test_X, test_Y)

        # "seed" the numpy random number generator for repeatable results
        result = differential_evolution(sumOfSquaredError_lab4p, parameterBounds, seed=3)
        return result.x

#-------------------------------------------------------------------
### Same formula used by Wenny et al 2015 and in the "Prelaunch lab Narrative" doc for TIR8 or slide set of Helder and Choi 2003 
def lfunc4pars(x_vals, a, b, c, d):
    return d + a/(1+np.exp((x_vals-b)/c))
#-------------------------------------------------------------------

def lfunc3pars(x_vals, a, c, d):
    return d + a/(1+np.exp((x_vals)/c))
#-------------------------------------------------------------------

def gauss_func(x_vals, ampl, sig, mu):
    return ampl * np.exp(-(-x_vals-mu)**2/(2*sig**2))
#-------------------------------------------------------------------
def sumOfSquaredError_lab3p(parameterTuple):
        warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
        val = lfunc3pars(test_X, *parameterTuple)
        return np.sum((test_Y - val) ** 2.0)
#-------------------------------------------------------------------

def sumOfSquaredError_lab4p(parameterTuple):
        warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
        val = lfunc4pars(test_X, *parameterTuple)
        return np.sum((test_Y - val) ** 2.0)
#-------------------------------------------------------------------
def generate_Initial_lab4Parameters(test_X, test_Y):
        # min and max used for bounds
        maxX = max(test_X)
        minX = min(test_X)
        maxY = max(test_Y)
        minY = min(test_Y)
        maxXY = max(maxX, maxY)

        parameterBounds = []
        parameterBounds.append([10000, 30000]) # seach bounds for a= b
        parameterBounds.append([10, 30]) # seach bounds for b = e
        parameterBounds.append([0.0001, 10]) # c = s
        parameterBounds.append([10000, 30000]) # seach bounds for d

        # "seed" the numpy random number generator for repeatable results
        result = differential_evolution(sumOfSquaredError_lab4p, parameterBounds, seed=3)
        return result.x
    
#-------------------------------------------------------------------
def generate_Initial_lab3Parameters(test_X, test_Y):
        # min and max used for bounds
        maxX = max(test_X)
        minX = min(test_X)
        maxY = max(test_Y)
        minY = min(test_Y)
        maxXY = max(maxX, maxY)

        parameterBounds = []
        parameterBounds.append([10000, 30000]) # seach bounds for a= b
        # parameterBounds.append([10, 30]) # seach bounds for b = e
        parameterBounds.append([0.0001, 10]) # c = s
        parameterBounds.append([10000, 30000]) # seach bounds for d
        sumOfSquaredError_lab3p = sumOfSquaredError_lab3p_func(test_X, test_Y)
        # "seed" the numpy random number generator for repeatable results
        result = differential_evolution(sumOfSquaredError_lab3p, parameterBounds, seed=3)
        return result.x

#-------------------------------------------------------------------
def generate_Initial_lab4Parameters(test_X, test_Y):
        # min and max used for bounds
        maxX = max(test_X)
        minX = min(test_X)
        maxY = max(test_Y)
        minY = min(test_Y)
        maxXY = max(maxX, maxY)

        parameterBounds = []
        parameterBounds.append([10000, 30000]) # seach bounds for a= b
        parameterBounds.append([10, 30]) # seach bounds for b = e
        parameterBounds.append([0.0001, 10]) # c = s
        parameterBounds.append([10000, 30000]) # seach bounds for d
        
        sumOfSquaredError_lab4p = sumOfSquaredError_lab4p_func(test_X, test_Y)

        # "seed" the numpy random number generator for repeatable results
        result = differential_evolution(sumOfSquaredError_lab4p, parameterBounds, seed=3)
        return result.x
    
#------------------------------------------------------------------
def pregausfit_cleanup(xpsfnorm,ypsfnorm):
    y_max = 0.5
    cond1 = False
    cond2 = False
    y_prev = -10000
    
    for k in range(len(xpsfnorm)):
        if ypsfnorm[k] > y_max:
            y_max = ypsfnorm[k]
            cond1 = True

        if cond1 and ypsfnorm[k]<y_prev:
            cond2 = True

        if cond2 and ypsfnorm[k]>y_prev:
            break
        y_prev = ypsfnorm[k]
        
    k = k - 1
    cxpsf, cypsf = xpsfnorm[:k], ypsfnorm[:k]
    return k,cxpsf, cypsf
 


#-------------------------------------------------------------------
from scipy.optimize import differential_evolution
import warnings
from scipy import optimize, interpolate, ndimage, stats
from scipy.optimize import OptimizeWarning

def getMTF(lsf,timecode, sca_ch, edgeside, plot = False):
    # NOTE: LSF input has to be 2D array with first column being the pixels and second column the LSF values

    summary = ""
    print("shape lsf", lsf.shape)
    
    # If needed, remove the last element of the PSF to get an even number of elements         

    if lsf.shape[1]/2. != lsf.shape[1]//2.:
        print('needed!')
        lsf = lsf[:,:-1]

    pixrange = np.max(lsf[0])-np.min(lsf[0])
        
    lsf = lsf[1]

    n = lsf.shape[0]
    lsf = np.append(
            np.append(
                np.zeros([20*n]), 
                lsf),
            np.zeros([20*n])
        )
    
    lsf = lsf/np.sum(lsf)
    
    N = len(lsf)
    print("Length lsf",N)
    
    mtf = np.fft.rfft(lsf)

    sampFreq = 1./30
    nyquistFreq = sampFreq/2.
    adjustedSampFreq = sampFreq*pixrange/N

    print('Imaging sample freq= ',sampFreq)
    print('LSF sampling rate= ',adjustedSampFreq)

    mtfFreq = np.linspace(0, (N/2)*adjustedSampFreq/nyquistFreq, num=mtf.shape[0], dtype=np.float64)
    mtfVsFreq = interpolate.interp1d(mtfFreq, np.absolute(mtf), kind='linear')
    freqVsMtf = interpolate.interp1d(np.absolute(mtf), mtfFreq, kind='linear')
    mtfNy = mtfVsFreq(0.5)
    # ResultsStr += "MTF0: %s \n" % mtf[0]
    # summary += "FWHM = %f "% np.round(gfwhm,4)
    # summary += "+/- %f \n" % np.round(gfwhmerr,4)
    # summary += "\n"
    # summary += "Edge Slope = %f "% np.round(gs,4)
    # summary += "+/- %f \n" % np.round(sloperr,4)
    # summary += "\n"
    # summary += "30percent MTF = %f \n" % freqVsMtf(0.3)
    # summary += "\n"
    # summary += "50percent MTF = %f \n" % freqVsMtf(0.5)
    # summary += "\n"
    summary += "MTF@f_Ny =  %f \n" % mtfVsFreq(0.5)
    print("\n############ MTF Results\n"+summary+"####################\n")
    
    
    if plot:
        fig,(ax4) = plt.subplots(1, 1, figsize=(6, 4))

        ax4.plot(mtfFreq[mtfFreq<0.6], mtfVsFreq(mtfFreq[mtfFreq<0.6]))
        ax4.set_xlabel('Normalized Frequencies (1/m)')
        ax4.set_ylabel('Modulation Transfer Function')
        ax4.set_xlim(0,0.62)
        ax4.set_ylim(0.5,1.05)
        ax4.grid(ls = '--')
        ax4.minorticks_on()
        ax4.axhline(y= mtfNy, xmin=0., xmax=1.0, linewidth=1, ls='--' ,color='red')
        ax4.axhline(y= 0.1, xmin=0., xmax=1.0, linewidth=1, ls='-' ,color='green')
        plt.savefig('MTF_'+timecode+'_'+sca_ch+'_'+edgeside+'.png')

    return mtf, mtfFreq, mtfVsFreq, summary, nyquistFreq, mtfNy
    
#-------------------------------------------------------------------

def LinearX(x, x1, y1, x2, y2):
    return (x - x1) * (y2 - y1) / (x2 - x1) + y1

#-------------------------------------------------------------------

def find_root(X, Y, y0):
    """
    X and Y are lists, where Y=F(X)
    returns:
        list of roots: X_root
    y0 is the value for which X_root[..] = F(y0)
    """
    
    X_root = []
    for j in range(1, len(Y)):
        if (y0 >= Y[j - 1] and y0 < Y[j]) or (y0 <= Y[j - 1] and y0 > Y[j]):
            X_root.append(LinearX(y0, Y[j - 1], X[j - 1], Y[j], X[j]))
    
    return X_root

#-------------------------------------------------------------------

def slope_finder(X, Y):
    r,l = find_root(X, Y, 0.6)[0],find_root(X, Y, 0.4)[0]
    s = 0.2/(r-l)
    print('--------------------------------------')
    print('Edge Slope (100m pixel size) = {} 1/100m'.format(s))
    print('--------------------------------------')
