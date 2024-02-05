import os
import sys
import copy
import numpy as np
from scipy import interpolate as ip
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

iSpecFol = "/home/max/CallofPhDuty/iSpec_v20230804/"
sys.path.append(iSpecFol)
import ispec

#--- Change LOG level -----------------------------------------------------
import logging
#LOG_LEVEL = "warning"
LOG_LEVEL = "info"
logger = logging.getLogger() # root logger, common for all
logger.setLevel(logging.getLevelName(LOG_LEVEL.upper()))

def O_ReadData():
    objName = sys.argv[1]; testName = ''
    if 'SZMon' in objName:
        testName = copy.deepcopy(objName); objName = 'J065127'
    elif 'DFCyg' in objName:
        testName = copy.deepcopy(objName); objName = 'J194853'
    visit = O_DefineVisit(objName); code = 'all' # 'self' / 'etom' / 'isot' / 'all'
    species,colNames,dt,SpWnNm,skiprows = O_Initial(objName, testName, sigRead=1, delRead=1., code=code)
    SpecWinds = O_ReadWinds(SpWnNm, dt, skiprows, code=code); SpecWinds=np.array([np.array(s) for s in SpecWinds])
    spec = np.loadtxt(iSpecFol+"mySample/outputAPOGEE/%7s/%7s_CleanSpec%s.txt" % (objName, objName, visit), delimiter='\t', unpack=True, skiprows=1)
    spec, SpecWinds = O_FilterMismatch(spec,SpecWinds) # Removes unmatched data points
    for i in range(1,len(SpecWinds)):
        SpecWinds[i] = O_FixWls(spec,SpecWinds[0],SpecWinds[i]) # Fixes wavelength scale of SpecWinds
    return(objName, testName, species, colNames, spec, SpecWinds)
    
def O_DefineVisit(objName):
    visit_dict = {'J065127': '_visit2', 'J194853': '_visit1'}
    return visit_dict.get(objName, '')

def O_Initial(objName, testName, sigRead=1, delRead=1., code='etom'):
    if code=='etom':
        species = ['He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ge', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cs', 'Ba', 'Ce', 'Nd', 'Yb'] #['13C', '15N', '17O', '18O'] #['C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'S', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Dy', 'Er', 'Yb', 'Lu', 'Hf', 'W', 'Pb']
    elif code=='isot':
        species = ['13C', '15N', '17O', '18O']
    elif code=='all':
        species = ['He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ge', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cs', 'Ba', 'Ce', 'Nd', 'Yb', '13C', '15N', '17O', '18O']
    #--- Initialising reading formats and files ---------------------------
    if code=='ASPCAP':
        colNames = {'Wave': 0, 'Fe': 1, 'C': 2, 'N': 3, 'O': 4, 'Na': 5, 'Mg': 6, 'Al': 7, 'Si': 8, 'S': 9, 'K': 10, 'Ca': 11, 'Ti': 12, 'V': 13, 'Mn': 14, 'Ni': 15}
        dt=np.dtype([('Wave',float), ('Fe',float), ('C',float), ('N',float), ('O',float), ('Na',float), ('Mg',float), ('Al',float), ('Si',float), ('S',float), ('K',float), ('Ca',float), ('Ti',float), ('V',float), ('Mn',float), ('Ni',float)])
        SpWnNm = "ASPCAPSpectralWindows.txt"
        skiprows=31
    elif code=='self':
        colNames = {'Wave': 0, 'C': 1, 'N': 2, 'O': 3, 'Na': 4, 'Mg': 5, 'Al': 6, 'Si': 7, 'S': 8, 'K': 9, 'Ca': 10, 'Sc': 11, 'Ti': 12, 'V': 13, 'Cr': 14, 'Mn': 15, 'Fe': 16, 'Co': 17, 'Ni': 18, 'Cu': 19, 'Zn': 20, 'Y': 21, 'Zr': 22, 'La': 23, 'Ce': 24, 'Pr': 25, 'Nd': 26, 'Sm': 27, 'Eu': 28, 'Gd': 29, 'Dy': 30, 'Er': 31, 'Yb': 32, 'Lu': 33, 'Hf': 34, 'W': 35, 'Pb': 36}
        dt=np.dtype([('Wave',float), ('C',float), ('N',float), ('O',float), ('Na',float), ('Mg',float), ('Al',float), ('Si',float), ('S',float), ('K',float), ('Ca',float), ('Sc',float), ('Ti',float), ('V',float), ('Cr',float), ('Mn',float), ('Fe',float), ('Co',float), ('Ni',float), ('Cu',float), ('Zn',float), ('Y',float), ('Zr',float), ('La',float), ('Ce',float), ('Pr',float), ('Nd',float), ('Sm',float), ('Eu',float), ('Gd',float), ('Dy',float), ('Er',float), ('Yb',float), ('Lu',float), ('Hf',float), ('W',float), ('Pb',float)])
        SpWnNm = iSpecFol+"mySample/WindowScripts/SpecWinds/SpecWinds_%s/%s_SpecWinds_s%1i_d%.1f.txt" % (objName, objName, sigRead, delRead)
        skiprows=31
    elif code=='etom':
        colNames = {'Wave': 0, 'He': 1, 'Li': 2, 'Be': 3, 'B': 4, 'C': 5, 'N': 6, 'O': 7, 'F': 8, 'Ne': 9, 'Na': 10, 'Mg': 11, 'Al': 12, 'Si': 13, 'P': 14, 'S': 15, 'Cl': 16, 'Ar': 17, 'K': 18, 'Ca': 19, 'Sc': 20, 'Ti': 21, 'V': 22, 'Cr': 23, 'Mn': 24, 'Fe': 25, 'Co': 26, 'Ni': 27, 'Cu': 28, 'Zn': 29, 'Ge': 30, 'Rb': 31, 'Sr': 32, 'Y': 33, 'Zr': 34, 'Nb': 35, 'Mo': 36, 'Tc': 37, 'Ru': 38, 'Rh': 39, 'Pd': 40, 'Ag': 41, 'Cs': 42, 'Ba': 43, 'Ce': 44, 'Nd': 45, 'Yb': 46}
        dt=np.dtype([('Wave',float), ('He',float), ('Li',float), ('Be',float), ('B',float), ('C',float), ('N',float), ('O',float), ('F',float), ('Ne',float), ('Na',float), ('Mg',float), ('Al',float), ('Si',float), ('P',float), ('S',float), ('Cl',float), ('Ar',float), ('K',float), ('Ca',float), ('Sc',float), ('Ti',float), ('V',float), ('Cr',float), ('Mn',float), ('Fe',float), ('Co',float), ('Ni',float), ('Cu',float), ('Zn',float), ('Ge',float), ('Rb',float), ('Sr',float), ('Y',float), ('Zr',float), ('Nb',float), ('Mo',float), ('Tc',float), ('Ru',float), ('Rh',float), ('Pd',float), ('Ag',float), ('Cs',float), ('Ba',float), ('Ce',float), ('Nd',float), ('Yb',float)])
        SpWnNm = iSpecFol+"mySample/WindowScripts/SpecWinds/SpecWinds_%s/%s_SpecWinds_s%1i_d%.1f.txt" % (objName, objName, sigRead, delRead)
        skiprows=0
    elif code=='isot':
        colNames = {'Wave': 0, '13C': 1, '15N': 2, '17O': 3, '18O': 4}
        dt=np.dtype([('Wave',float), ('13C',float), ('15N',float), ('17O',float), ('18O',float)])
        SpWnNm = iSpecFol+"mySample/WindowScripts/SpecWinds/SpecWinds_%s/%s_SpecWinds_iso.txt" % (objName, objName)
        skiprows=0
    elif code=='all':
        colNames = {'Wave': 0, 'He': 1, 'Li': 2, 'Be': 3, 'B': 4, 'C': 5, 'N': 6, 'O': 7, 'F': 8, 'Ne': 9, 'Na': 10, 'Mg': 11, 'Al': 12, 'Si': 13, 'P': 14, 'S': 15, 'Cl': 16, 'Ar': 17, 'K': 18, 'Ca': 19, 'Sc': 20, 'Ti': 21, 'V': 22, 'Cr': 23, 'Mn': 24, 'Fe': 25, 'Co': 26, 'Ni': 27, 'Cu': 28, 'Zn': 29, 'Ge': 30, 'Rb': 31, 'Sr': 32, 'Y': 33, 'Zr': 34, 'Nb': 35, 'Mo': 36, 'Tc': 37, 'Ru': 38, 'Rh': 39, 'Pd': 40, 'Ag': 41, 'Cs': 42, 'Ba': 43, 'Ce': 44, 'Nd': 45, 'Yb': 46, '13C': 47, '15N': 48, '17O': 49, '18O': 50}
        dt=np.dtype([('Wave',float), ('He',float), ('Li',float), ('Be',float), ('B',float), ('C',float), ('N',float), ('O',float), ('F',float), ('Ne',float), ('Na',float), ('Mg',float), ('Al',float), ('Si',float), ('P',float), ('S',float), ('Cl',float), ('Ar',float), ('K',float), ('Ca',float), ('Sc',float), ('Ti',float), ('V',float), ('Cr',float), ('Mn',float), ('Fe',float), ('Co',float), ('Ni',float), ('Cu',float), ('Zn',float), ('Ge',float), ('Rb',float), ('Sr',float), ('Y',float), ('Zr',float), ('Nb',float), ('Mo',float), ('Tc',float), ('Ru',float), ('Rh',float), ('Pd',float), ('Ag',float), ('Cs',float), ('Ba',float), ('Ce',float), ('Nd',float), ('Yb',float), ('13C',float), ('15N',float), ('17O',float), ('18O',float)])
        SpWnNm = iSpecFol+"mySample/WindowScripts/SpecWinds/SpecWinds_%s/%s_SpecWinds%s_s%1i_d%.1f.txt" % (objName, objName, testName, sigRead, delRead)
        skiprows=0
    if not os.path.exists(iSpecFol+"mySample/WindowScripts/Abunds/Abunds_%s%s/" % (objName, testName)):
        os.makedirs(iSpecFol+"mySample/WindowScripts/Abunds/Abunds_%s%s/" % (objName, testName))
    return(species,colNames,dt,SpWnNm,skiprows)

def O_ReadWinds(SpWnNm, dt, skiprows, code):
    SpecWinds = np.loadtxt(SpWnNm, dtype=dt, delimiter=' ', unpack=True, skiprows=skiprows)
    #--- Micron-to-nanometer conversion (+vacuum-to-air for ASPCAP) -------
    if code=='ASPCAP':
        SpecWinds[0] *= 10000.
        SpecWinds[0] /= (1. + 5.792105E-2/(238.0185 - (1.E4/SpecWinds[0])**2) + 1.67917E-3/(57.362 - (1.E4/SpecWinds[0])**2))
        SpecWinds[0] /= 10.
    elif code=='self' or code=='etom' or code=='isot' or code=='all':
        SpecWinds[0] *= 1000.
    for k in range(len(SpecWinds)):
        SpecWinds[k] = np.round(SpecWinds[k], 5)
    return(SpecWinds)

def O_FilterMismatch(spec, SpecWinds): # Checks whether the wavelength columns of spectra and
    for l,dp in enumerate(SpecWinds[0]): # spectral windows are similar
        if np.min(abs(spec[0]-dp))<0.01:
            continue
        else:
            SpecWinds[0][l] = np.nan
    mask2 = [~np.isnan(s) for s in SpecWinds[0]]
    SpecWinds = SpecWinds[:,mask2]
    for l,dp in enumerate(spec[0]):
        if np.min(abs(SpecWinds[0]-dp))<0.01:
            continue
        else:
            spec[0][l] = np.nan
    mask1 = [~np.isnan(s) for s in spec[0]]
    spec = spec[:,mask1]
    plt.scatter(spec[0], np.ones_like(spec[0]), s=20, c='r', marker=3)
    plt.scatter(SpecWinds[0], np.ones_like(SpecWinds[0]), s=5, c='b')
    plt.show()
    return(spec, SpecWinds)

def O_FixWls(spec,wlSpWn,flSpWn):
    #--- Bringing everything to the same wavelengths ----------------------
    f = ip.interp1d(wlSpWn,flSpWn,fill_value="extrapolate"); flSpWn = f(spec[0])
    return(flSpWn)
    
def O_ReadParams(objName):
    #--- Reading the stellar parameters from 'Spoiler.txt' ----------------
    Aname, Ateff, Alogg, Amet, Avmic, Acomm = np.loadtxt(iSpecFol+"Spoiler.txt", delimiter='\t', dtype=np.dtype([('name','U8'), ('teff',float), ('logg',float), ('met',float), ('vmic',float), ('comm','U5')]), skiprows=1, unpack=True)
    ind = np.where(Aname==objName)[0]
    teff = float(Ateff[ind]); logg = float(Alogg[ind])
    met = float(Amet[ind]); vmic = float(Avmic[ind])
    if objName=='J054057':
        vsini = 40.
    else:
        vsini = 0.
    return(teff, logg, met, vmic, vsini)
    
def O_FindPatch(window):
    # Create an array that is 1 where a is `value`, and pad each end with an extra 0.
    isvalue = np.concatenate(([0.], np.not_equal(window, 0.).view(np.int8), [0.]))
    absdiff = np.abs(np.diff(isvalue))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return(ranges)

def O_CalcChi2(fl0, fl, spw):
    chi = (fl[spw!=0.]-fl0[spw!=0.])/np.nanstd(fl[spw!=0.]); chi2 = np.sum(chi**2) #np.sum(spw[spw!=0.]/np.sum(spw)*dum**2) #*spw
    return(chi2)
    
def O_MetScalSolAbund(objName, species):
    teff, logg, met, vmic, vsini = O_ReadParams(objName)
    if len(species)==3:
        species = species[-1]
    solar_abundances_file = iSpecFol + "/input/abundances/Asplund.2009/stdatom.dat"
    solar_abundances = ispec.read_solar_abundances(solar_abundances_file)
    chemical_elements_file = iSpecFol + "/input/abundances/chemical_elements_symbols.dat"
    chemical_elements = ispec.read_chemical_elements(chemical_elements_file)
    ID = chemical_elements['atomic_num'][chemical_elements['symbol']==species]
    ab = solar_abundances['Abund'][solar_abundances['code']==ID]
    abCntr = ab+12.036+met
    return(abCntr)

def O_DisentangleBumps(SpWnCut, cwl):
    #--- Removing possible neighbor spectral windows in 10 nm range -------
    spw = SpWnCut[1]
    nonzeros = O_FindPatch(spw)
    for patch in nonzeros:
        if SpWnCut[0][patch[0]]<=cwl and SpWnCut[0][patch[1]-1]>=cwl:
            spw[:patch[0]] = 0.
            spw[patch[1]:] = 0.
            if len(spw[patch[0]:patch[1]])<3:
                spw[patch[0]-1:patch[1]+1] = 0.4
    return(spw)

def O_CutWindow(spectrum, SpecWinds, index, cwl, species):
    print('Cutting a region of %s at %.5f' % (species, cwl))
    #--- Cutting 10 nm range ----------------------------------------------
    maskCut = (spectrum[0]>cwl-5.) & (spectrum[0]<cwl+5.)
    specCut = np.array([spectrum[0][maskCut], spectrum[1][maskCut], spectrum[2][maskCut]])
    SpWnCut = np.array([SpecWinds[0][maskCut], SpecWinds[index][maskCut]])
    return(specCut, SpWnCut)

def O_Fib(n): # Calculates Fibonacci number by its position number (used in Fibonacci search below)
    F = ((1+np.sqrt(5))**n-(1-np.sqrt(5))**n)/(2**n*np.sqrt(5))
    return(int(np.round(F,0)))

def O_IsoRatioName(species):
    isotopes = {
        '13C': '$^{12}$C/$^{13}$C',
        '15N': '$^{14}$N/$^{15}$N',
        '17O': '$^{16}$O/$^{17}$O',
        '18O': '$^{16}$O/$^{18}$O',
    }
    return isotopes.get(species, None)
    
def O_IsoRatioRange(species):
    iso_ranges = {
        '13C': (2., 92.5),
        '15N': (2., 271.),
        '17O': (2., 2630.),
        '18O': (2., 487.),
    }
    return iso_ranges.get(species, (None, None))

def V_ElemOrder(objName, testName, species, spec, SpecWinds, colNames): # Defines order of chemical elements for abundance calculation
    updS = []; updA = []; updD = []
    teff, logg, met, vmic, vsini = O_ReadParams(objName)
    #--- Fe ---------------------------------------------------------------
    newFe, dFe = V_CheckSpecWinds(objName, testName, 'Fe', spec, SpecWinds, colNames, speciesType='elem', updS=updS, updA=updA) #[7.5+met, -1.]
    updS.append('Fe'); updA.append(newFe); updD.append(dFe)
    print('Results are: A(Fe) = %.2f+/-%.2f' % (newFe, dFe))
    #--- CNO --------------------------------------------------------------
    updS, updA, updD, order = V_ElemOrder_NOiter(objName, testName, updS, updA, updD, met) # V_ElemOrder_CNOiter(objName, testName, updS, updA, updD, met)
    print('Results are: A(C)=%.2f+/-%.2f, A(N)=%.2f+/-%.2f, A(O)=%.2f+/-%.2f\nCalculation order of species: %s' % (updA[updS.index('C')], updD[updS.index('C')], updA[updS.index('N')], updD[updS.index('N')], updA[updS.index('O')], updD[updS.index('O')], str(order)))
    #--- Everything else --------------------------------------------------
    species = np.array(species)
    mask = (species!='C') & (species!='N') & (species!='O') & (species!='Fe')
    remains = species[mask].tolist()
    print('Now analysing these species: ', remains)
    for species in remains:
        if len(species)!=3:
            #--- Other elements (Z<6 and Z>8) -----------------------------
            newA, sA = V_CheckSpecWinds(objName, testName, species, spec, SpecWinds, colNames, speciesType='elem', updS=updS, updA=updA)
            updS.append(species); updA.append(newA); updD.append(sA)
        else:
            #--- Isotopes -------------------------------------------------
            newA, sA = V_CheckSpecWinds(objName, testName, species, spec, SpecWinds, colNames, speciesType='isot', updS=updS, updA=updA)
            updS.append(species); updA.append(newA); updD.append(sA)
    print(objName+' overall results are: ', [(updS[i], np.round(updA[i],2)) for i in range(len(updS))])
    return(updS, updA)

def V_ElemOrder_CNOiter(objName, testName, updS, updA, updD, met): # In case there are NOT any CNO atomic lines
    precisC = 0.01; oldC = 0.; newC = 8.43+met
    precisN = 0.01; oldN = 0.; newN = 7.83+met
    precisO = 0.01; oldO = 0.; newO = 8.69+met
    iterC=0; iterN=0; iterO=0; order=[]
    updS.append('C'); updS.append('N'); updS.append('O')
    updA.append(newC); updA.append(newN); updA.append(newO)
    updD.append(-1.); updD.append(-1.); updD.append(-1.)
    while np.abs(oldN-newN)>=precisN:
        if iterN==0:
            iterN+=1
        else:
            oldN = copy.deepcopy(newN); iterC = 0; oldO = 0.
            newN, dN = V_CheckSpecWinds(objName, testName, 'N', spec, SpecWinds, colNames, speciesType='elem', updS=updS, updA=updA) #[7.74, 0.2] #[7.65, 0.] [7.4, 0.]
            if dN==0.:
                dN = 0.2
            iterN+=1; order.append('N'); newN=np.round(newN,3); dN=np.round(dN,3)
            updA[updS.index('N')] = newN*1.; updD[updS.index('N')] = dN*1.; precisN = dN*1.
            logging.info('N: '+str(oldN)+'->'+str(newN)+'+/-'+str(dN)+'; full house is: '+str(updS)+' '+str(updA))
            if np.abs(oldN-newN)<precisN:
                break
        while np.abs(oldO-newO)>=precisO:
            if iterC==0:
                iterO+=1; oldC = 0.
            else:
                oldO = copy.deepcopy(newO); oldC = 0.
                newO, dO = V_CheckSpecWinds(objName, testName, 'O', spec, SpecWinds, colNames, speciesType='elem', updS=updS, updA=updA) #[8.50, 0.2] #[8.47, 0.] [8.33, 0.]
                if dO==0.:
                    dO = 0.2
                iterO+=1; order.append('O'); newO=np.round(newO,3); dO=np.round(dO,3)
                updA[updS.index('O')] = newO*1.; updD[updS.index('O')] = dO*1.; precisO = dO*1.
                logging.info('O: '+str(oldO)+'->'+str(newO)+'+/-'+str(dO)+'; full house is: '+str(updS)+' '+str(updA))
            while np.abs(oldC-newC)>=precisC:
                newC, dC = V_CheckSpecWinds(objName, testName, 'C', spec, SpecWinds, colNames, speciesType='elem', updS=updS, updA=updA) #[8.45, 0.25] #[8.67, 0.] [8.76, 0.]
                if dC==0.:
                    dC = 0.2
                iterC+=1; order.append('C'); newC=np.round(newC,3); dC=np.round(dC,3)
                updA[updS.index('C')] = newC*1.; updD[updS.index('C')] = dC*1.; precisC = dC*1.
                logging.info('C: '+str(oldC)+'->'+str(newC)+'+/-'+str(dC)+'; full house is: '+str(updS)+' '+str(updA))
                oldC = copy.deepcopy(np.round(newC,3))
    return(updS, updA, updD, order)

def V_ElemOrder_NOiter(objName, testName, updS, updA, updD, met): # In case there are atomic lines of only C
    precisC = 0.01; oldC = 0.; newC = 8.43+met
    precisN = 0.01; oldN = 0.; newN = 7.83+met
    precisO = 0.01; oldO = 0.; newO = 8.69+met
    iterC=0; iterN=0; iterO=0; order=[]
    updS.append('C'); updS.append('N'); updS.append('O')
    updA.append(newC); updA.append(newN); updA.append(newO)
    updD.append(-1.); updD.append(-1.); updD.append(-1.)
    newC, dC = V_CheckSpecWinds(objName, testName, 'C', spec, SpecWinds, colNames, speciesType='elem', updS=updS, updA=updA) #[8.25, 0.2] #
    updA[updS.index('C')] = newC*1.; updD[updS.index('C')] = dC*1.; precisC = dC*1.
    logging.info('Starting point: '+str(updS)+' '+str(updA))
    while np.abs(oldN-newN)>=precisN:
        if iterN==0:
            iterN+=1
        else:
            oldN = copy.deepcopy(newN); iterC = 0; oldO = 0.
            newN, dN = V_CheckSpecWinds(objName, testName, 'N', spec, SpecWinds, colNames, speciesType='elem', updS=updS, updA=updA) #[8.12, 0.2] #
            if dN==0.:
                dN = 0.2
            iterN+=1; order.append('N'); newN=np.round(newN,3); dN=np.round(dN,3)
            updA[updS.index('N')] = newN*1.; updD[updS.index('N')] = dN*1.; precisN = dN*1.
            logging.info('N: '+str(oldN)+'->'+str(newN)+'+/-'+str(dN)+'; full house is: '+str(updS)+' '+str(updA))
            if np.abs(oldN-newN)<precisN:
                break
        while np.abs(oldO-newO)>=precisO:
            oldO = copy.deepcopy(newO)
            newO, dO = V_CheckSpecWinds(objName, testName, 'O', spec, SpecWinds, colNames, speciesType='elem', updS=updS, updA=updA) #[8.78, 0.2] #
            if dO==0.:
                dO = 0.2
            iterO+=1; order.append('O'); newO=np.round(newO,3); dO=np.round(dO,3)
            updA[updS.index('O')] = newO*1.; updD[updS.index('O')] = dO*1.; precisO = dO*1.
            logging.info('O: '+str(oldO)+'->'+str(newO)+'+/-'+str(dO)+'; full house is: '+str(updS)+' '+str(updA))
    #Last re-estimation
    newC, dC = V_CheckSpecWinds(objName, testName, 'C', spec, SpecWinds, colNames, speciesType='elem', updS=updS, updA=updA) #[8.25, 0.2] #
    updA[updS.index('C')] = newC*1.; updD[updS.index('C')] = dC*1.; precisC = dC*1.
    return(updS, updA, updD, order)

def V_CheckSpecWinds(objName, testName, species, spec, SpecWinds, colNames, speciesType='elem', updS=[], updA=[]):
    with open("Abunds/Abunds_%s%s/%s_SynthLines%s_%s.txt" % (objName, testName, objName, testName, species), "w") as outfile:
        outfile.write('Wave_low\twave_upp\tAbund\tchi-2\tSp.w. peak\n')
        if np.all(SpecWinds[colNames[species]]==0):
            outfile.write('-\t-\t-\t-\t-\n')
            newAb = O_MetScalSolAbund(objName, species); delAb = -1.
        else:
            abunds = V_Calc1Species(objName, testName, species, spec, SpecWinds, colNames, outfile, speciesType=speciesType, updateSpecie=updS, updateAbund=updA)
            newAb = np.nanmean(abunds); delAb = np.nanstd(abunds)
    print('Resulting A(%s)=%.2f+-%.2f' % (species, newAb, delAb))
    return(newAb, delAb)

def V_Calc1Species(objName, testName, species, spec, SpecWinds, colNames, outfile, speciesType, updateSpecie=[], updateAbund=[]):
    abunds = []; teff, logg, met, vmic, vsini = O_ReadParams(objName)
    pdf = PdfPages("Abunds/Abunds_%s%s/%s_SynthLines%s_%s.pdf" % (objName, testName, objName, testName, species))
    nonzeros = O_FindPatch(SpecWinds[colNames[species]])
    for patch in nonzeros:
        if (patch[-1]-patch[0])<3:
            patch[0]-=5; patch[-1]+=5
        print('Using patch with wavelengths from %.5f nm to %.5f nm' % (SpecWinds[0][patch[0]], SpecWinds[0][patch[1]-1]))
        cwl = (SpecWinds[0][patch[0]] + SpecWinds[0][patch[1]-1])/2.
        spec0, SpWnCut = O_CutWindow(spec, SpecWinds, colNames[species], cwl, species)
        specZ = V_Synth1Spec(SpWnCut, [teff, logg, met, vmic, vsini, species, 0.], updateSpecie=updateSpecie, updateAbund=updateAbund)
        spw = copy.deepcopy(SpWnCut); spw[1] = O_DisentangleBumps(spw, cwl)
        if speciesType=='elem':
            abCntr = I_IterateFib(objName, cwl, species, spec0, spw, updateSpecie, updateAbund)[0]
            specL,specM,specU = V_SynthSpecTriplet(cwl, abCntr, spw, colNames, species, speciesType=speciesType, updateSpecie=updateSpecie, updateAbund=updateAbund)
            chi2 = [O_CalcChi2(spec0[1],specZ[1],spw[1]), O_CalcChi2(spec0[1],specL[1],spw[1]), O_CalcChi2(spec0[1],specM[1],spw[1]), O_CalcChi2(spec0[1],specU[1],spw[1])]
            abunds.append(np.round(abCntr, 2)); outfile.write('%.5f\t%.5f\t%.2f\t%.2e\t%.2f\n' % (SpecWinds[0][patch[0]],SpecWinds[0][patch[1]-1],abCntr,chi2[2],np.nanmax(spw[1])))
            I_Plot(species, objName, abCntr, chi2, spec0, specZ, specL, specM, specU, SpWnCut, pdf)
            logging.info('Iterated A(%s)=%.2f with chi2=%.2f' % (species, abCntr, chi2[2]))
        elif speciesType=='isot':
            isoR = II_IterateFib(objName, cwl, species, spec0, spw, updateSpecie, updateAbund)
            if updateAbund:
                abund = updateAbund[updateSpecie.index(species[-1])]
            else:
                abund = O_MetScalSolAbund(objName, species[-1])
            specL,specM,specU = V_SynthSpecTriplet(cwl, abund, spw, colNames, species, isoR, speciesType=speciesType, updateSpecie=updateSpecie, updateAbund=updateAbund)
            chi2 = [O_CalcChi2(spec0[1],specZ[1],spw[1]), O_CalcChi2(spec0[1],specL[1],spw[1]), O_CalcChi2(spec0[1],specM[1],spw[1]), O_CalcChi2(spec0[1],specU[1],spw[1])]
            abunds.append(np.round(isoR, 2)); outfile.write('%.5f\t%.5f\t%.2f\t%.2e\t%.2f\n' % (SpecWinds[0][patch[0]],SpecWinds[0][patch[1]-1],isoR,chi2[2],np.nanmax(spw[1])))
            II_Plot(species, objName, isoR, chi2, spec0, specZ, specL, specM, specU, SpWnCut, pdf)
            logging.info('Iterated R(%s)=%.0f with chi2=%.2f' % (species, isoR, chi2[2]))
    pdf.close()
    return(abunds)
    
def V_SynthSpecTriplet(cwl, abCntr, SpecWinds, colNames, species, isoR=1., speciesType='elem', updateSpecie=[], updateAbund=[]):
    #--- Populating arrays (abundances to plot, spectral windows and spectra)
    teff, logg, met, vmic, vsini = O_ReadParams(objName)
    if speciesType=='elem':
        #--- Spectral regions synthesis -----------------------------------
        specL = V_Synth1Spec(SpecWinds, [teff, logg, met, vmic, vsini, species, abCntr-0.3], updateSpecie=updateSpecie, updateAbund=updateAbund)
        specM = V_Synth1Spec(SpecWinds, [teff, logg, met, vmic, vsini, species, abCntr], updateSpecie=updateSpecie, updateAbund=updateAbund)
        specU = V_Synth1Spec(SpecWinds, [teff, logg, met, vmic, vsini, species, abCntr+0.3], updateSpecie=updateSpecie, updateAbund=updateAbund)
    elif speciesType=='isot':
        abund = updateAbund[updateSpecie.index(species[-1])]
        isoRinit, isoRfin = O_IsoRatioRange(species)
        #--- Spectral regions synthesis -----------------------------------
        specL = V_Synth1Spec(SpecWinds, [teff, logg, met, vmic, vsini, species, abund], isoR=isoRinit, updateSpecie=updateSpecie, updateAbund=updateAbund)
        specM = V_Synth1Spec(SpecWinds, [teff, logg, met, vmic, vsini, species, abund], isoR=isoR, updateSpecie=updateSpecie, updateAbund=updateAbund)
        specU = V_Synth1Spec(SpecWinds, [teff, logg, met, vmic, vsini, species, abund], isoR=isoRfin, updateSpecie=updateSpecie, updateAbund=updateAbund)
    return(specL, specM, specU)

def V_Synth1Spec(SpecWinds, pars, isoR=1., updateSpecie=[], updateAbund=[]):
    Teff, logg, met, vmic, vsini, species, abund = pars
    alpha = ispec.determine_abundance_enchancements(met)
    macroturbulence = ispec.estimate_vmac(Teff, logg, met)
    limb_darkening_coeff = 0.6
    initial_R = 22500.
    RTcode = 'turbospectrum'
    atomic_linelist_file = iSpecFol + "mySample/outputAPOGEE/%7s/%7s_LineList.txt" % (objName, objName)
    atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file)
    isotope_file = iSpecFol + "/input/isotopes/SPECTRUM.lst"
    isotopes = ispec.read_isotope_data(isotope_file)
    solar_abundances_file = iSpecFol + "/input/abundances/Asplund.2009/stdatom.dat"
    solar_abundances = ispec.read_solar_abundances(solar_abundances_file)
    chemical_elements_file = iSpecFol + "/input/abundances/chemical_elements_symbols.dat"
    chemical_elements = ispec.read_chemical_elements(chemical_elements_file)
    model = iSpecFol + "input/atmospheres/MARCS.GES/" #ATLAS9.APOGEE
    modeled_layers_pack = ispec.load_modeled_layers_pack(model)
    regions = None
    if not ispec.valid_atmosphere_target(modeled_layers_pack, {'teff':Teff, 'logg':logg, 'MH':met, 'alpha':alpha}):
        msg = "The specified effective temperature, gravity (log g) and metallicity [M/H] \
                fall out of theatmospheric models."
        print(msg)
    atmosphere_layers = ispec.interpolate_atmosphere_layers(modeled_layers_pack, {'teff':Teff, 'logg':logg, 'MH':met, 'alpha':alpha}, code=RTcode)
    #--- Scaling all abundances -------------------------------------------
    fixed_abundances = ispec.create_free_abundances_structure(chemical_elements['symbol'][5:-33].tolist(), chemical_elements, solar_abundances) # from C to Pb
    fixed_abundances['Abund'] += met # Scale to metallicity
    if updateSpecie:
        for k in range(len(updateSpecie)):
            ID = chemical_elements['atomic_num'][chemical_elements['symbol']==updateSpecie[k]]
            fixed_abundances['Abund'][fixed_abundances['code']==ID] = np.round(updateAbund[k]-12.036, 2)
    #--- Populating known abundances --------------------------------------
    if len(species)==3:
        z = species[-1]
    else:
        z = species
    abElem = []; loge = []
    if species not in abElem:
        abElem.append(z); loge.append(abund)
    else:
        loge[abElem.index(z)] = abund
    abVals = [x-12.036 for x in loge]
    for k in range(len(abElem)):
        numb = chemical_elements['atomic_num'][chemical_elements['symbol']==abElem[k]]
        fixed_abundances['Abund'][fixed_abundances['code']==numb] = abVals[k] # Abundances in SPECTRUM scale (i.e., x - 12.0 - 0.036) and in the same order ["C", "N", "O"]
    #--- Modifying corresponding isotopic ratio --------------------------------
    if isoR!=1.:
        ratio = isoR
        if species=='13C':
            isotopes[10][3]=ratio/(ratio+1.); isotopes[11][3] = 1./(ratio+1.) #13C
        elif species=='15N':
            isotopes[12][3]=ratio/(ratio+1.); isotopes[13][3] = 1./(ratio+1.) #15N
        elif species=='17O':
            isotopes[15][3]=(1.-isotopes[16][3])/(ratio+1.); isotopes[14][3] = 1.-isotopes[15][3]-isotopes[16][3] #17O
        elif species=='18O':
            isotopes[16][3]=(1.-isotopes[15][3])/(ratio+1.); isotopes[14][3] = 1.-isotopes[16][3]-isotopes[15][3] #18O
    #--- Calculating synthetic spectrum -----------------------------------
    print("Creating synthetic spectrum for Teff=%.0fK, logg=%3s, [M/H]=%4s, Vmic=%.1fkm/s, A(%s)=%.2f..." % (Teff, logg, met, vmic, species, abund))
    synth_spec = ispec.create_spectrum_structure(SpecWinds[0])
    synth_spec['flux'] = ispec.generate_spectrum(synth_spec['waveobs'], \
            atmosphere_layers, Teff, logg, met, alpha, atomic_linelist, \
            isotopes, solar_abundances, fixed_abundances, \
            microturbulence_vel = vmic, macroturbulence=macroturbulence, \
            vsini=vsini, limb_darkening_coeff=limb_darkening_coeff, \
            R=initial_R, regions=regions, verbose=1, code=RTcode)
    synth_spec = np.array(tuple(zip(*synth_spec)))
    return(synth_spec)

def I_IterateFib(objName, cwl, elem, spec0, SpecWinds, updateSpecie=[], updateAbund=[]):
    teff, logg, met, vmic, vsini = O_ReadParams(objName)
    abund = O_MetScalSolAbund(objName, elem)
    # F(0)=0, F(1)=1, F(2)=1, F(3)=2, F(4)=3, F(5)=5, F(6)=8, F(7)=13, F(8)=21,
    # F(9)=34, F(10)=55, F(11)=89, F(12)=144, F(13)=233, F(14)=377, F(15)=610
    n = 14; fibon = O_Fib(n)/2.*0.01; a = abund-fibon; b = abund+fibon
    logging.info('%.3f nm: searching A(%s) (in range [%.2f; %.2f])' % (cwl, elem, a, b))
    l = a + O_Fib(n-2)/O_Fib(n)*(b-a); m = a + O_Fib(n-1)/O_Fib(n)*(b-a)
    specL = V_Synth1Spec(SpecWinds, [teff, logg, met, vmic, vsini, elem, l], updateSpecie=updateSpecie, updateAbund=updateAbund)
    specM = V_Synth1Spec(SpecWinds, [teff, logg, met, vmic, vsini, elem, m], updateSpecie=updateSpecie, updateAbund=updateAbund)
    fL = O_CalcChi2(spec0[1],specL[1],SpecWinds[1]); fM = O_CalcChi2(spec0[1],specM[1],SpecWinds[1])
    for k in range(2,n+1):
        if fL>fM:
            a = l; l = m; specL = specM; fL = fM
            m = a + O_Fib(n-k)/O_Fib(n-k+1)*(b-a)
            specM = V_Synth1Spec(SpecWinds, [teff, logg, met, vmic, vsini, elem, m], updateSpecie=updateSpecie, updateAbund=updateAbund)
            fM = O_CalcChi2(spec0[1],specM[1],SpecWinds[1])
        elif fL<fM:
            b = m; m = l; specM = specL; fM = fL
            l = a + O_Fib(n-k-1)/O_Fib(n-k+1)*(b-a)
            specL = V_Synth1Spec(SpecWinds, [teff, logg, met, vmic, vsini, elem, l], updateSpecie=updateSpecie, updateAbund=updateAbund)
            fL = O_CalcChi2(spec0[1],specL[1],SpecWinds[1])
    logging.info('%.3f nm: solution found at A(%s)=%.2f (in range [%.2f; %.2f])' % (cwl, elem, (a+b)/2., a, b))
    return(np.round((a+b)/2.,2))

def I_Plot(elem, objName, abCntr, chi2, spec0, specZ, specL, specM, specU, SpecWinds, pdf):
    #--- Plotting the spectra along with spectral windows (all in one) ----
    plt.rcParams["font.family"] = "Times New Roman"
    plt.title(objName)
    plt.grid()
    plt.xlim([SpecWinds[0][0]+2., SpecWinds[0][-1]-2.])
    plt.ylim([-0.2, 1.2])
#    plt.plot(spec0[0],SpecWinds[1],c='magenta',ls='--',lw=1,label='C spectral windows') # With addition of spectral windows, the plots become messy
#    plt.plot(spec0[0],SpecWinds[2],c='cyan',ls='--',lw=1,label='N spectral windows')
#    plt.plot(spec0[0],SpecWinds[3],c='olive',ls='--',lw=1,label='O spectral windows')
#    plt.plot(SpecWinds[0],SpecWinds[1],c='k',ls='--',lw=1, label=elem+' spectral windows')
    nonzeros = O_FindPatch(SpecWinds[1])
    for i,patch in enumerate(nonzeros):
        if i==0:
            plt.fill_between(SpecWinds[0][patch[0]:patch[1]-1], y1=-1., y2=2., color='lightgray', zorder=0, label=elem+' spectral windows')
        else:
            plt.fill_between(SpecWinds[0][patch[0]:patch[1]-1], y1=-1., y2=2., color='lightgray', zorder=0)
    plt.scatter(spec0[0], spec0[1], s=.5, c='k', label='APOGEE spectrum', zorder=1)
    plt.plot(specL[0],specL[1],c='r',ls='-',lw=.5, label='A(%s) = %.2f ($\chi^2$=%.2e)' % (elem, abCntr-0.3, chi2[1]))
    plt.plot(specM[0],specM[1],c='g',ls='-',lw=.5, label='A(%s) = %.2f ($\chi^2$=%.2e)' % (elem, abCntr, chi2[2]))
    plt.plot(specU[0],specU[1],c='b',ls='-',lw=.5, label='A(%s) = %.2f ($\chi^2$=%.2e)' % (elem, abCntr+0.3, chi2[3]))
    plt.plot(specZ[0],specZ[1],c='magenta',ls='-',lw=.5, label='A(%s) = 0.0 ($\chi^2$=%.2e)' % (elem, chi2[0]))
    plt.legend(ncol=1, prop={'size': 8}, loc='lower left')
    pdf.savefig(bbox_inches='tight', pad_inches=0)
    plt.close()
    
def II_IterateFib(objName, cwl, species, spec0, SpecWinds, updateSpecie=[], updateAbund=[]):
    teff, logg, met, vmic, vsini = O_ReadParams(objName)
    if updateAbund:
        abund = updateAbund[updateSpecie.index(species[-1])]
    else:
        abund = O_MetScalSolAbund(objName, species[-1])
    # F(0)=0, F(1)=1, F(2)=1, F(3)=2, F(4)=3, F(5)=5, F(6)=8, F(7)=13, F(8)=21,
    # F(9)=34, F(10)=55, F(11)=89, F(12)=144, F(13)=233, F(14)=377, F(15)=610
    a = 1.5; n = 12
    if species=='13C':
        fibon = O_Fib(n)*1. # From 1 to 145 (solar value ~92.5)
    else:
        fibon = O_Fib(n)*4. # From 1 to 577 (solar values: 15N~271, 17O~2630, 18O~487)
    b = a+fibon
    logging.info('%.3f nm: searching R(%s) (in range [%.2f; %.2f])' % (cwl, species, a, b))
    l = a + O_Fib(n-2)/O_Fib(n)*(b-a); m = a + O_Fib(n-1)/O_Fib(n)*(b-a)
    specL = V_Synth1Spec(SpecWinds, [teff, logg, met, vmic, vsini, species, abund], l, updateSpecie=updateSpecie, updateAbund=updateAbund)
    specM = V_Synth1Spec(SpecWinds, [teff, logg, met, vmic, vsini, species, abund], m, updateSpecie=updateSpecie, updateAbund=updateAbund)
    fL = O_CalcChi2(spec0[1],specL[1],SpecWinds[1]); fM = O_CalcChi2(spec0[1],specM[1],SpecWinds[1])
    for k in range(2,n+1):
        if fL>=fM:
            a = l; l = m; specL = specM; fL = fM
            m = a + O_Fib(n-k)/O_Fib(n-k+1)*(b-a)
            specM = V_Synth1Spec(SpecWinds, [teff, logg, met, vmic, vsini, species, abund], m, updateSpecie=updateSpecie, updateAbund=updateAbund)
            fM = O_CalcChi2(spec0[1],specM[1],SpecWinds[1])
        elif fL<fM:
            b = m; m = l; specM = specL; fM = fL
            l = a + O_Fib(n-k-1)/O_Fib(n-k+1)*(b-a)
            specL = V_Synth1Spec(SpecWinds, [teff, logg, met, vmic, vsini, species, abund], l, updateSpecie=updateSpecie, updateAbund=updateAbund)
            fL = O_CalcChi2(spec0[1],specL[1],SpecWinds[1])
    logging.info('%.3f nm: solution found at R(%s)=%.2f (in range [%.2f; %.2f])' % (cwl, species, (a+b)/2., a, b))
    return(np.round((a+b)/2.,2))

def II_Plot(species, objName, ratio, chi2, spec0, specZ, specL, specM, specU, SpecWinds, pdf):
    #--- Plotting the spectra along with spectral windows (all in one) ----
    plt.rcParams["font.family"] = "Times New Roman"
    plt.title(objName)
    plt.grid()
    plt.xlim([SpecWinds[0][0]+2., SpecWinds[0][-1]-2.])
    plt.ylim([-0.2, 1.2])
#    plt.plot(spec0[0],SpecWinds[1],c='magenta',ls='--',lw=1,label='C spectral windows') # With addition of spectral windows, the plots become messy
#    plt.plot(spec0[0],SpecWinds[2],c='cyan',ls='--',lw=1,label='N spectral windows')
#    plt.plot(spec0[0],SpecWinds[3],c='olive',ls='--',lw=1,label='O spectral windows')
#    plt.plot(SpecWinds[0],SpecWinds[1],c='k',ls='--',lw=1, label=O_IsoRatioName(species)[9:]+' spectral windows')
    isoRinit, isoRfin = O_IsoRatioRange(species)
    nonzeros = O_FindPatch(SpecWinds[1])
    for i,patch in enumerate(nonzeros):
        if i==0:
            plt.fill_between(SpecWinds[0][patch[0]:patch[1]-1], y1=-1., y2=2., color='lightgray', zorder=0, label=O_IsoRatioName(species)[9:]+' spectral windows')
        else:
            plt.fill_between(SpecWinds[0][patch[0]:patch[1]-1], y1=-1., y2=2., color='lightgray', zorder=0)
    plt.scatter(spec0[0], spec0[1], s=.5, c='k', label='APOGEE spectrum', zorder=1)
    plt.plot(specL[0],specL[1],c='r',ls='-',lw=.5, label='%s = %.0f ($\chi^2$=%.2e)' % (O_IsoRatioName(species), isoRinit, chi2[1]))
    plt.plot(specM[0],specM[1],c='g',ls='-',lw=.5, label='%s = %.0f ($\chi^2$=%.2e)' % (O_IsoRatioName(species), ratio, chi2[2]))
    plt.plot(specU[0],specU[1],c='b',ls='-',lw=.5, label='%s = %.0f ($\chi^2$=%.2e)' % (O_IsoRatioName(species), isoRfin, chi2[3]))
    plt.plot(specZ[0],specZ[1],c='magenta',ls='-',lw=.5, label='A(%s) = 0.0 ($\chi^2$=%.2e)' % (species[-1], chi2[0]))
    plt.legend(ncol=1, prop={'size': 8}, loc='lower left')
    pdf.savefig(bbox_inches='tight', pad_inches=0)
    plt.close()



if __name__ == '__main__':
    objName, testName, species, colNames, spec, SpecWinds = O_ReadData()
    sFound, aFound = V_ElemOrder(objName, testName, species, spec, SpecWinds, colNames)
