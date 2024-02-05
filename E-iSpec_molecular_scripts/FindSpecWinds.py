import os
import sys
import copy
import numpy as np
from scipy import interpolate as ip
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

iSpecFol = "/home/max/CallofPhDuty/iSpec_v20230804/"
sys.path.insert(0, os.path.abspath(iSpecFol))
import ispec
import logging
#--- Change LOG level -----------------------------------------------------
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
    species, delAb, sigNum = O_Initial(objName)
    elems = [s for s in species if len(s)<3]
    isots = [s for s in species if len(s)==3]
    logging.info('Reading stellar parameters and chemical abundances...')
    params = O_ReadStePar(objName)
    spec_solar = [O_ReadSolAbund(e) for e in elems]
    sols = [float(s)+12.036+params[2] for s in spec_solar]
    soliso = [sols[elems.index(iso[-1])] for iso in isots]
    return(objName, testName, species, delAb, sigNum, elems, isots, params, sols, soliso)

def O_Initial(objName):
    species = ['He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ge', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cs', 'Ba', 'Ce', 'Nd', 'Yb', '13C', '15N', '17O', '18O'] #['C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'S', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Dy', 'Er', 'Yb', 'Lu', 'Hf', 'W', 'Pb']
    print('Initialising calculation of spectral windows for %s' % objName)
    if not os.path.exists(iSpecFol+"mySample/WindowScripts/SpecWinds/SpecWinds_%s/" % objName):
        os.makedirs(iSpecFol+"mySample/WindowScripts/SpecWinds/SpecWinds_%s/" % objName)
    delAb = [1.] #[.1, 1., 2., 3.] # Abundance enhancement by 0.1, 1, 2, and 3 dex
    sigNum = [1.] #[.1, 1., 2., 3.] # Sensitivity threshold of 0.1, 1, 2, and 3 sigmas
    return(species, delAb, sigNum)

def O_DefineVisit(objName):
    if objName=='J065127':
        visit = '_visit2'
    elif objName=='J194853':
        visit = '_visit1'
    else:
        visit = ''
    return(visit)

def O_ReadStePar(objName):
    ###########################################################################
    ### Reading stellar parameters from 'Spoiler.txt'
    Aname, Ateff, Alogg, Amet, Avmic, Acomm = np.loadtxt(iSpecFol+"Spoiler.txt", delimiter='\t', dtype=np.dtype([('name','U8'), ('teff',float), ('logg',float), ('met',float), ('vmic',float), ('comm','U5')]), skiprows=1, unpack=True)
    params = []
    params.append(float(Ateff[Aname==objName]))
    params.append(float(Alogg[Aname==objName]))
    params.append(float(Amet[Aname==objName]))
    params.append(float(Avmic[Aname==objName]))
    return(params)

def O_ReadSolAbund(elem):
    if len(elem)==3:
        elem = elem[-1]
    # Reading solar abundances from Asplund (2009)
    solar_abundances = ispec.read_solar_abundances(iSpecFol+"input/abundances/Asplund.2009/stdatom.dat")
    chemical_elements = ispec.read_chemical_elements(iSpecFol+"input/abundances/chemical_elements_symbols.dat")
    ID = chemical_elements['atomic_num'][chemical_elements['symbol']==elem]
    solar_abund = float(solar_abundances['Abund'][solar_abundances['code']==ID])
    return(solar_abund)

def O_SigmaClip(SpecWinds, sigNum, stddev, elems):
    # Adjusting spectral windows
    for i in range(1, len(SpecWinds)):
        if len(elems[i-1])!=3:
            SpecWinds[i][SpecWinds[i]<sigNum*stddev] = 0. #y -= sigNum*stddev; y[y<0.] = 0.
        elif len(elems[i-1])==3:
            SpecWinds[i][np.abs(SpecWinds[i])<sigNum*stddev] = 0.
    return(SpecWinds)

def O_FindPatch(window):
    # Create an array that is 1 where a is `value`, and pad each end with an extra 0.
    isvalue = np.concatenate(([0.], np.not_equal(window, 0.).view(np.int8), [0.]))
    absdiff = np.abs(np.diff(isvalue))
    # Runs start and end when absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return(ranges)

def O_RecordLineRegs(objName, SpecWinds, sigNum, delAb, elems):
    # Calculating "dirty" spectral window peaks
    if not os.path.exists(iSpecFol+"mySample/WindowScripts/LineRegs/%s/" % objName):
        os.makedirs(iSpecFol+"mySample/WindowScripts/LineRegs/%s/" % objName)
    fOut = open(iSpecFol+"mySample/WindowScripts/LineRegs/%s/%s_LineRegs_s%i_d%.1f.txt" % (objName, objName, sigNum, delAb), "w")
    fOut.write('wave_peak\twave_base\twave_top\tnote\n')
    for j in range(1, len(SpecWinds)):
        nonzeros = O_FindPatch(SpecWinds[j])
        for patch in nonzeros:
            wave_base = SpecWinds[0][patch[0]]; wave_top = SpecWinds[0][patch[1]-1]
            fOut.write('%.4f\t%.4f\t%.4f\t%s\n' % ((wave_base+wave_top)/2., wave_base, wave_top, elems[j-1]))
    fOut.close()

def O_RenormSpW(SpecWinds):
    for k in range(1, len(SpecWinds)):
        if max(SpecWinds[k])!=0.:
            SpecWinds[k] *= .5/max(SpecWinds[k])
    return(SpecWinds)

def O_Merge(SpecWinds, SpecWindsIso):
    SpecWindsAll = np.vstack((SpecWinds, SpecWindsIso[1:]))
    return(SpecWindsAll)
    
def O_PlotAll(elems, objName, testName, delAb, sigNum, params, sols, enhs, SpecWinds):
    pdf = PdfPages(iSpecFol+"mySample/WindowScripts/SpecWinds/SpecWinds_%s/%s_SpecWinds%s_s%1i_d%.1f.pdf" % (objName, objName, testName, sigNum, delAb))
    for k in range(len(elems)):
        plt.rcParams["font.family"] = "Times New Roman"
        plt.figure(figsize=(11,8))
        plt.grid()
        plt.xlabel('Wavelength (nm)')
        if len(elems[k])<3:
            print('Plotting spectral windows of %2s (dA = %.2f dex, %1i-sigma): %2i/%2i' % (elems[k],delAb,sigNum,k+1,len(elems)))
            plt.ylabel('$F_{\\rm A(%s)=%.2f}-F_{\\rm A(%s)=%.2f}$' % (elems[k], sols[k], elems[k], enhs[k]))
            plt.title('%s ($T_{\\rm eff}$=%.0f K, log$g$=%.1f, [M/H]=%.1f, $v_{\\rm mic}$=%.1f km/s)\n%s spectral windows (above %.1f-$\sigma_{\\rm tot}$ level)' % (objName, params[0], params[1], params[2], params[3], elems[k], sigNum))
            plt.plot(SpecWinds[0],SpecWinds[k+1],c='k',ls='-',lw=1)
            pdf.savefig()
            plt.close()
        else:
            knew = elems.index(elems[k][-1])
            print('Plotting spectral windows of %s (A = %.2f dex, %1i-sigma): %2i/%2i' % (elems[k], sols[knew], sigNum, k+1, len(elems)))
            plt.ylabel('$F_\odot-F_{\\rm custom}$ (A(%s)=%.2f)' % (elems[knew], sols[knew]))
            plt.title('%s ($T_{\\rm eff}$=%.0f K, log$g$=%.1f, [M/H]=%.1f, $v_{\\rm mic}$=%.1f km/s)\n%s spectral windows' % (objName, params[0], params[1], params[2], params[3], elems[k]))
            plt.plot(SpecWinds[0],SpecWinds[k+1],c='k',ls='-',lw=1)
            pdf.savefig()
            plt.close()
    pdf.close()

def O_WriteAll(elems, objName, testName, delAb, sigNum, SpecWinds):
    print(' Writing spectral windows (dA = %.1f dex, %1i-sigma)'%(delAb,sigNum))
    fAll = open(iSpecFol+"mySample/WindowScripts/SpecWinds/SpecWinds_%s/%s_SpecWinds%s_s%1i_d%.1f.txt" % (objName, objName, testName, sigNum, delAb), "w")
    for i in range(len(SpecWinds[0])):
        fAll.write('%14.12f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f %10.8f\n' % (SpecWinds[0][i]/1000., SpecWinds[1][i], SpecWinds[2][i], SpecWinds[3][i], SpecWinds[4][i], SpecWinds[5][i], SpecWinds[6][i], SpecWinds[7][i], SpecWinds[8][i], SpecWinds[9][i], SpecWinds[10][i], SpecWinds[11][i], SpecWinds[12][i], SpecWinds[13][i], SpecWinds[14][i], SpecWinds[15][i], SpecWinds[16][i], SpecWinds[17][i], SpecWinds[18][i], SpecWinds[19][i], SpecWinds[20][i], SpecWinds[21][i], SpecWinds[22][i], SpecWinds[23][i], SpecWinds[24][i], SpecWinds[25][i], SpecWinds[26][i], SpecWinds[27][i], SpecWinds[28][i], SpecWinds[29][i], SpecWinds[30][i], SpecWinds[31][i], SpecWinds[32][i], SpecWinds[33][i], SpecWinds[34][i], SpecWinds[35][i], SpecWinds[36][i], SpecWinds[37][i], SpecWinds[38][i], SpecWinds[39][i], SpecWinds[40][i], SpecWinds[41][i], SpecWinds[42][i], SpecWinds[43][i], SpecWinds[44][i], SpecWinds[45][i], SpecWinds[46][i], SpecWinds[47][i], SpecWinds[48][i], SpecWinds[49][i], SpecWinds[50][i]))
    fAll.close()

def i_CleanLineRegs(x, y, elem, objName):
    if objName=='J065127':
        codename = 'SZMon'
    elif objName=='J194853':
        codename = 'DFCyg'
    else:
        codename = objName
    line_regions = ispec.read_line_regions("%smySample/WindowScripts/LineRegs/%s_all.txt" % (iSpecFol, codename))
    goodRegions = (x>1700.)
    for line in line_regions:
        if elem==line['note']:
            goodRegions = np.logical_or(goodRegions, (x>line['wave_base'])&(x<line['wave_top']))
    y[~goodRegions] = 0.
    y[goodRegions] += 1.
    return(x, y)

def V_GenSpec(objName, elem, abund, isotope=None, regime='turbospectrum'):
    t, l, m, v = O_ReadStePar(objName)
    Teff = [t]; logg = [l]; met = [m]; vmic = [v]
    if elem is None:
        elem = 'Fe'
    if abund==-1.:
        abund = O_ReadSolAbund(elem) + 12.036 + m
    elems = [elem]; abunds = [abund]; modelList = []
    for t in Teff:
        for l in logg:
            for m in met:
                for v in vmic:
                    for e in elems:
                        # Variating different variables
                        #abund = [ab+12.036+m] #[0., ab+12.036+m, ab+12.136+m, ab+13.036+m, ab+14.036+m, ab+15.036+m] #np.arange(7., 9.2, 0.5).tolist() #[7.1, 7.2, 7.3, 7.4, 7.6, 7.7, 7.8, 7.9]
                        for a in abunds:
                            modelList.append([t, l, m, v, e, a])
    for model in modelList:
        spec = V_SpecSynth(objName, model, isotope, code="turbospectrum", regime=regime)
    return(spec['waveobs'], spec['flux'], spec['err'])

def V_SpecSynth(objName, pars, iso=None, code="turbospectrum", regime="derivAbu"):
    visit = O_DefineVisit(objName)
    star_spectrum = ispec.read_spectrum(iSpecFol+"mySample/outputAPOGEE/%s/%s_CleanSpec%s.txt" % (objName, objName, visit)) # visit2 for SZ Mon, visit1 for DF Cyg
    initial_R = 22500.; Teff, logg, met, vmic, elem, abund = pars
    alpha = ispec.determine_abundance_enchancements(met)
    macroturbulence = ispec.estimate_vmac(Teff, logg, met)
    limb_darkening_coeff = 0.6; vsini = 0.
    mySamOut_dir = "mySample/outputAPOGEE/%7s/" % objName
    atomic_linelist_file = iSpecFol+"mySample/outputAPOGEE/%7s/%7s_LineList.txt" % (objName, objName) #iSpecFol + "input/linelists/transitions/VALD.300_1100nm/atomic_lines.tsv" #APOGEE.1500_1700nm
    atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file)
    isotope_file = iSpecFol + "input/isotopes/SPECTRUM.lst"
    isotopes = ispec.read_isotope_data(isotope_file)
    solar_abundances_file = iSpecFol + "input/abundances/Asplund.2009/stdatom.dat"
    solar_abundances = ispec.read_solar_abundances(solar_abundances_file)
    chemical_elements_file = iSpecFol + "input/abundances/chemical_elements_symbols.dat"
    chemical_elements = ispec.read_chemical_elements(chemical_elements_file)
    model = iSpecFol + "input/atmospheres/ATLAS9.KuruczODFNEW/"
    modeled_layers_pack = ispec.load_modeled_layers_pack(model)
    regions = None

    if not ispec.valid_atmosphere_target(modeled_layers_pack, {'teff':Teff, 'logg':logg, 'MH':met, 'alpha':alpha}):
        msg = "The specified effective temperature, gravity (log g) and metallicity [M/H] \
                fall out of theatmospheric models."
        print(msg)

    atmosphere_layers = ispec.interpolate_atmosphere_layers(modeled_layers_pack, {'teff':Teff, 'logg':logg, 'MH':met, 'alpha':alpha}, code=code)

    if regime=="derivAbu":
        ##--- Scaling all abundances --------------------------------------
        fixed_abundances = ispec.create_free_abundances_structure(chemical_elements['symbol'][1:-33].tolist(), chemical_elements, solar_abundances) # from He to Pb
        fixed_abundances['Abund'] += met # Scale to metallicity
        ##--- Populating known abundances ---------------------------------
        abElem = []; loge = []
        if elem not in abElem:
            abElem.append(elem)
            loge.append(abund)
        else:
            index = abElem.index(elem)
            loge[index] = abund
        abVals = [x-12.036 for x in loge]
        for k in range(len(abElem)):
            numb = chemical_elements['atomic_num'][chemical_elements['symbol']==abElem[k]]
            fixed_abundances['Abund'][fixed_abundances['code']==numb] = abVals[k] # Abundances in SPECTRUM scale (i.e., x - 12.0 - 0.036) and in the same order ["C", "N", "O"]
        ##--- Calculating synthetic spectrum ------------------------------
        logging.info("Creating synthetic spectrum for Teff=%.0fK, logg=%3s, [M/H]=%4s, Vmic=%.1fkm/s, logeps(%s)=%.2f..." % (Teff, logg, met, vmic, elem, abund))
        synth_spec = ispec.create_spectrum_structure(star_spectrum['waveobs'])
        synth_spec['flux'] = ispec.generate_spectrum(synth_spec['waveobs'], \
                atmosphere_layers, Teff, logg, met, alpha, atomic_linelist, \
                isotopes, solar_abundances, fixed_abundances, \
                microturbulence_vel = vmic, macroturbulence=macroturbulence, \
                vsini=vsini, limb_darkening_coeff=limb_darkening_coeff, \
                R=initial_R, regions=regions, verbose=1, code=code)
        synth_spec['flux'][star_spectrum['flux']<0.] = 0.
        ##--- Save spectrum -----------------------------------------------
        logging.info("Saving synthetic spectrum for Teff=%.0fK, logg=%.1f, [M/H]=%.1f, Vmic=%.1fkm/s, logeps(%s)=%.2f..." % (Teff, logg, met, vmic, elem, abund))
        dirName = iSpecFol+"mySample/SynthSpectra/abund_%7s_deriv%s/" % (objName, testName)
        if not os.path.exists(dirName):
            os.makedirs(dirName)
        f = open(dirName+"synth_%.0f_%.1f_%.1f_%.1f_%s_%.2f.txt" % (Teff, logg, met, vmic, elem, abund), "w")
        f.write('waveobs\tflux\terr\n')
        for s in synth_spec:
            f.write('%.5f\t%.7f\t%.7f\n' % (s['waveobs'],s['flux'],s['err']))
        f.close()

    if regime=="derivIso":
        ##--- Scaling all abundances --------------------------------------
        fixed_abundances = ispec.create_free_abundances_structure(chemical_elements['symbol'][1:-33].tolist(), chemical_elements, solar_abundances) # from He to Pb
        fixed_abundances['Abund'] += met # Scale to metallicity
        ##--- Modifying isotopic ratio ------------------------------------
        ratio = 2.
        isotope_ratios = {
            '13C': (10, 11),
            '15N': (12, 13),
            '17O': (15, 14, 16),
            '18O': (16, 14, 15),
        }
        if iso in isotope_ratios:
            indices = isotope_ratios[iso]
            if len(indices)==2: #C and N have only two isotopes each
                ratios = [ratio/(ratio+1.) if i == indices[0] else 1./(ratio+1.) for i in indices]
            else: #O has three isotopes
                ratios = [ratio/(ratio+1.) if i == indices[0] else 1./(ratio+1.) for i in indices[:-1]]
                ratios = [r*(1.-isotopes[indices[-1]][3]) for r in ratios]
            for i, r in zip(indices, ratios):
                isotopes[i][3] = r
            abund = {'13C': 8.43, '15N': 7.83, '17O': 8.69, '18O': 8.69}[iso] + met
        ##--- Populating abundances -----------------------------------
        abElem = [elem]; loge = [abund]; abVals = [x-12.036 for x in loge]
        for k in range(len(abElem)):
            numb = chemical_elements['atomic_num'][chemical_elements['symbol']==abElem[k]]
            fixed_abundances['Abund'][fixed_abundances['code']==numb] = abVals[k] # Abundances in SPECTRUM scale (i.e., x - 12.0 - 0.036) and in the same order ["C", "N", "O"]
        ##--- Calculating synthetic spectrum --------------------------
        logging.info(f"Creating synthetic spectrum for Teff={Teff:.0f}K, logg={logg:.1f}, [M/H]={met:.1f}, Vmic={vmic:.1f}km/s, logeps({elem})={abund:.2f}, iso={iso}...")
        synth_spec = ispec.create_spectrum_structure(star_spectrum['waveobs'])
        synth_spec['flux'] = ispec.generate_spectrum(synth_spec['waveobs'], \
            atmosphere_layers, Teff, logg, met, alpha, atomic_linelist, \
            isotopes, solar_abundances, fixed_abundances, \
            microturbulence_vel = vmic, macroturbulence=macroturbulence, \
            vsini=vsini, limb_darkening_coeff=limb_darkening_coeff, \
            R=initial_R, regions=regions, verbose=1, code=code)
        synth_spec['flux'][star_spectrum['flux']<0.] = 0.
        ##--- Save spectrum -------------------------------------------
        logging.info(f"Saving synthetic spectrum for Teff={Teff:.0f}K, logg={logg:.1f}, [M/H]={met:.1f}, Vmic={vmic:.1f}km/s, logeps({elem})={abund:.2f}, iso={iso}...")
        from pathlib import Path
        dirName = Path(iSpecFol) / f"mySample/SynthSpectra/abund_{objName}_deriv{testName}/"
        dirName.mkdir(parents=True, exist_ok=True)
        f = open(dirName / f"synth_{Teff:.0f}_{logg:.1f}_{met:.1f}_{vmic:.1f}_{elem}_{abund:.2f}_{iso}.txt", "w")
        f.write('waveobs\tflux\terr\n')
        for s in synth_spec:
            f.write(f"{s['waveobs']:.5f}\t{s['flux']:.7f}\t{s['err']:.7f}\n")
        f.close()
    return(synth_spec)

def I_Sense(elems, objName, delAb, params, sols, enhs):
    ###########################################################################
    ### Populating arrays (abundances to plot, spectral windows and spectra)
    SpecWinds=[]
    # To synthesize the "normal" spectrum on-the-fly (too long)
    #lamb0, flux0, flerr0 = V_GenSpec(objName, None, -1., regime='derivAbu')
    # To use a synthetic "normal" spectrum from a pre-generated grid
    lamb0, flux0, flerr0 = np.loadtxt(iSpecFol+"mySample/SynthSpectra/abund_%7s_deriv/synth_%.0f_%.1f_%.1f_%.1f_Fe_%.2f.txt" % (objName, params[0], params[1], params[2], params[3], 7.5+params[2]), delimiter='\t', unpack=True, skiprows=1)
    for k in range(len(elems)):
        # To synthesize the "enhanced" spectrum on-the-fly (too long)
        #lamb1, flux1, flerr1 = V_GenSpec(objName, elems[k], enhs[k], regime='derivAbu')
        # To use a synthetic "enhanced" spectrum from a pre-generated grid
        lamb1, flux1, flerr1 = np.loadtxt(iSpecFol+"mySample/SynthSpectra/abund_%7s_deriv/synth_%.0f_%.1f_%.1f_%.1f_%s_%.2f.txt" % (objName, params[0], params[1], params[2], params[3], elems[k], enhs[k]), delimiter='\t', unpack=True, skiprows=1)
        stddev = np.nanstd(flux0[flux0>0.])
        #######################################################################
        ### Interpolating second array just in case it's shifted or different length
        if not np.array_equal(lamb0, lamb1, equal_nan=False):
            print('Adopting scale of %s (%.1f dex - %.1f dex)' % (elems[k], sols[k], enhs[k]))
            f = ip.interp1d(lamb1,flux1,fill_value="extrapolate"); flux1 = f(lamb0)
            ferr = ip.interp1d(lamb1,flerr1,fill_value="extrapolate"); flerr1 = ferr(lamb0)
        #######################################################################
        ### Calculating "derivatives" y = dFlux/dAbund
        x = lamb0
        y = (flux0-flux1)/delAb
        #######################################################################
        ### Cleaning spectral windows
        x, y = i_CleanLineRegs(x, y, elems[k], objName)
        #######################################################################
        ### Populating spectral windows 2D array
        if k==0:
            SpecWinds.append(x)
        SpecWinds.append(y)
    return(SpecWinds, stddev)

def II_SenseIso(iso, objName, params, sol):
    ###########################################################################
    ### Populating arrays (abundances to plot, spectral windows and spectra)
    SpecWindsIso=[]
    # To synthesize the "normal" spectrum on-the-fly (too long)
    #lamb0, flux0, flerr0 = V_GenSpec(objName, None, -1., regime='derivIso')
    # To use a synthetic "normal" spectrum from a pre-generated grid
    lamb0, flux0, flerr0 = np.loadtxt(iSpecFol+"mySample/SynthSpectra/abund_%7s_deriv/synth_%.0f_%.1f_%.1f_%.1f_Fe_%.2f.txt" % (objName, params[0], params[1], params[2], params[3], 7.5+params[2]), delimiter='\t', unpack=True, skiprows=1)
    for k in range(len(iso)):
        # To synthesize the "enhanced" spectrum on-the-fly (too long)
        #lamb1, flux1, flerr1 = V_GenSpec(objName, iso[k][-1], -1., iso[k], regime='derivIso')
        # To use a synthetic "enhanced" spectrum from a pre-generated grid
        lamb1, flux1, flerr1 = np.loadtxt(iSpecFol+"mySample/SynthSpectra/abund_%7s_deriv/synth_%.0f_%.1f_%.1f_%.1f_%s_%.2f_%s.txt" % (objName, params[0], params[1], params[2], params[3], iso[k][-1], sol[k], iso[k]), delimiter='\t', unpack=True, skiprows=1)
        stddev = np.nanstd(flux0[flux0>0.])
        #######################################################################
        ### Interpolating second array just in case it's shifted or different length
        if not np.array_equal(lamb0, lamb1, equal_nan=False):
            print('Adopting wavelength scale of main isotope of %s' % iso[k])
            f = ip.interp1d(lamb1,flux1,fill_value="extrapolate"); flux1 = f(lamb0)
            ferr = ip.interp1d(lamb1,flerr1,fill_value="extrapolate"); flerr1 = ferr(lamb0)
        #######################################################################
        ### Calculating "derivatives" y = dF/dA
        x = lamb0
        y = (flux0-flux1)/delAb
        #######################################################################
        ### Cleaning spectral windows
        x, y = i_CleanLineRegs(x, y, iso[k], objName)
        #######################################################################
        ### Populating spectral windows 2D array
        if k==0:
            SpecWindsIso.append(x)
        SpecWindsIso.append(y)
    return(SpecWindsIso, stddev)



if __name__ == '__main__':
    objName, testName, species, delAb, sigNum, elems, isots, params, sols, soliso = O_ReadData()
    #--- Isotopes (only between Rsun and custom R=2) ------------------------------
    SpecWindsIso, stddevIso = II_SenseIso(isots, objName, params, soliso)
    for d in delAb:
        #--- Elements -----------------------------------------------------
        enhs = [sol+d for sol in sols] # enhanced by 'd' dex
        SpecWinds, stddev = I_Sense(elems, objName, d, params, sols, enhs)
        for s in sigNum:
            SpecWinds = O_SigmaClip(SpecWinds, s, stddev, elems)
            SpecWindsIso = O_SigmaClip(SpecWindsIso, s, stddevIso, isots)
            SpecWindsAll = O_Merge(SpecWinds, SpecWindsIso)
            #--- All species ----------------------------------------------
            SpecWindsAll = O_RenormSpW(SpecWindsAll)
            O_RecordLineRegs(objName, SpecWindsAll, s, d, species)
            O_PlotAll(species, objName, testName, d, s, params, sols, enhs, SpecWindsAll)
            O_WriteAll(species, objName, testName, d, s, SpecWindsAll)
    pass
