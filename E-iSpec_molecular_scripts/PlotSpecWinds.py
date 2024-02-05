import os
import sys
import numpy as np
from scipy import interpolate as ip
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

iSpecFol = "/home/max/CallofPhDuty/iSpec_v20230804/"
sys.path.append(iSpecFol)
import ispec

def Initial(code='etom'):
    if code=='etom':
        elems = ['He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ge', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cs', 'Ba', 'Ce', 'Nd', 'Yb'] #['13C', '15N', '17O', '18O'] #['C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'S', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Dy', 'Er', 'Yb', 'Lu', 'Hf', 'W', 'Pb']
    elif code=='isot':
        elems = ['13C', '15N', '17O', '18O']
    elif code=='all':
        elems = ['He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ge', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cs', 'Ba', 'Ce', 'Nd', 'Yb', '13C', '15N', '17O', '18O']
    sigRead=1; delRead=1.
    colNames, dt, SpWnNm, skiprows = InitForm(objName, sigRead, delRead, code=code)
    if not os.path.exists(iSpecFol+"mySample/WindowScripts/Abunds/Abunds_%s/" % objName):
        os.makedirs(iSpecFol+"mySample/WindowScripts/Abunds/Abunds_%s/" % objName)
    return(elems,sigRead,delRead,colNames,dt,SpWnNm,skiprows)

def InitForm(objName, sigRead, delRead, code='etom'):
    ###########################################################################
    ### Initialising reading formats and files
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
        SpWnNm = iSpecFol+"mySample/WindowScripts/SpecWinds/SpecWinds_%s/%s_SpecWinds_s%1i_d%.1f.txt" % (objName, objName, sigRead, delRead)
        skiprows=0
    return(colNames, dt, SpWnNm, skiprows)

def ReadParams():
    ###########################################################################
    ### Reading the stellar parameters from 'Spoiler.txt'
    Aname, Ateff, Alogg, Amet, Avmic, Acomm = np.loadtxt(iSpecFol+"Spoiler.txt", delimiter='\t', dtype=np.dtype([('name','U8'), ('teff',float), ('logg',float), ('met',float), ('vmic',float), ('comm','U5')]), skiprows=1, unpack=True)
    teff = float(Ateff[Aname==objName]); logg = float(Alogg[Aname==objName])
    met = float(Amet[Aname==objName]); vmic = float(Avmic[Aname==objName])
    return(teff, logg, met, vmic)
    
def ReadWinds(SpWnNm, dt, skiprows, code):
    SpecWinds = np.loadtxt(SpWnNm, dtype=dt, delimiter=' ', unpack=True, skiprows=skiprows)
    ###########################################################################
    ### Micron-to-nanometer conversion (+vacuum-to-air for ASPCAP)
    if code=='ASPCAP':
        SpecWinds[0] *= 10000.
        SpecWinds[0] /= (1. + 5.792105E-2/(238.0185 - (1.E4/SpecWinds[0])**2) + 1.67917E-3/(57.362 - (1.E4/SpecWinds[0])**2))
        SpecWinds[0] /= 10.
    elif code=='self' or code=='etom' or code=='isot' or code=='all':
        SpecWinds[0] *= 1000.
    for k in range(len(SpecWinds)):
        SpecWinds[k] = np.round(SpecWinds[k], 5)
    return(SpecWinds)

def SynthSpec(SpecWinds, pars, code=None):
    Teff, logg, met, vmic, vsini, elem, abund = pars
    alpha = ispec.determine_abundance_enchancements(met)
    macroturbulence = ispec.estimate_vmac(Teff, logg, met)
    limb_darkening_coeff = 0.6
    initial_R = 22500.
    codeRTC = 'turbospectrum'
    atomic_linelist_file = iSpecFol + "mySample/outputAPOGEE/%7s/%7s_LineList.txt" % (objName, objName)
    atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file)
    isotope_file = iSpecFol + "/input/isotopes/SPECTRUM.lst"
    isotopes = ispec.read_isotope_data(isotope_file)
    solar_abundances_file = iSpecFol + "/input/abundances/Asplund.2009/stdatom.dat"
    solar_abundances = ispec.read_solar_abundances(solar_abundances_file)
    chemical_elements_file = iSpecFol + "/input/abundances/chemical_elements_symbols.dat"
    chemical_elements = ispec.read_chemical_elements(chemical_elements_file)
    model = iSpecFol + "input/atmospheres/ATLAS9.APOGEE/"
    modeled_layers_pack = ispec.load_modeled_layers_pack(model)
    regions = None
    if not ispec.valid_atmosphere_target(modeled_layers_pack, {'teff':Teff, 'logg':logg, 'MH':met, 'alpha':alpha}):
        msg = "The specified effective temperature, gravity (log g) and metallicity [M/H] \
                fall out of theatmospheric models."
        print(msg)
    atmosphere_layers = ispec.interpolate_atmosphere_layers(modeled_layers_pack, {'teff':Teff, 'logg':logg, 'MH':met, 'alpha':alpha}, code=codeRTC)
    ##--- Scaling all abundances ------------------------------------------
    fixed_abundances = ispec.create_free_abundances_structure(chemical_elements['symbol'][5:-33].tolist(), chemical_elements, solar_abundances) # from C to Pb
    fixed_abundances['Abund'] += met # Scale to metallicity
    ##--- Populating known abundances -------------------------------------
    if code is None:
        if len(elem)>1:
            if elem[0]=='1':
                z = elem[-1]
            else:
                z = elem
        else:
            z = elem
        abElem = []; loge = []
        if elem not in abElem:
            abElem.append(z)
            loge.append(abund)
        else:
            index = abElem.index(z)
            loge[index] = abund
        abVals = [x-12.036 for x in loge]
        for k in range(len(abElem)):
            numb = chemical_elements['atomic_num'][chemical_elements['symbol']==abElem[k]]
            fixed_abundances['Abund'][fixed_abundances['code']==numb] = abVals[k] # Abundances in SPECTRUM scale (i.e., x - 12.0 - 0.036) and in the same order ["C", "N", "O"]
    ##--- Modifying isotopic ratio ----------------------------------------
    if code=='isot':
        ratio = 2.
        if elem=='13C':
            if abund<0.:
                isotopes[10][3]=ratio/(ratio+1.); isotopes[11][3] = 1./(ratio+1.) #13C
            abund = 8.43+met
        if elem=='15N':
            if abund<0.:
                isotopes[12][3]=ratio/(ratio+1.); isotopes[13][3] = 1./(ratio+1.) #15N
            abund = 7.83+met
        if elem=='17O':
            if abund<0.:
                isotopes[15][3]=(1.-isotopes[16][3])/(ratio+1.); isotopes[14][3] = 1.-isotopes[15][3]-isotopes[16][3] #17O
            abund = 8.69+met
        if elem=='18O':
            if abund<0.:
                isotopes[16][3]=(1.-isotopes[15][3])/(ratio+1.); isotopes[14][3] = 1.-isotopes[16][3]-isotopes[15][3] #18O
            abund = 8.69+met
    ##--- Calculating synthetic spectrum ----------------------------------
    print("Creating synthetic spectrum for Teff=%.0fK, logg=%3s, [M/H]=%4s, Vmic=%.1fkm/s, A(%s)=%.2f..." % (Teff, logg, met, vmic, elem, abund))
    synth_spec = ispec.create_spectrum_structure(SpecWinds[0])
    synth_spec['flux'] = ispec.generate_spectrum(synth_spec['waveobs'], \
            atmosphere_layers, Teff, logg, met, alpha, atomic_linelist, \
            isotopes, solar_abundances, fixed_abundances, \
            microturbulence_vel = vmic, macroturbulence=macroturbulence, \
            vsini=vsini, limb_darkening_coeff=limb_darkening_coeff, \
            R=22500., regions=regions, verbose=1, code=codeRTC)
    return(synth_spec)

def FixWls(spec,SpecWinds):
    ###########################################################################
    ### Bringing everything to the same wavelengths
    f = ip.interp1d(spec[0],spec[1],fill_value="extrapolate"); spec[1] = f(SpecWinds[0])
    ferr = ip.interp1d(spec[0],spec[2],fill_value="extrapolate"); spec[2] = ferr(SpecWinds[0])
    return(spec)
    
def FindPatch(window):
    # Create an array that is 1 where a is `value`, and pad each end with an extra 0.
    isvalue = np.concatenate(([0.], np.not_equal(window, 0.).view(np.int8), [0.]))
    absdiff = np.abs(np.diff(isvalue))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return(ranges)

def Plot(elem, objName, spec0, spec1, SpecWinds, pdf):
    ###########################################################################
    ### Plotting the spectra along with spectral windows (all in one)
    plt.title(objName)
    plt.grid()
    plt.xlim([np.nanmedian(SpecWinds[0])-4., np.nanmedian(SpecWinds[0])+4.])
    plt.ylim([-0.2, 1.2])
    plt.plot(spec0[0],SpecWinds[1],c='k',ls='--',lw=1, label='%s spectral windows' % elem)
    plt.plot(spec0[0],spec0[1],c='k',ls='-',lw=1, label='Obs')
    plt.plot(spec1[0],spec1[1],c='magenta',ls='-',lw=.5, label='log$\epsilon$ (%s) = 0.0' % elem)
    plt.legend(ncol=1, prop={'size': 8})
    pdf.savefig(bbox_inches='tight', pad_inches=0)
    plt.close()
    
def CutWindow(spectrum, SpecWinds, index, cwl, elem):
    print('Cutting a region of %s at %.5f' % (elem, cwl))
    ###########################################################################
    ### Cutting 10 nm range
    maskCut = (spectrum[0]>cwl-5.) & (spectrum[0]<cwl+5.)# & (spectrum[1]>0.)
    spectrum = np.array([spectrum[0][maskCut], spectrum[1][maskCut], spectrum[2][maskCut]])
    SpWnCut = np.array([SpecWinds[0][maskCut], SpecWinds[index][maskCut]])
    return(spectrum, SpWnCut)

def DisentangleBumps(SpWnCut, cwl):
    spw = SpWnCut[1]
    nonzeros = FindPatch(spw)
    for patch in nonzeros:
        if SpWnCut[0][patch[0]]<=cwl and SpWnCut[0][patch[1]-1]>=cwl:
            spw[:patch[0]] = 0.
            spw[patch[1]:] = 0.
            if len(spw[patch[0]:patch[1]])<3:
                spw[patch[0]-1:patch[1]+1] += 0.001
    return(spw)

def IterateElem(objName, elem, spec, SpecWinds, colNames):
    if not os.path.exists(iSpecFol+"mySample/WindowScripts/Plots/Plots_%s/" % objName):
        os.makedirs(iSpecFol+"mySample/WindowScripts/Plots/Plots_%s/" % objName)
    teff, logg, met, vmic = ReadParams()
    pdf = PdfPages("Plots/Plots_%s/%s_DirtyWinds_%s.pdf" % (objName, objName, elem))
    nonzeros = FindPatch(SpecWinds[colNames[elem]])
    if objName=='J054057':
        vsini = 40.
    else:
        vsini = 0.
    for patch in nonzeros:
        print('Using patch with wavelengths from %.5f nm to %.5f nm' % (SpecWinds[0][patch[0]], SpecWinds[0][patch[1]-1]))
        cwl = (SpecWinds[0][patch[0]] + SpecWinds[0][patch[1]-1])/2.
        spec0, SpWnCut = CutWindow(spec, SpecWinds, colNames[elem], cwl, elem)
        spec1 = SynthSpec(SpWnCut, [teff, logg, met, vmic, vsini, elem, 0.]); spec1 = np.array(tuple(zip(*spec1)))
        SpWnCut[1] = DisentangleBumps(SpWnCut, cwl)
        Plot(elem, objName, spec0, spec1, SpWnCut, pdf)
    pdf.close()
    
if __name__ == '__main__':
    objName = sys.argv[1]
    code = 'all' # 'self' / 'etom' / 'isot' / 'all'
    elems,sigRead,delRead,colNames,dt,SpWnNm,skiprows = Initial(code)
    spec = np.loadtxt(iSpecFol+"mySample/outputAPOGEE/%7s/%7s_CleanSpec.txt" % (objName, objName), delimiter='\t', unpack=True, skiprows=1)
    SpecWinds = ReadWinds(SpWnNm, dt, skiprows, code)
    SpecWinds=np.array([np.array(s) for s in SpecWinds])
    spec = FixWls(spec, SpecWinds)
    for elem in elems:
        if not np.all(SpecWinds[colNames[elem]]==0):
            IterateElem(objName, elem, spec, SpecWinds, colNames)
