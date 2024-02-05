#!/usr/bin/env python
#
#    This file is part of iSpec.
#    Copyright Sergi Blanco-Cuaresma - http://www.blancocuaresma.com/s/
#
#    iSpec is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    iSpec is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with iSpec. If not, see <http://www.gnu.org/licenses/>.
#
import os
import numpy as np
import logging
import multiprocessing
from multiprocessing import Pool
import sys
sys.path.insert(0, '/home/max/CallofPhDuty/iSpec_v20201001')
import ispec

###########################################################################
#--- iSpec directory ------------------------------------------------------
objName = sys.argv[1]
if len(sys.argv) > 2:
    free_param = sys.argv[2]
else:
    free_param = 9
ispec_dir = "/home/max/CallofPhDuty/iSpec_v20201001/"
mySamIn_dir = "mySample/inputAPOGEE/"
mySamOut_dir = "mySample/outputAPOGEE/%7s/" % objName
if not os.path.exists(mySamOut_dir):
    os.makedirs(mySamOut_dir)
sys.path.insert(0, os.path.abspath(ispec_dir))
import ispec

#--- Change LOG level -----------------------------------------------------
#LOG_LEVEL = "warning"
LOG_LEVEL = "info"
logger = logging.getLogger() # root logger, common for all
logger.setLevel(logging.getLevelName(LOG_LEVEL.upper()))
###########################################################################
#--- Average stellar parameter and variables definition
initial_teff = 4500.0; initial_logg = 1.0; initial_MH = -0.5
initial_vmic = 3.00; initial_R = 22500.
star_spectrum = []; star_continuum_model = []; star_continuum_regions= []
estimated_snr = []; segments = []; star_linemasks = []



def SpecToAir(waveobs):
    waveobs /= (1.0 + 5.792105E-2/(238.0185 - (1.E3/waveobs)**2) + 1.67917E-3/(57.362 - (1.E3/waveobs)**2))
    return(waveobs)

def BrokenSpecContinCosmic(star_spectrum, code):
    if objName=='J004441':
        blue_left = 1512.026; blue_right = 1578.998; green_left = 1583.934;
        green_right = 1641.956; red_left = 1645.998; red_right = 1694.430
    elif objName=='J065127':
        blue_left = 1513.698; blue_right = 1580.176; green_left = 1585.139;
        green_right = 1642.659; red_left = 1646.612; red_right = 1694.547
    elif objName=='J050221':
        blue_left = 1511.441; blue_right = 1578.344; green_left = 1583.344;
        green_right = 1641.338; red_left = 1645.338; red_right = 1694.430
    elif objName=='J051728':
        blue_left = 1511.650; blue_right = 1578.563; green_left = 1583.563;
        green_right = 1641.588; red_left = 1645.588; red_right = 1694.430
    else:
        blue_left, blue_right, green_left, green_right, red_left, red_right = ChipLimits(star_spectrum)
    blue_spec = star_spectrum[(star_spectrum['waveobs'] >= blue_left) & (star_spectrum['waveobs'] < blue_right)]
    green_spec = star_spectrum[(star_spectrum['waveobs'] >= green_left) & (star_spectrum['waveobs'] < green_right)]
    red_spec = star_spectrum[(star_spectrum['waveobs'] >= red_left) & (star_spectrum['waveobs'] < red_right)]
    zeroA_spec = star_spectrum[star_spectrum['waveobs'] < blue_left]
    zeroB_spec = star_spectrum[(star_spectrum['waveobs'] >= blue_right) & (star_spectrum['waveobs'] < green_left)]
    zeroC_spec = star_spectrum[(star_spectrum['waveobs'] >= green_right) & (star_spectrum['waveobs'] < red_left)]
    zeroD_spec = star_spectrum[star_spectrum['waveobs'] >= red_right]
    if code=='csmc':
        for k in range(10):
            blue_spec = OutlierFilter(blue_spec, var_limit=0.5)
            green_spec = OutlierFilter(green_spec, var_limit=0.5)
            red_spec = OutlierFilter(red_spec, var_limit=0.5)
    if code=='csmc':
        blue_spec = CosmicFilter(blue_spec, var_limit=0.5)
        green_spec = CosmicFilter(green_spec, var_limit=0.5)
        red_spec = CosmicFilter(red_spec, var_limit=0.5)
    star_spectrum = np.concatenate((zeroA_spec, blue_spec, zeroB_spec, green_spec, zeroC_spec, red_spec, zeroD_spec), axis=0)
    star_continuum_model = ispec.fit_continuum(star_spectrum, fixed_value=1.0, model="Fixed value")
    return(star_spectrum, star_continuum_model)

def ChipLimits(star_spectrum):
    IndOfZeros = np.where(star_spectrum['flux'] == 0.)[0]
    delta = IndOfZeros[1:]-IndOfZeros[:-1]
    delta = np.concatenate((delta, [1]), axis=0)
    mask = np.where(delta>1)[0]
    ind1, ind2, ind3 = mask
    blue_left = star_spectrum['waveobs'][IndOfZeros[ind1]]; blue_right = star_spectrum['waveobs'][IndOfZeros[ind1+1]]
    green_left = star_spectrum['waveobs'][IndOfZeros[ind2]]; green_right = star_spectrum['waveobs'][IndOfZeros[ind2+1]]
    red_left = star_spectrum['waveobs'][IndOfZeros[ind3]]; red_right = star_spectrum['waveobs'][IndOfZeros[ind3+1]]
    return(blue_left, blue_right, green_left, green_right, red_left, red_right)

def OutlierFilter(star_spectrum, var_limit):
    diff = star_spectrum['flux'][1:]-star_spectrum['flux'][:-1]
    diff1 = np.concatenate(([0.], diff), axis=0)
    diff2 = np.concatenate((diff, [0.]), axis=0)
    med = np.nanmedian(star_spectrum['flux'])
    cosmics = (np.abs(diff1)/med>var_limit) | (np.abs(diff2)/med>var_limit)

    cosmics = np.logical_or(cosmics, (star_spectrum['flux'] < 0.))
    #cosmics = np.logical_or(cosmics, (star_spectrum['flux'] > 1.5*med)) # One of the ways to exclude the prominent cosmics
    star_spectrum = star_spectrum[~cosmics]
    return star_spectrum

def ContFitAndNorm(star_spectrum):
    global initial_R
    logging.info("Fitting continuum...")
    median_wave_range=16.*np.median(star_spectrum['waveobs'][1:]-star_spectrum['waveobs'][:-1])
    nknots=None # Automatic: 1 spline every 5 nm
    #--- Checking params --------------------------------------------------
    teff, logg, met, vmic = SteParDerivation()
    teffBin = np.round(teff/500.,0)*500.
    loggBin = np.round(logg*2.,0)/2.
    metBin = np.round(met*2.,0)/2.
    vmicBin = np.round(vmic*2.,0)/2.
    #--- Continuum model --------------------------------------------------
    if teff<5000.:
        # Strategy: Filter first median values and secondly MAXIMUMs in order to find the continuum
        max_wave_range=5.
        logging.info("Using median = %.3f nm and max = %.3f nm" % (median_wave_range, max_wave_range))
        model = "Polynomy" # "Splines"/"Polynomy"
        degree = 1
        order='median+max' # 'median+max', 'max+median', 'median+custom'
        star_continuum_model = ispec.fit_continuum(star_spectrum, \
            from_resolution=initial_R, nknots=nknots, degree=degree, \
            median_wave_range=median_wave_range, \
            max_wave_range=max_wave_range, model=model, order=order, \
           automatic_strong_line_detection=True, \
            strong_line_probability=0.5, use_errors_for_fitting=True)
    else:
        logging.info("Using template with median = %.3f nm" % median_wave_range)
        model = "Template"
        synth_spectrum = ispec.read_spectrum(ispec_dir+"mySample/SynthSpectra/grid/synth/%.0f/%.1f/synth_%.0f_%.1f_%.1f_%.1f.txt" % (teffBin,loggBin,teffBin,loggBin,metBin,vmicBin))
        smoothed_spectrum = ispec.convolve_spectrum(synth_spectrum, 1000., from_resolution=initial_R)
        strong_lines = ispec.read_line_regions(ispec_dir + "/input/linelists/transitions/APOGEE.Thomas/BrackettSeries.txt")
        star_continuum_model = ispec.fit_continuum(star_spectrum, \
            from_resolution=initial_R, ignore=strong_lines, \
            nknots=nknots, median_wave_range=median_wave_range, model=model, \
            template=smoothed_spectrum)
    #--- Normalize --------------------------------------------------------
    star_spectrum = ispec.normalize_spectrum(star_spectrum, star_continuum_model, consider_continuum_errors=False)
    return star_spectrum

def CosmicFilter(star_spectrum, var_limit):
    # Spectrum should be already normalized
    star_continuum_model = ispec.fit_continuum(star_spectrum, fixed_value=1.0, model="Fixed value")
    step = star_spectrum['waveobs'][len(star_spectrum['waveobs'])//2+1] - star_spectrum['waveobs'][len(star_spectrum['waveobs'])//2]
    cosmics = ispec.create_filter_cosmic_rays(star_spectrum,\
            star_continuum_model, resampling_wave_step=step,\
            window_size=15, variation_limit=var_limit)
    cosmics = np.logical_or(cosmics, (star_spectrum['flux'] < 0.))
    #cosmics = np.logical_or(cosmics, (star_spectrum['flux'] > 1.05)) # One of the ways to exclude the prominent cosmics
    star_spectrum = star_spectrum[~cosmics]
    return star_spectrum

def RVCorr(star_spectrum):
    # - Read synthetic template
    teff, logg, met, vmic = SteParDerivation()
    teffBin = np.round(teff/500.,0)*500.
    loggBin = np.round(logg*2.,0)/2.
    metBin = np.round(met*2.,0)/2.
    vmicBin = np.round(vmic*2.,0)/2.
    if met>-0.5 and met<0.:
        metBin = 0.
    template = ispec.read_spectrum(ispec_dir+"mySample/SynthSpectra/grid/synth/%.0f/%.1f/synth_%.0f_%.1f_%.1f_%.1f.txt" % (teffBin,loggBin,teffBin,loggBin,metBin,vmicBin))
    logging.info("Radial velocity determination with template (Teff=%.0fK, logg=%.1f, [M/H]=%.1f, Vmic = %.1fkm/s)..." % (teffBin,loggBin,metBin,vmicBin))
    models, ccf = ispec.cross_correlate_with_template(star_spectrum, template,\
            lower_velocity_limit=-200., upper_velocity_limit=200.,\
            velocity_step=0.1,fourier=True)
    # Plotting CCF (to exclude those contaminated by the shocks)
    vel = [c[0] for c in ccf] # Velocity
    val = [c[1] for c in ccf] # CCF values
    from matplotlib import pyplot as plt
    plt.rcParams["font.family"] = "Times New Roman"
    plt.xlabel('Velocity (km/s)')
    plt.ylabel('Normalised CCF')
    plt.title(objName)
    plt.plot(vel, val)
    plt.grid()
    plt.show()
    # Number of models represent the number of components
    components = len(models)
    # First component:
    if components<1:
        rv = 0.
        rv_err = 0.
    else:
        rv = np.round(models[0].mu(), 2) # km/s
        rv_err = np.round(models[0].emu(), 2) # km/s
    #--- Radial Velocity correction ---------------------------------------
    logging.info("Radial velocity correction dropped (data already corrected)")
    #logging.info("Radial velocity correction... %.2f +/- %.2f" % (rv, rv_err))
    #star_spectrum = ispec.correct_velocity(star_spectrum, rv) #APOGEE spectra are already RV corrected
    return (star_spectrum, rv, rv_err)

def ContReg(star_spectrum, chip):
    global initial_R
    logging.info("Finding continuum regions...")
    sigma = 0.025
    max_continuum_diff = 0.05
    star_continuum_regions = ispec.find_continuum(star_spectrum, \
            resolution=initial_R, max_std_continuum = sigma, \
            continuum_model = 0.95, \
            max_continuum_diff=max_continuum_diff, \
            fixed_wave_step=None)
    ispec.write_continuum_regions(star_continuum_regions, ispec_dir + mySamOut_dir + "continuum_regions_%7s_%4s.txt" % (objName,chip))
    return star_continuum_regions

def SteParDerivation():
    Aname, Ateff, Alogg, Amet, Avmic, Acomm = np.loadtxt(ispec_dir + "Spoiler.txt", delimiter='\t', dtype=np.dtype([('name','U8'), ('teff',np.float64), ('logg',np.float64), ('met',np.float64), ('vmic',np.float64), ('comm','U5')]), skiprows=1, unpack=True)
    index = (Aname==objName)
    teff = float(Ateff[index]); logg = float(Alogg[index])
    met = float(Amet[index]); vmic = float(Avmic[index])
    return teff, logg, met, vmic

def ListCreation():
    #--- Calculate theoretical equivalent widths and depths for a linelist
    teff, logg, met, vmic = SteParDerivation()
    logging.info("Creating a line list for the following atmospheric parameters: %.0f, %.1f, %.2f, %.1f" % (teff, logg, met, vmic))
    alpha = ispec.determine_abundance_enchancements(met)
    model = ispec_dir + "/input/atmospheres/ATLAS9.KuruczODFNEW/"
    modeled_layers_pack = ispec.load_modeled_layers_pack(model)
    atmosphere_layers = ispec.interpolate_atmosphere_layers(modeled_layers_pack, {'teff':teff, 'logg':logg, 'MH':met, 'alpha':alpha})
    isotopes = ispec.read_isotope_data(ispec_dir + "/input/isotopes/SPECTRUM.lst")
    solar_abundances = ispec.read_solar_abundances(ispec_dir + "/input/abundances/Asplund.2009/stdatom.dat")

    #--- APOGEE line list ------------------------------------------------------
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/APOGEE.1500_1700nm/atomic_lines.tsv"
    #--- Masseron's line list --------------------------------------------------
    atomic_linelist_file = ispec_dir + "/input/linelists/transitions/APOGEE.Masseron/excALLibur.tsv"
    
    atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file, wave_base=np.min(star_spectrum['waveobs']), wave_top=np.max(star_spectrum['waveobs']))
    atomic_linelist.sort(order='turbospectrum_species')

    #--- To select only Fe lines -----------------------------------------------
    #Fe_mask = atomic_linelist['element']=='Fe 1'
    #Fe_mask = np.logical_or(Fe_mask, atomic_linelist['element']=='Fe 2')
    #atomic_linelist = atomic_linelist[Fe_mask]

    new_atomic_linelist = ispec.calculate_theoretical_ew_and_depth(atmosphere_layers, \
            teff, logg, met, alpha, \
            atomic_linelist, isotopes, solar_abundances, microturbulence_vel=vmic, \
            verbose=1, gui_queue=None, timeout=900)
    new_atomic_linelist = new_atomic_linelist[new_atomic_linelist['theoretical_ew']>0.01] #0.01 #1.
    new_atomic_linelist.sort(order='wave_A')

    #--- APOGEE line list ------------------------------------------------------
    #ispec.write_atomic_linelist(new_atomic_linelist, linelist_filename=ispec_dir+mySamOut_dir+objName+"_LineList_APOGEE.txt")
    #--- Masseron's line list --------------------------------------------------
    ispec.write_atomic_linelist(new_atomic_linelist, linelist_filename=ispec_dir+mySamOut_dir+objName+"_LineList.txt")
    #--- Literature line list --------------------------------------------------
    #ispec.write_atomic_linelist(new_atomic_linelist, linelist_filename=ispec_dir+mySamOut_dir+objName+"_LitLineList.txt")

def LineFit(star_spectrum, star_continuum_model, use_ares):
    #--- Reading required files ------------------------------------------------
    atomic_linelist_file = ispec_dir+mySamOut_dir+objName+"_LineList.txt"
    atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file, wave_base=np.min(star_spectrum['waveobs']), wave_top=np.max(star_spectrum['waveobs']))
    smoothed_spectrum = ispec.convolve_spectrum(star_spectrum, 15000., from_resolution=initial_R)

    # If line regions are already recorded:
    #line_regions = ispec.read_line_regions(ispec_dir + "/mySample/inputAPOGEE/DFCyg_lines.txt")

    # Else - find linemasks:
    line_regions = ispec.find_linemasks(smoothed_spectrum, star_continuum_model,\
                            atomic_linelist=atomic_linelist, \
                            max_atomic_wave_diff = 0.005, \
                            telluric_linelist=None, \
                            vel_telluric=None, \
                            minimum_depth=0.1, maximum_depth=0.5, \
                            smoothed_spectrum=None, \
                            check_derivatives=False, \
                            discard_gaussian=False, discard_voigt=True, \
                            closest_match=False)
    # Discard bad masks
    flux_peak = star_spectrum['flux'][line_regions['peak']]
    flux_base = star_spectrum['flux'][line_regions['base']]
    flux_top = star_spectrum['flux'][line_regions['top']]
    bad_mask = np.logical_or(line_regions['wave_peak'] <= line_regions['wave_base'], line_regions['wave_peak'] >= line_regions['wave_top'])
    bad_mask = np.logical_or(bad_mask, flux_peak >= flux_base)
    bad_mask = np.logical_or(bad_mask, flux_peak >= flux_top)
    bad_mask = np.logical_or(bad_mask, line_regions['element']=='')
    bad_mask = np.logical_or(bad_mask, line_regions['molecule']=='T')
    line_regions = line_regions[~bad_mask]; ID = 'element'
###########################################################################

    line_regions.sort(order='wave_peak')
    ids = [l['element'] for l in line_regions]
    ids_counted = {item:ids.count(item) for item in ids}
    logging.info("LINES READ: " + str(ids_counted))
    logging.info("TOTAL NUMBER: " + str(len(line_regions)))

    ##--- Fit lines -----------------------------------------------------------
    logging.info("Fitting lines...")
    line_regions = ispec.adjust_linemasks(star_spectrum, line_regions, max_margin=0.25)
    linemasks = ispec.fit_lines(line_regions, star_spectrum, star_continuum_model, \
                atomic_linelist = atomic_linelist, \
                max_atomic_wave_diff = 0.005, \
                telluric_linelist = None, \
                smoothed_spectrum = None, \
                check_derivatives = True, \
                vel_telluric = None, discard_gaussian=False, \
                discard_voigt=True, \
                free_mu=True, crossmatch_with_mu=False, closest_match=False)
    line_regions.sort(order='wave_peak'); linemasks.sort(order='wave_peak')
    linemasks = LineFitFilt(line_regions, linemasks, ID)
    LineFitPlot(star_spectrum, linemasks)
    return(linemasks)

def LineFitFilt(line_regions, linemasks, ID):
    #############################################################################    
    f = open(ispec_dir+mySamOut_dir+objName+"_lineregs.txt", "w")
    for l in linemasks:
        f.write('%.4f\t%.4f\t%.4f\t%5s\n' % (l['wave_peak'], l['wave_base'], l['wave_top'], l['element']))
    f.close()
    #############################################################################      
    ispec.write_line_regions(linemasks, ispec_dir+mySamOut_dir+objName+ "_LineFit.txt", extended=True)
    f = open(ispec_dir+mySamOut_dir+objName+"_LineFit_short.txt", "w")
    for l in line_regions:
        f.write('%5s\t%.4f\n' % (l['note'], l['wave_peak']))
    f.close()
    # Discard lines that are not cross matched with the same original element stored in the note. Uncomment the below line if the linelist is not manually cleaned up.
    linemasks = linemasks[linemasks['element'] == line_regions[ID]]
    print("After element=note")
    print(len(linemasks)) #8686
    #############################################################################    
    f = open(ispec_dir+mySamOut_dir+objName+"_linemask_AfterElementNoteComparison", "w")
    for l in linemasks:
        f.write('%.4f\t%.4f\t%.4f\t%5s\n' % (l['wave_peak'], l['wave_base'], l['wave_top'], l['element']))
    f.close()
    #############################################################################      
    # Exclude lines that have not been successfully cross matched with the atomic data
    # because we cannot calculate the chemical abundance (it will crash the corresponding routines)
    linemasks = linemasks[linemasks['wave_nm']!=0.]    
    #############################################################################    
    f = open(ispec_dir+mySamOut_dir+objName+"_linemask_AfterWaveNotZero", "w")
    for l in linemasks:
        f.write('%.4f\t%.4f\t%.4f\t%5s\n' % (l['wave_peak'], l['wave_base'], l['wave_top'], l['element']))
    f.close()
    #############################################################################  
    linemasks = linemasks[linemasks['ew']>5.]#5
    linemasks = linemasks[linemasks['ew']<350.]#250
    linemasks.sort(order='spectrum_moog_species')
    #############################################################################    
    f = open(ispec_dir+mySamOut_dir+objName+"_linemask_AfterEWFilter", "w")
    for l in linemasks:
        f.write('%.4f\t%.4f\t%.4f\t%5s\n' % (l['wave_peak'], l['wave_base'], l['wave_top'], l['element']))
    f.close()
    #############################################################################     
    ispec.write_line_regions(linemasks, ispec_dir+mySamOut_dir+objName+ "_LineFit.txt", extended=True)
    f = open(ispec_dir+mySamOut_dir+objName+"_LineFit_short.txt", "w")
    for l in linemasks:
        f.write('%5s\t%.4f\t%.2f\t%.2f\n' % (l['element'], l['wave_nm'], l['ew'], l['ew_err']))
    f.close()
    #############################################################################    
    f = open(ispec_dir+mySamOut_dir+objName+"_linemask_AfterAreaDetermination", "w")
    for l in linemasks:
        f.write('%.4f\t%.4f\t%.4f\t%5s\n' % (l['wave_peak'], l['wave_base'], l['wave_top'], l['element']))
    f.close()
    #############################################################################  
    print(len(linemasks))
    linemasks = linemasks[linemasks['ew']>0.]
    print(len(linemasks))
    #############################################################################    
    f = open(ispec_dir+mySamOut_dir+objName+"_linemask_AfterAllFilters", "w")
    for l in linemasks:
        f.write('%.4f\t%.4f\t%.4f\t%5s\n' % (l['wave_peak'], l['wave_base'], l['wave_top'], l['element']))
    f.close()
    #############################################################################  
    ids = [l['element'] for l in linemasks]
    ids_counted = {item:ids.count(item) for item in ids}
    logging.info("LINES IDENTIFIED: " + str(ids_counted))
    logging.info("TOTAL NUMBER: " + str(len(linemasks)))
    return(linemasks)

def LineFitPlot(star_spectrum, linemasks):
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams["font.family"] = "Times New Roman"
    pdf = PdfPages(ispec_dir+mySamOut_dir+objName+"_FittedLines.pdf")
    for i in range(len(linemasks)):
        logging.info('PLOTTING LINE ' + str(i+1) + '/' + str(len(linemasks)))
        spec = star_spectrum[linemasks['peak'][i]-100:linemasks['peak'][i]+100]
        gauss = 1.+linemasks['A'][i]*np.exp(-(spec['waveobs']-linemasks['mu'][i])**2/(2.*linemasks['sig'][i]**2))
        step = spec['waveobs'][len(spec['waveobs'])//2+1]-spec['waveobs'][len(spec['waveobs'])//2]
        wave_filter_custom = (spec['waveobs'] >= linemasks['wave_base'][i]) & (spec['waveobs'] <= linemasks['wave_top'][i])
        line = spec[wave_filter_custom]
        lineEndsX = [line['waveobs'][0], line['waveobs'][-1]]
        lineEndsY = [line['flux'][0], line['flux'][-1]]
        cont_a = (line['flux'][-1]-line['flux'][0])/(line['waveobs'][-1]-line['waveobs'][0])
        cont_b = (line['waveobs'][-1]*line['flux'][0]-line['waveobs'][0]*line['flux'][-1])/(line['waveobs'][-1]-line['waveobs'][0])
        continuum_custom = cont_a*line['waveobs'] + cont_b

        fig = plt.figure(figsize=(11,8))
        ax = fig.add_subplot(1, 1, 1)
        from matplotlib import ticker
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:4.1f}"))
        plt.xlim([linemasks['wave_peak'][i]-0.75, linemasks['wave_peak'][i]+0.75])
        plt.ylim([0.4,1.2])
        plt.xlabel('$\lambda$ (nm)')
        plt.title(linemasks['element'][i] + ', $\lambda$ = %3.3f nm, EW = %3.1f mA (%i/%i)' % (linemasks['wave_nm'][i],linemasks['ew'][i],i+1,len(linemasks)))
        ax.plot(spec['waveobs'], spec['flux'], '-k')
        ax.plot(lineEndsX, lineEndsY, 'co', ms=5, zorder=0)
        ax.plot(lineEndsX, lineEndsY, 'c-', lw=3, zorder=0)
        ax.axvline(linemasks['wave_peak'][i], c='red', zorder=1)
        ax.axhline(1., c='gray', ls='--', zorder=0)
        ax.fill_between(line['waveobs'], line['flux'], continuum_custom, color='cyan', zorder=0)
        pdf.savefig(fig, pad_inches=0) # bbox_inches='tight',
        plt.close()
    pdf.close()

def SpecSave(star_spectrum, addon):
    f = open(ispec_dir+mySamOut_dir+objName+addon, "w")
    f.write('waveobs\tflux\terr\n')
    for s in star_spectrum:
        f.write('%.5f\t%.7f\t%.7f\n' % (s['waveobs'], s['flux'], s['err']))
    f.close()

def SpecSynth(pars, code="moog", regime="abund"):
    global initial_R
    Teff, logg, met, vmic, elem, abund = pars
    alpha = ispec.determine_abundance_enchancements(met)
    macroturbulence = ispec.estimate_vmac(Teff, logg, met) # 4.21
    vsini = 0. # 1.60 for the Sun
    limb_darkening_coeff = 0.6

    atomic_linelist_file = ispec_dir+mySamOut_dir+objName+"_LineList.txt" #ispec_dir + "input/linelists/transitions/VALD.300_1100nm/atomic_lines.tsv" #APOGEE.1500_1700nm
    atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file)
    isotope_file = ispec_dir + "/input/isotopes/SPECTRUM.lst"
    isotopes = ispec.read_isotope_data(isotope_file)
    solar_abundances_file = ispec_dir + "/input/abundances/Asplund.2009/stdatom.dat"
    solar_abundances = ispec.read_solar_abundances(solar_abundances_file)
    chemical_elements_file = ispec_dir + "/input/abundances/chemical_elements_symbols.dat"
    chemical_elements = ispec.read_chemical_elements(chemical_elements_file)
    model = ispec_dir + "input/atmospheres/ATLAS9.KuruczODFNEW/"
    modeled_layers_pack = ispec.load_modeled_layers_pack(model)
    regions = None

    if not ispec.valid_atmosphere_target(modeled_layers_pack, {'teff':Teff, 'logg':logg, 'MH':met, 'alpha':alpha}):
        msg = "The specified effective temperature, gravity (log g) and metallicity [M/H] \
                fall out of theatmospheric models."
        print(msg)

    atmosphere_layers = ispec.interpolate_atmosphere_layers(modeled_layers_pack, {'teff':Teff, 'logg':logg, 'MH':met, 'alpha':alpha}, code=code)

    if regime=="grid":
        atomic_linelist_file = ispec_dir + "/input/linelists/transitions/APOGEE.Thomas/excALLibur.tsv"
        atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file)
        Teff = np.round(Teff/500.,0)*500.
        logg = np.round(logg*2.,0)/2.
        met = np.round(met*2.,0)/2.
        vmic = np.round(vmic*2.,0)/2.
        if objName=='J054057':
            Teff=5200.; logg=1.; met=-1.5; vmic=2.; vsini = 40.
        ##--- Scaling all abundances --------------------------------------
        fixed_abundances = ispec.create_free_abundances_structure(['C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'S', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Dy', 'Er', 'Yb', 'Lu', 'Hf', 'W', 'Pb'], chemical_elements, solar_abundances)
        fixed_abundances['Abund'] += met # Scale to metallicity
        ##--- Calculating synthetic spectrum ------------------------------
        logging.info("Creating synthetic spectrum for Teff=%.0fK, logg=%3s, [M/H]=%4s, Vmic=%.1fkm/s" % (Teff, logg, met, vmic))
        synth_spec = ispec.create_spectrum_structure(star_spectrum['waveobs'])
        synth_spec['flux'] = ispec.generate_spectrum(synth_spec['waveobs'], \
            atmosphere_layers, Teff, logg, met, alpha, atomic_linelist, isotopes, solar_abundances, \
            fixed_abundances, microturbulence_vel = vmic, \
            macroturbulence=macroturbulence, vsini=vsini, limb_darkening_coeff=limb_darkening_coeff, \
            R=initial_R, regions=regions, verbose=1, code=code)
        ##--- Save spectrum -----------------------------------------------
        logging.info("Saving synthetic spectrum for Teff=%.0fK, logg=%.1f, [M/H]=%.1f, Vmic=%.1fkm/s" % (Teff, logg, met, vmic))
        dirName = ispec_dir+"mySample/SynthSpectra/grid/"
        if not os.path.exists(dirName):
            os.makedirs(dirName)
        f = open(dirName+"synth/%.0f/%.1f/synth_%.0f_%.1f_%.1f_%.1f.txt" % (Teff,logg,Teff,logg,met,vmic), "w")
        f.write('waveobs\tflux\terr\n')
        for s in synth_spec:
            f.write('%9s\t%9s\t%9s\n' % (np.round(s['waveobs'],5),np.round(s['flux'],7), np.round(s['err'],7)))
        f.close()

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
        dirName = ispec_dir+"mySample/SynthSpectra/abund_%7s_deriv/" % objName
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
        ratio = 2.; isotps = ['13C', '15N', '17O', '18O']
        for iso in isotps:
            if iso=='13C':
                isotopes[10][3]=ratio/(ratio+1.); isotopes[11][3] = 1./(ratio+1.); abund = 8.6 #8.43+met #13C
            if iso=='15N':
                isotopes[12][3]=ratio/(ratio+1.); isotopes[13][3] = 1./(ratio+1.); abund = 7.83+met #15N
            if iso=='17O':
                isotopes[15][3]=(1.-isotopes[16][3])/(ratio+1.); isotopes[14][3] = 1.-isotopes[15][3]-isotopes[16][3]; abund = 8.33 #8.69+met #17O
            if iso=='18O':
                isotopes[16][3]=(1.-isotopes[15][3])/(ratio+1.); isotopes[14][3] = 1.-isotopes[16][3]-isotopes[15][3]; abund = 8.33 #8.69+met #18O
            #print(isotopes[14][3], isotopes[15][3], isotopes[16][3], isotopes[14][3]+isotopes[15][3]+isotopes[16][3])
            ##--- Populating abundances -----------------------------------
            abElem = []; loge = []; elem = iso[-1]
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
            ##--- Calculating synthetic spectrum --------------------------
            logging.info("Creating synthetic spectrum for Teff=%.0fK, logg=%3s, [M/H]=%4s, Vmic=%.1fkm/s, logeps(%s)=%.2f, iso=%s..." % (Teff, logg, met, vmic, elem, abund, iso))
            synth_spec = ispec.create_spectrum_structure(star_spectrum['waveobs'])
            synth_spec['flux'] = ispec.generate_spectrum(synth_spec['waveobs'], \
                atmosphere_layers, Teff, logg, met, alpha, atomic_linelist, \
                isotopes, solar_abundances, fixed_abundances, \
                microturbulence_vel = vmic, macroturbulence=macroturbulence, \
                vsini=vsini, limb_darkening_coeff=limb_darkening_coeff, \
                R=initial_R, regions=regions, verbose=1, code=code)
            synth_spec['flux'][star_spectrum['flux']<0.] = 0.
            ##--- Save spectrum -------------------------------------------
            logging.info("Saving synthetic spectrum for Teff=%.0fK, logg=%.1f, [M/H]=%.1f, Vmic=%.1fkm/s, logeps(%s)=%.2f, iso=%s..." % (Teff, logg, met, vmic, elem, abund, iso))
            dirName = ispec_dir+"mySample/SynthSpectra/abund_%7s_deriv/" % objName
            if not os.path.exists(dirName):
                os.makedirs(dirName)
            f = open(dirName+"synth_%.0f_%.1f_%.1f_%.1f_%s_%.2f_%s.txt" % (Teff, logg, met, vmic, elem, abund, iso), "w")
            f.write('waveobs\tflux\terr\n')
            for s in synth_spec:
                f.write('%.5f\t%.7f\t%.7f\n' % (s['waveobs'], s['flux'], s['err']))
            f.close()

    if regime=="abund":
        ##--- Scaling all abundances --------------------------------------
        fixed_abundances = ispec.create_free_abundances_structure(chemical_elements['symbol'][1:-33].tolist(), chemical_elements, solar_abundances) # from He to Pb
        fixed_abundances['Abund'] += met # Scale to metallicity
        ##--- Populating known abundances ---------------------------------
        iso = None; ratio = 2.
        abElem = ['C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'S', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Ba', 'La', 'Ce', 'Nd'] #DF Cyg['C', 'Na', 'Si', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Fe', 'Co', 'Ni', 'Zn', 'Y', 'Eu'] #SZ Mon['C', 'O', 'Na', 'Mg', 'Al', 'Si', 'S', 'Ca', 'Sc', 'Ti', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'Y', 'La', 'Sm', 'Eu'] #J004441['C', 'O', 'Na', 'Mg', 'Si', 'S', 'Ca', 'Sc', 'Ti', 'Cr', 'Mn', 'Fe', 'Ni', 'Zn', 'Y', 'Zr', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Dy', 'Er', 'Yb', 'Lu', 'Hf', 'W', 'Co', 'Tm', 'Ta']
        #XFe = [-0.07-met, 0.46-met, 0.05-met, 0.44, 0.33, -0.85, 0.37, 0.64, 0.21, -1.06, -0.85, -0.09, -0.18, 0., -0.11, -0.15, -0.03, -1.42, -0.96, -0.31, -0.37] #DF Cyg[0.26, 0.17, 0.11, -0.23, -0.96, 0.06, 0.24, -0.05, 0., 0.17, -0.01, -0.62, -0.73, -0.09] #SZ Mon[0.57, 0.04, 0.44, 0.33, -0.85, 0.37, 0.64, 0.21, -1.06, -0.85, -0.09, -0.18, 0., -0.11, -0.15, -0.03, -1.42, -0.96, -0.31, -0.37] #J004441[1.67, 1.14, 0.95, 0.49, 0.46, 0.29, 0.33, 0.35, 0.39, -0.06, 0.68, 0., 0.21, 0.63, 2.15, 1.97, 2.84, 2.53, 2.7, 2.74, 2.21, 1.93, 2., 2.27, 2.3, 2.63, 2.52, 2.62, 2.72, 2.6, 4.64, 3.56]
        logeSol = [np.round(solar_abundances['Abund'][solar_abundances['code']==chemical_elements['atomic_num'][chemical_elements['symbol']==el]][0]+12.036,2) for el in abElem] #SZ Mon[8.43, 8.69, 6.24, 7.60, 6.45, 7.51, 7.12, 6.34, 3.15, 4.95, 5.64, 5.43, 7.5, 6.22, 4.19, 4.56, 2.21, 1.1, 0.96, 0.52] #J004441[8.43, 8.69, 6.24, 7.60, 7.51, 7.12, 6.34, 3.15, 4.95, 5.64, 5.43, 7.5, 6.22, 4.56, 2.21, 2.58, 1.1, 1.58, 0.72, 1.42, 0.96, 0.52, 1.07, 1.1, 0.92, 0.84, 0.1, 0.85, 0.85, 4.99, 0.1, -0.12]
        XH = [-0.07, 0.46, 0.05, 0.08, -0.38, -1.28, -0.38, 0.09, -0.73, -1.5, -1.10, -0.39, -0.40, -0.37, -0.51, -0.58, -0.52, -0.6, -0.73, -1.51, -1.05, -1.21, -1.20, -1.13] #[x+met for x in XFe]
        loge = [l+x for l,x in zip(logeSol,XH)]
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
        if iso=='13C':
            isotopes[10][3]=ratio/(ratio+1.); isotopes[11][3] = 1./(ratio+1.) #13C
        if iso=='15N':
            isotopes[12][3]=ratio/(ratio+1.); isotopes[13][3] = 1./(ratio+1.) #15N
        if iso=='17O':
            isotopes[15][3]=(1.-isotopes[16][3])/(ratio+1.); isotopes[14][3] = 1.-isotopes[15][3]-isotopes[16][3] #17O
        if iso=='18O':
            isotopes[16][3]=(1.-isotopes[15][3])/(ratio+1.); isotopes[14][3] = 1.-isotopes[16][3]-isotopes[15][3] #18O
        ##--- Calculating synthetic spectrum ------------------------------
        logging.info("Creating synthetic spectrum for Teff=%.0fK, logg=%3s, [M/H]=%4s, Vmic=%.1fkm/s, logeps(%s)=%.2f..." % (Teff, logg, met, vmic, elem, abund))
        synth_spec = ispec.create_spectrum_structure(star_spectrum['waveobs'])
        synth_spec['flux'] = ispec.generate_spectrum(synth_spec['waveobs'], \
                atmosphere_layers, Teff, logg, met, alpha, atomic_linelist, \
                isotopes, solar_abundances, fixed_abundances, \
                microturbulence_vel = vmic, macroturbulence=macroturbulence, \
                vsini=vsini, limb_darkening_coeff=limb_darkening_coeff, \
                R=initial_R, regions=regions, verbose=1, code=code)
        ##--- Save spectrum -----------------------------------------------
        logging.info("Saving synthetic spectrum for Teff=%.0fK, logg=%.1f, [M/H]=%.1f, Vmic=%.1fkm/s, logeps(%s)=%.2f..." % (Teff, logg, met, vmic, elem, abund))
        dirName = ispec_dir+"mySample/SynthSpectra/abund_%7s/" % objName
        if not os.path.exists(dirName):
            os.makedirs(dirName)
        f = open(dirName+"synth_%.0f_%.1f_%.1f_%.1f_%s_%.2f.txt" % (Teff, logg, met, vmic, elem, abund), "w") # ...solar.txt
        f.write('waveobs\tflux\terr\n')
        for s in synth_spec:
            f.write('%.5f\t%.7f\t%.7f\n' % (s['waveobs'],s['flux'],s['err']))
        f.close()

def PerformSynth(regime):
    t, l, m, v = SteParDerivation()
    Teff = [t] #[3500., 4000., 4500., 5000., 5500., 6000., 6500., 7000., 7500., 8000., 8500., 9000.]
    logg = [l] #[0., 0.5, 1., 1.5, 2., 2.5]
    met = [m] #[-2.5, -2., -1.5, -1., -0.5, 0.]
    vmic = [v] #[1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.]
    elem = ['Fe'] #['He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ge', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cs', 'Ba', 'Ce', 'Nd', 'Yb'] #['He', 'Be', 'B', 'F', 'Ne', 'P', 'Cl', 'Ar', 'Ge', 'Rb', 'Sr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cs', 'Ba'] #['C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'S', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Dy', 'Er', 'Yb', 'Lu', 'Hf', 'W', 'Pb'] Maybe: Yb
    #abund = [7.1] #np.arange(7., 9.01, 0.1).tolist(); abund.append(0.)
    solar_abundances_file = ispec_dir + "/input/abundances/Asplund.2009/stdatom.dat"
    solar_abundances = ispec.read_solar_abundances(solar_abundances_file)
    chemical_elements_file = ispec_dir + "/input/abundances/chemical_elements_symbols.dat"
    chemical_elements = ispec.read_chemical_elements(chemical_elements_file)
    modelList = []
    for t in Teff:
        for l in logg:
            for m in met:
                for v in vmic:
                    for e in elem:
                        ID = chemical_elements['atomic_num'][chemical_elements['symbol']==e]
                        ab = solar_abundances['Abund'][solar_abundances['code']==ID]
                        abund = [ab+12.036+m] #[0., ab+12.036+m, ab+12.136+m, ab+13.036+m, ab+14.036+m, ab+15.036+m] #np.arange(7., 9.2, 0.5).tolist() #[7.1, 7.2, 7.3, 7.4, 7.6, 7.7, 7.8, 7.9]
                        for a in abund:
                            modelList.append([t, l, m, v, e, a])
    for model in modelList:
        SpecSynth(model, code="turbospectrum", regime=regime)

def DefineVisit(objName):
    if objName=='J065127':
        visit = '_visit2'
    elif objName=='J194853':
        visit = '_visit1'
    else:
        visit = ''
    return(visit)

if __name__ == '__main__':
    visit = DefineVisit(objName) # APOGEE_visit2 for SZ Mon, APOGEE_visit1 for DF Cyg (see Mohorian et al. (2024))
    star_spectrum = ispec.read_spectrum(ispec_dir+mySamIn_dir+objName+visit+".txt")
    star_spectrum['flux'][star_spectrum['flux']<0.] = 0.
    star_spectrum['waveobs'] = SpecToAir(star_spectrum['waveobs'])
    star_spectrum, star_continuum_model = BrokenSpecContinCosmic(star_spectrum, code='csmc')
    star_spectrum, rv, rv_err = RVCorr(star_spectrum) # To verify the RV correctoin of APOGEE
    SpecSave(star_spectrum, "_CleanSpec"+visit+".txt")
    pass
