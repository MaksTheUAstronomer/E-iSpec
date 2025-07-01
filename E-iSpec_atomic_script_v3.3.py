#!/usr/bin/env python3
import os
import sys
import ispec
import numpy as np
import logging
import multiprocessing
from multiprocessing import Pool

#--- Paths definitions ---------------------------------------------------------
objName = sys.argv[1]
ispec_dir = os.path.dirname(os.path.realpath(__file__)) + "/"
mySamIn_dir = "mySampleAll/input/%s/" % objName
mySamOut_dir = "mySampleAll/output/%s/" % objName
if not os.path.exists(mySamOut_dir):
    os.makedirs(mySamOut_dir)
sys.path.insert(0, os.path.abspath(ispec_dir))
#--- Definition of initial stellar parameters and other global variables -------
Aname, Ateff, Alogg, Amet, Avmic, Acomm = np.loadtxt(ispec_dir + "SpoilerAll.txt", 
          delimiter='\t', dtype=np.dtype([('name','<U10'), ('teff',np.float64), 
          ('logg',np.float64), ('met',np.float64), ('vmic',np.float64), 
          ('comm','U5')]), skiprows=1, unpack=True)
index = (Aname==objName)
if np.any(index):
    initial_teff = float(Ateff[index][0]); initial_logg = float(Alogg[index][0])
    initial_MH = float(Amet[index][0]); initial_vmic = float(Avmic[index][0])
else:
    initial_teff = 6000.0; initial_logg = 1.0; initial_MH = -2.; initial_vmic = 3.5
initial_R = 57000.; initial_alpha = 0.4
star_spectrum = []; star_continuum_model = []; star_continuum_regions= []
estimated_snr = []; segments = []; star_linemasks = []
master_list_targets = [objName]
#master_list_targets = ["EPLyr_00508457", "EPLyr_00911320", "EPLyr_01030997"] 
#master_list_targets = ["SSGem", "V382Aur", "CCLyr", "RSct", "AUVul", "BD+394926", "J052204", "J053254"]
#--- Change LOG level ----------------------------------------------------------
#LOG_LEVEL = "warning"
LOG_LEVEL = "info"
logger = logging.getLogger() # root logger, common for all
logger.setLevel(logging.getLevelName(LOG_LEVEL.upper()))



def ContFitAndNorm(star_spectrum):
    """
    Fit the continuum of a stellar spectrum and normalize it accordingly.

    This function uses a spline-based approach to model the continuum of an observed 
    stellar spectrum, then normalizes the spectrum by dividing it by the fitted 
    continuum. The continuum is determined using a combination of median filtering and 
    maximum filtering across custom wavelength intervals. The process includes strong 
    line detection and uses the observational errors during the fitting.

    Parameters
    ----------
    star_spectrum : dict
        A dictionary-like object representing the observed stellar spectrum, 
        formatted for use with iSpec. It must contain the following keys: 'waveobs' 
        (in nm), 'flux' and 'error'.

    Returns
    -------
    dict
        The input 'star_spectrum' dictionary updated with normalized values of flux 
        and flux errors.

    Notes
    -----
    - The continuum fitting uses a cubic spline model by default ('model = "Splines"').
    - The degree of the spline is set to 3.
    - The resolution of the input spectrum is taken from the global variable 'initial_R'.
    - The median and maximum filtering window sizes are set to 0.05 nm and 0.25 nm respectively.
    - The number of spline knots ('nknots') is set to 'None', which lets iSpec choose 
      an automatic spacing (roughly one spline every 5 nm).
    - Strong absorption lines are automatically detected and downweighted in the fit.
    - The continuum model is used to normalize the spectrum, but errors in the 
      continuum are not propagated to the normalized result.

    Logging
    -------
    - Logs the fitting process and median wavelength step size via the 'logging' module.

    Global Variables
    ----------------
    initial_R : float
        The resolving power of the input spectrum. This variable must be defined 
        in the global namespace before calling the function.

    Dependencies
    ------------
    - Requires the 'ispec' package.
    - Requires 'numpy' as 'np'.
    - Uses the 'logging' module for status messages.

    Examples
    --------
    >>> normalized_spectrum = ContFitAndNorm(raw_spectrum)
    >>> plt.plot(normalized_spectrum['waveobs'], normalized_spectrum['flux_normalized'])
    """
    global initial_R
    logging.info("Fitting continuum...")
    model = "Splines" # "Polynomy"
    degree = 3
    from_resolution = initial_R
    
    # Strategy to find the continuum: filter median then maximum values
    order='median+max'
    logging.info("Median wavelength step: %.5f nm" % 
                np.median(star_spectrum['waveobs'][1:]-star_spectrum['waveobs'][:-1]))
    median_wave_range=0.05
    max_wave_range=0.25
    nknots = None # Automatic: 1 spline every 5 nm
    
    star_continuum_model = ispec.fit_continuum(star_spectrum, from_resolution=initial_R, \
                                nknots=nknots, degree=degree, \
                                median_wave_range=median_wave_range, \
                                max_wave_range=max_wave_range, \
                                model=model, order=order, \
                                automatic_strong_line_detection=True, \
                                strong_line_probability=0.5, \
                                use_errors_for_fitting=True)
    #--- Normalise -------------------------------------------------------------
    star_spectrum = ispec.normalize_spectrum(star_spectrum, star_continuum_model, 
                                consider_continuum_errors=False)
    return(star_spectrum)
    
def RVCorr(star_spectrum):
    """
    Perform radial velocity determination and correction on a stellar spectrum.

    This function:
        1. Convolves the input stellar spectrum to a specified resolving power.
        2. Filters the convolved spectrum to a specific wavelength range (450–650 nm).
        3. Reads a synthetic template spectrum (default: Arcturus).
        4. Cross-correlates the filtered stellar spectrum with the template using Fourier-based methods to estimate the radial velocity.
        5. Plots and saves the cross-correlation function (CCF) for diagnostic purposes.
        6. Applies the radial velocity correction to the input spectrum.

    Parameters
    ----------
    star_spectrum : list of dict
        A stellar spectrum formatted for use with iSpec. Each entry must contain the keys:
        'waveobs' (wavelength in nm), 'flux' (normalized flux), and 'err' (flux uncertainty).

    Returns
    -------
    tuple
        A 3-element tuple containing:
        - corrected_spectrum : list of dict
            The input spectrum corrected for the measured radial velocity.
        - rv : float
            The radial velocity (in km/s), derived from the peak of the cross-correlation.
        - rv_err : float
            The uncertainty in the derived radial velocity (in km/s).

    Notes
    -----
    - The spectrum is smoothed to a resolving power of R = 50,000 before cross-correlation.
    - The wavelength range for cross-correlation is restricted to 450–650 nm to avoid noisy regions.
    - The cross-correlation is done against an Arcturus template from the iSpec templates directory.
    - The resulting CCF is plotted and saved as a PDF named after the global 'objName' variable.
    - Global variables used within this function (e.g., 'objName', 'initial_R', 'ispec_dir', 'mySamOut_dir')
      must be defined in the surrounding code.

    Raises
    ------
    FileNotFoundError
        If the template spectrum file cannot be found.
    ValueError
        If the input spectrum format is incompatible or missing required fields.
    """
    #--- Radial Velocity determination with template ---------------------------
    logging.info("Radial velocity determination with mask...")
    # - Read synthetic template
    smooth_spectrum = ispec.convolve_spectrum(star_spectrum, 50000., from_resolution=initial_R)
    #f = open(ispec_dir+mySamOut_dir+objName+"_degraded.txt", "w")
    #f.write('waveobs\tflux\terr\n')
    #for s in star_spectrum:
    #    f.write('%9s\t%9s\t%9s\n' % (np.round(s['waveobs'],5),np.round(s['flux'],7), np.round(s['err'],7)))
    #f.close()
    wfilter = ispec.create_wavelength_filter(smooth_spectrum, wave_base=450.0, wave_top=650.0)
    smooth_spectrum = smooth_spectrum[wfilter]
    template = ispec.read_spectrum(ispec_dir + "input/spectra/templates/Atlas.Arcturus.372_926nm/template.txt.gz")
    #template = ispec.read_spectrum(ispec_dir + "input/spectra/templates/Atlas.Sun.372_926nm/template.txt.gz")
    logging.info("Radial velocity determination with Arcturus template...")
    models, ccf = ispec.cross_correlate_with_template(smooth_spectrum, template,\
            lower_velocity_limit=-120., upper_velocity_limit=120.,\
            velocity_step=0.5,fourier=True)
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
    plt.savefig(ispec_dir+mySamOut_dir+objName+"_RV.pdf")
    #plt.show()
    # Number of models represent the number of components
    components = len(models)
    # First component:
    rv = np.round(models[0].mu(), 2) # km/s
    rv_err = np.round(models[0].emu(), 2) # km/s
    
    #--- Radial Velocity correction --------------------------------------------
    #logging.info("Radial velocity correction... %.2f +/- %.2f" % (rv, rv_err))
    #star_spectrum = ispec.correct_velocity(star_spectrum, rv)
    return(star_spectrum, rv, rv_err)
    
def CosmicFilter(spec, star_continuum_model):
    """
    Filters out cosmic ray-affected pixels from a normalised stellar spectrum.

    This function identifies and removes cosmic ray hits in a normalised spectrum
    by comparing the observed flux to a provided stellar continuum model. It uses
    a sliding window approach to detect sharp flux variations inconsistent with the
    expected spectral shape, and filters out the affected data points.

    Parameters
    ----------
    spec : astropy Table or structured NumPy array
        The input spectrum, which must be already normalised to the continuum.
        It must contain at least the column ''waveobs'' (wavelengths in Å) and
        corresponding flux values (typically under a column like ''flux'').

    star_continuum_model : array-like
        The continuum model values corresponding to the spectrum, used as the
        reference to detect anomalous flux variations caused by cosmic rays.

    Returns
    -------
    spec : same type as input
        A filtered copy of the input spectrum with rows affected by cosmic rays
        removed.

    Notes
    -----
    - The wavelength sampling step is estimated near the center of the spectrum
      and assumed to be representative of the entire dataset.
    - Internally uses 'ispec.create_filter_cosmic_rays' with a window size of 15
      and a variation limit of 0.5 to detect cosmic rays.
    - Assumes the input spectrum is already normalised; unnormalised input may lead
      to incorrect filtering.

    See Also
    --------
    ispec.create_filter_cosmic_rays : The iSpec function used for detecting cosmic rays.

    Examples
    --------
    >>> filtered_spec = CosmicFilter(spectrum_table, continuum_model)
    >>> plt.plot(filtered_spec['waveobs'], filtered_spec['flux'])

    """
    # Spectrum should already be normalised
    step = spec['waveobs'][len(spec)//2+1] - spec['waveobs'][len(spec)//2]
    cosmics = ispec.create_filter_cosmic_rays(spec,star_continuum_model,\
              resampling_wave_step=step, window_size=15, \
              variation_limit=0.5)
    spec = spec[~cosmics]
    return(spec)

def SNRErrCalc(star_spectrum):
    """
    Estimate the signal-to-noise ratio (SNR) of a stellar spectrum in selected wavelength bands.

    This function computes the SNR in three specific spectral regions ("blue", "green", and "red") 
    from the flux values of a given stellar spectrum. Prior to calculation, all negative flux 
    values are set to zero to avoid unphysical SNR values. The SNR is estimated using a moving 
    window method with 'ispec.estimate_snr'.

    Parameters
    ----------
    star_spectrum : pandas.DataFrame
        A DataFrame containing the stellar spectrum with at least two columns:
        - 'waveobs': Observed wavelength values (in the same units as the defined band ranges).
        - 'flux': Observed flux values corresponding to the wavelengths.

    Returns
    -------
    list of float
        A list of three SNR values corresponding to the following wavelength bands:
        - "blue"  : 662.0 – 662.4 nm
        - "green" : 663.9 – 664.2 nm
        - "red"   : 675.8 – 676.4 nm

    Notes
    -----
    - This function uses 'ispec.estimate_snr(flux, num_points=3)' for SNR estimation.
    - If any band has insufficient flux points, the resulting SNR may be less reliable or throw an error,
      depending on how 'ispec.estimate_snr' handles short arrays.
    - Negative fluxes are physically implausible in this context and are set to zero before computation.

    Example
    -------
    >>> import pandas as pd
    >>> spectrum = pd.DataFrame({
    ...     'waveobs': [661.9, 662.1, 662.3, 664.0, 664.1, 675.9, 676.1],
    ...     'flux': [10, 12, 11, 13, 15, 9, 10]
    ... })
    >>> snr_values = SNRErrCalc(spectrum)
    >>> print(snr_values)
    [<SNR_blue>, <SNR_green>, <SNR_red>]
    """
    #--- Estimate SNR from flux and errors from SNR ----------------------------
    logging.info("Estimating SNR from fluxes (negative flux values will be set to zero)...")
    star_spectrum['flux'] = np.where(star_spectrum['flux'] < 0.0, 0.0, star_spectrum['flux'])
    ranges = {
        "blue": (662.0, 662.4),
        "green": (663.9, 664.2),
        "red": (675.8, 676.4)
    }
    snr = []
    for key, (low, high) in ranges.items():
        band_spec = star_spectrum[(star_spectrum['waveobs'] >= low) & (star_spectrum['waveobs'] <= high)]
        band_snr = ispec.estimate_snr(band_spec['flux'], num_points=3)
        snr.append(band_snr)
    return(snr)
    
def ListCreation(wave_base, wave_top, model_atmospheres):
    """
    Generate a theoretical atomic line list with computed equivalent widths and depths
    for a specified wavelength range and model atmosphere grid.

    This function loads the appropriate model atmosphere layers based on global stellar 
    atmospheric parameters ('initial_teff', 'initial_logg', 'initial_MH', 'initial_vmic', 
    and 'initial_alpha'), then computes theoretical equivalent widths and line depths 
    for atomic transitions within the given wavelength range. The resulting line list 
    is written to disk.

    Parameters
    ----------
    wave_base : float
        The lower bound of the wavelength range (in nm) for which to calculate atomic lines.
    
    wave_top : float
        The upper bound of the wavelength range (in nm) for which to calculate atomic lines.
    
    model_atmospheres : str
        Path to the modeled atmosphere grid file to be loaded and interpolated.

    Global Variables Used
    ---------------------
    initial_teff : float
        Effective temperature (in K) of the stellar atmosphere model.
    
    initial_logg : float
        Logarithm of the surface gravity (in cm/s²) of the stellar atmosphere model.
    
    initial_MH : float
        Metallicity ([M/H]) of the stellar atmosphere model.
    
    initial_vmic : float
        Microturbulence velocity (in km/s) of the stellar atmosphere model.
    
    initial_alpha : float
        Alpha-element enhancement ([α/Fe]) of the stellar atmosphere model.
    
    objName : str
        Identifier for the target object, used in the output filename.
    
    ispec_dir : str
        Path to the base iSpec directory where inputs are located.
    
    mySamOut_dir : str
        Subdirectory under 'ispec_dir' where output files are written.

    Outputs
    -------
    None
        The function writes the resulting atomic line list to a file named:
        "{ispec_dir}/{mySamOut_dir}/{objName}_LineList.txt"

    Notes
    -----
    - Uses iSpec functions to load isotope data, chemical element information, and 
      solar abundances.
    - Filters the atomic linelist to the given wavelength range before calculations.
    - The function currently writes all computed lines regardless of theoretical EW,
      but it includes commented code to filter out lines with EW ≤ 5 mÅ.
    - Requires iSpec to be properly installed and 'ispec' to be imported and configured.
    - Logging is used to print the adopted stellar parameters.
    """
    #--- Calculate theoretical equivalent widths and depths for a linelist -----
    global initial_teff, initial_logg, initial_MH, initial_vmic, initial_alpha
    logging.info("CREATING A LINE LIST FOR THE FOLLOWING AP: %4.0f, %1.1f, %1.2f, %1.1f" % 
                (initial_teff, initial_logg, initial_MH, initial_vmic))
    #initial_alpha = ispec.determine_abundance_enchancements(initial_MH)
    modeled_layers_pack = ispec.load_modeled_layers_pack(model_atmospheres)
    atmosphere_layers = ispec.interpolate_atmosphere_layers(modeled_layers_pack, \
                {'teff':initial_teff, 'logg':initial_logg, 'MH':initial_MH, 'alpha':initial_alpha})
    isotopes = ispec.read_isotope_data(ispec_dir + "/input/isotopes/SPECTRUM.lst")
    atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.300_1100nm/atomic_lines.tsv"
    atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file, wave_base=wave_base, wave_top=wave_top)
    chemical_elements = ispec.read_chemical_elements(ispec_dir + "/input/abundances/chemical_elements_symbols.dat")
    solar_abundances = ispec.read_solar_abundances(ispec_dir + "/input/abundances/Asplund.2021/stdatom.dat")
    #solar_abundances['Abund'][solar_abundances['code']==
    #            chemical_elements['atomic_num'][chemical_elements['symbol']=='C']] += 1.
    #CNO_mask = atomic_linelist['element']=='C 1'
    #CNO_mask = np.logical_or(CNO_mask, atomic_linelist['element']=='N 1')
    #CNO_mask = np.logical_or(CNO_mask, atomic_linelist['element']=='O 1')
    #atomic_linelist = atomic_linelist[CNO_mask]
    new_atomic_linelist = ispec.calculate_theoretical_ew_and_depth(atmosphere_layers, \
            initial_teff, initial_logg, initial_MH+1., initial_alpha, \
            atomic_linelist, isotopes, solar_abundances, microturbulence_vel=initial_vmic, \
            verbose=1, gui_queue=None, timeout=900)
    #new_atomic_linelist = new_atomic_linelist[new_atomic_linelist['theoretical_ew']>5.]
    ispec.write_atomic_linelist(new_atomic_linelist, linelist_filename=ispec_dir + mySamOut_dir + \
                objName + "_LineList.txt")
    
def LineFit(star_spectrum, star_continuum_model, model_atmospheres, rv, rv_err, FeCNO=1, mode='seek'):
    """
    Identify and fit absorption lines in a stellar spectrum using atomic line data and a continuum model.

    This function automates the process of detecting and fitting spectral lines in a given stellar spectrum,
    optionally focusing on specific elements (e.g., Fe, C, N, O) and applying a mode-dependent strategy for
    line region identification. It returns a table of fitted line parameters for use in further spectroscopic
    analysis (e.g., abundance derivation).

    Parameters
    ----------
    star_spectrum : np.ndarray or structured array
        Observed stellar spectrum with fields such as 'waveobs' and 'flux', formatted for use with iSpec.

    star_continuum_model : np.ndarray or structured array
        The continuum-normalized model spectrum used as a reference for identifying absorption features.

    model_atmospheres : dict or other
        Model atmosphere parameters to be passed to the plotting or downstream analysis functions.

    rv : float
        Radial velocity of the star in km/s (used to approximate telluric velocity offset).

    rv_err : float
        Uncertainty on the radial velocity measurement in km/s.

    FeCNO : int, optional, default=1
        If 1, select only lines corresponding to Fe I, Fe II, C I, N I, O I, S I, and Zn I.
        If 0, exclude these elements from the fitting process.

    mode : str, optional, default='seek'
        Determines the mode of line region selection:
        - 'seek': Automatically detect line regions from the spectrum.
        - 'tweak' or 'pick': Load pre-defined line regions from file.

    Returns
    -------
    linemasks : np.ndarray
        Structured array of fitted line parameters, including wavelength, equivalent width, depth, and other
        diagnostic quantities. The results are filtered for physical plausibility and plotted.

    Notes
    -----
    - Uses the 'iSpec' framework for spectrum processing and line fitting.
    - If 'mode' is 'seek', the function smooths the spectrum, identifies candidate absorption lines,
      filters out poor detections and non-target elements, and fits them.
    - If 'mode' is 'tweak' or 'pick', line regions are read from an existing file instead.
    - Line fits are performed with optional convolution and Gaussian/Voigt profile fitting disabled.
    - A helper function 'LineFitFilt()' is used to apply additional post-fit filtering.
    - Fitted lines are visualized with 'LineFitPlot()'.

    Example
    -------
    >>> linemasks = LineFit(star_spectrum, continuum_model, model_atmos, rv=15.2, rv_err=0.5, FeCNO=1, mode='seek')
    >>> print(linemasks['wave_peak'], linemasks['ew'])

    See Also
    --------
    ispec.read_atomic_linelist
    ispec.find_linemasks
    ispec.fit_lines
    LineFitFilt
    LineFitPlot
    """
    #--- Reading required files ------------------------------------------------
    atomic_linelist_file = ispec_dir+mySamOut_dir+objName+"_LineList.txt"
    atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file, wave_base=np.min(star_spectrum['waveobs']), \
                wave_top=np.max(star_spectrum['waveobs']))
    #atomic_linelist = atomic_linelist[atomic_linelist['theoretical_depth']>0.]
    
    telluric_linelist_file = ispec_dir + "/input/linelists/CCF/Synth.Tellurics.500_1100nm/mask.lst"
    telluric_linelist = ispec.read_telluric_linelist(telluric_linelist_file, minimum_depth=0.01)
    models, ccf = ispec.cross_correlate_with_mask(star_spectrum, telluric_linelist, \
            lower_velocity_limit=-300., upper_velocity_limit=300., \
            velocity_step=0.5, mask_depth=0.01, \
            fourier = False, only_one_peak = True)
    vel_telluric = np.round(models[0].mu(), 2) # km/s
    vel_telluric_err = np.round(models[0].emu(), 2) # km/s
    
    # Line regions
    if mode=='seek':
        smooth_R = initial_R/1.25; ID = 'element'
        smooth_spectrum = ispec.convolve_spectrum(star_spectrum, smooth_R, from_resolution=initial_R)
        #####--- Find linemasks ------------------------------------------------
        line_regions = ispec.find_linemasks(star_spectrum, star_continuum_model,\
                                atomic_linelist=atomic_linelist, \
                                max_atomic_wave_diff = 0.005, \
                                telluric_linelist=None, \
                                vel_telluric=vel_telluric, \
                                minimum_depth=0.1, maximum_depth=0.5, \
                                smoothed_spectrum=smooth_spectrum, \
                                check_derivatives=False, \
                                discard_gaussian=False, discard_voigt=True, \
                                closest_match=False)
        # Discard bad masks
        flux_peak = smooth_spectrum['flux'][line_regions['peak']]
        flux_base = smooth_spectrum['flux'][line_regions['base']]
        flux_top = smooth_spectrum['flux'][line_regions['top']]
        bad_mask = np.logical_or(line_regions['wave_peak'] <= line_regions['wave_base'], 
                    line_regions['wave_peak'] >= line_regions['wave_top'])
        bad_mask = np.logical_or(bad_mask, flux_peak >= flux_base)
        bad_mask = np.logical_or(bad_mask, flux_peak >= flux_top)
        bad_mask = np.logical_or(bad_mask, line_regions['element']=='')
        bad_mask = np.logical_or(bad_mask, line_regions['molecule']!='F')
        line_regions = line_regions[~bad_mask]
        # Leave only Fe 1 and Fe 2 lines (and others if needed)
        FeCNO_mask = np.isin(line_regions['element'], ['Fe 1', 'Fe 2', 'C 1', 'N 1', 'O 1', 'S 1', 'Zn 1'])
        if FeCNO==1:
            line_regions = line_regions[FeCNO_mask]
        else:
            line_regions = line_regions[~FeCNO_mask]
        line_regions = line_regions[line_regions['theoretical_ew'] > 0.]
    
    if mode=='tweak' or mode=='pick':
        line_regions = ispec.read_line_regions(ispec_dir + mySamIn_dir + objName + "_line_regs.txt"); ID = 'note'

    ids = [l[ID] for l in line_regions]
    ids_counted = {item:ids.count(item) for item in ids}
    logging.info("LINES READ: " + str(ids_counted))
    logging.info("TOTAL NUMBER: " + str(len(line_regions)))
    #--- Fit lines -------------------------------------------------------------
    logging.info("Fitting lines...")
    #line_regions = ispec.adjust_linemasks(star_spectrum, line_regions, max_margin=0.5)
    linemasks = ispec.fit_lines(line_regions, star_spectrum, star_continuum_model, \
                atomic_linelist = atomic_linelist, \
                max_atomic_wave_diff = 0.005, \
                telluric_linelist = None, \
                smoothed_spectrum = None, \
                check_derivatives = False, \
                vel_telluric = None, discard_gaussian=False, \
                discard_voigt=True, \
                free_mu=True, crossmatch_with_mu=False, closest_match=False)
    line_regions.sort(order='wave_peak'); linemasks.sort(order='wave_peak')
    linemasks = LineFitFilt(line_regions, linemasks, ID)
    LineFitPlot(star_spectrum, linemasks, model_atmospheres, mode)
    return(linemasks)

def LineFitFilt(line_regions, linemasks, ID): # A function to plot the output of each step
    """
    Filters and exports spectral line data for further analysis, retaining only well-characterized lines 
    based on consistency, equivalent width (EW), and wavelength criteria.

    This function processes line fitting outputs by:
    - Saving input line masks and regions to disk.
    - Filtering linemasks by element match with line_regions[ID].
    - Excluding lines with zero wavelength or out-of-range equivalent widths.
    - Logging the number and identity of valid lines retained after filtering.
    - Writing intermediate outputs to text files at various steps for diagnostic purposes.

    Parameters
    ----------
    line_regions : structured array or list of dict
        A list or structured array of identified spectral line regions, each element expected to have 
        at least 'note' and 'wave_peak' keys. These are used for cross-matching with linemasks.

    linemasks : structured NumPy array
        Array of line mask entries, each representing a spectral line. Expected fields include:
        - 'wave_peak': float — central wavelength of the line
        - 'wave_base': float — base wavelength
        - 'wave_top': float — top wavelength
        - 'element': str — identifier of the element
        - 'wave_nm': float — wavelength in nm (used for further filtering)
        - 'ew': float — equivalent width
        - 'ew_err': float — error in equivalent width
        - 'spectrum_moog_species': int — identifier used for sorting

    ID : int
        Index to select a specific line region from 'line_regions' for filtering by element match.

    Returns
    -------
    linemasks : structured NumPy array
        Filtered linemasks after applying all quality and consistency filters, ready for abundance 
        determination or further analysis.

    Side Effects
    ------------
    - Creates multiple diagnostic and result output text files in the directory:
        '<ispec_dir>/<mySamOut_dir>/LineFitOutput/'
    - Files include:
        - Raw and filtered line lists
        - Short summaries of lines
        - Fit quality placeholders
        - Logging of final retained line statistics

    Notes
    -----
    - Uses global variables: 'ispec_dir', 'mySamOut_dir', and 'objName'.
    - Intended for use in post-processing steps of iSpec spectral analysis pipelines.
    - Lines with EW <= 5 or EW >= 350 are discarded to avoid overly weak or saturated lines.
    - Logging info includes counts per element and total number of retained lines.

    Example
    -------
    >>> filtered_lines = LineFitFilt(line_regions, linemasks, ID=0)
    >>> print(len(filtered_lines))
    48
    """
    LineFitOut_dir = ispec_dir+mySamOut_dir+'LineFitOutput/'
    if not os.path.exists(LineFitOut_dir):
        os.makedirs(LineFitOut_dir)
    #############################################################################    
    f = open(LineFitOut_dir+objName+"_lineregs.txt", "w")
    for l in linemasks:
        f.write('%.4f\t%.4f\t%.4f\t%s\n' % (l['wave_peak'], l['wave_base'], l['wave_top'], l['element']))
    f.close()
    #############################################################################      
    ispec.write_line_regions(linemasks, LineFitOut_dir+objName+"_LineFit.txt", extended=True)
    f = open(LineFitOut_dir+objName+"_LineFit_short.txt", "w")
    for l in line_regions:
        f.write('%s\t%.4f\n' % (l['note'], l['wave_peak']))
    f.close()
    
    # Discard lines that are not cross matched with the same original element stored in the note. 
    linemasks = linemasks[linemasks['element'] == line_regions[ID]]
    print("After element=note")
    print(len(linemasks))
    #############################################################################    
    f = open(LineFitOut_dir+objName+"_linemask_AfterElementNoteComparison", "w")
    for l in linemasks:
        f.write('%.4f\t%.4f\t%.4f\t%s\n' % (l['wave_peak'], l['wave_base'], l['wave_top'], l['element']))
    f.close()
    #############################################################################      
    # Select lines that have some minimal contribution in pAGBs
#    atomic_linelist = atomic_linelist[atomic_linelist['theoretical_depth'] >= 0.01]
    # Exclude lines that have not been successfully cross matched with the atomic data
    # because we cannot calculate the chemical abundance (it will crash the corresponding routines)
    linemasks = linemasks[linemasks['wave_nm']!=0.]
    #############################################################################    
    f = open(LineFitOut_dir+objName+"_linemask_AfterWaveNotZero", "w")
    for l in linemasks:
        f.write('%.4f\t%.4f\t%.4f\t%s\n' % (l['wave_peak'], l['wave_base'], l['wave_top'], l['element']))
    f.close()
    #############################################################################  
    linemasks = linemasks[linemasks['ew']>5.]#5
    linemasks = linemasks[linemasks['ew']<350.]#250
    linemasks.sort(order='spectrum_moog_species')
    #############################################################################    
    f = open(LineFitOut_dir+objName+"_linemask_AfterEWFilter", "w")
    for l in linemasks:
        f.write('%.4f\t%.4f\t%.4f\t%s\n' % (l['wave_peak'], l['wave_base'], l['wave_top'], l['element']))
    f.close()
    #############################################################################     
    ispec.write_line_regions(linemasks, LineFitOut_dir+objName+ "_LineFit.txt", extended=True)
    f = open(LineFitOut_dir+objName+"_LineFit_short.txt", "w")
    for l in linemasks:
        f.write('%s\t%.4f\t%.2f\t%.2f\n' % (l['element'], l['wave_nm'], l['ew'], l['ew_err']))
    f.close()
    f = open(LineFitOut_dir+objName+"_FitGoodness.txt", "w")
    f.write('lambda (nm)\tArea Ratio (%)\n')
    dev = 0.
    goodLnNum = 0
    f.close()
    #############################################################################    
    f = open(LineFitOut_dir+objName+"_linemask_AfterAreaDetermination", "w")
    for l in linemasks:
        f.write('%.4f\t%.4f\t%.4f\t%s\n' % (l['wave_peak'], l['wave_base'], l['wave_top'], l['element']))
    f.close()
    #############################################################################  
    print(len(linemasks))
    linemasks = linemasks[linemasks['ew']>0.]
    print(len(linemasks))
    #############################################################################    
    f = open(LineFitOut_dir+objName+"_linemask_AfterAllFilters", "w")
    f.write('wave_peak\twave_base\twave_top\tnote\n')
    for l in linemasks:
        f.write('%.4f\t%.4f\t%.4f\t%s\n' % (l['wave_peak'], l['wave_base'], l['wave_top'], l['element']))
    f.close()
    #############################################################################  
    ids = [l['element'] for l in linemasks]
    ids_counted = {item:ids.count(item) for item in ids}
    logging.info("LINES IDENTIFIED: %s" % ids_counted)
    logging.info("TOTAL NUMBER: %i" % len(linemasks))
    return(linemasks)

def LineFitPlot(star_spectrum, linemasks, model_atmospheres, mode):
    """
    Generate diagnostic plots of individual spectral line fits for a given star spectrum and save them to a multi-page PDF.

    This function processes a stellar spectrum and its corresponding line masks to produce visualizations
    of fitted spectral lines. Each plot displays the observed line, a Gaussian fit, annotated nearby lines
    from an atomic linelist, and (optionally) synthetic spectra computed from a model atmosphere at two different
    abundance assumptions. The function outputs these plots to a single PDF file for visual inspection.

    Parameters
    ----------
    star_spectrum : numpy structured array or astropy Table
        Observed stellar spectrum with at least the fields 'waveobs' and 'flux' representing wavelength (in nm)
        and normalized flux.

    linemasks : numpy structured array or astropy Table
        Table of individual line fitting parameters. Each entry should include the following fields:
        - 'wave_peak' : central wavelength of the line (nm)
        - 'A', 'mu', 'sig' : parameters of the Gaussian fit (amplitude, mean, and sigma)
        - 'wave_base', 'wave_top' : wavelength region of interest for the line
        - 'wave_nm' : nominal line center
        - 'element' : chemical element symbol (e.g., 'Fe I')
        - 'ew', 'theoretical_ew' : measured and initial guess equivalent widths (mÅ)

    model_atmospheres : object or structured input
        Data structure used by the 'SynthSpec' function to compute synthetic spectra. Must be compatible with your local 'SynthSpec' implementation.

    mode : str
        Plotting mode. Supported values:
        - ''tweak'': include synthetic spectra for abundance comparison
        - any other value: exclude synthetic spectra

    Behavior
    --------
    - Reads an atomic linelist and solar abundances from disk (via 'ispec').
    - Iterates over each line in 'linemasks' and extracts the corresponding portion of the observed spectrum.
    - Skips lines with too narrow a wavelength window.
    - Plots the observed flux, Gaussian fit, and (if mode is ''tweak'') synthetic spectra at two abundances:
        - a very low baseline abundance (-10.0)
        - an abundance derived from solar + [M/H]
    - Annotates nearby atomic lines (with significant EW contribution) within the plotting region.
    - Marks the line region used in EW calculation.
    - Saves each figure into a multi-page PDF named:
        '"{objName}_FittedLines.pdf"' or '"{objName}_FittedLinesWithSynth.pdf"' depending on mode.

    Notes
    -----
    - Relies on several global variables: 'ispec_dir', 'mySamOut_dir', 'objName', and 'initial_MH'.
    - Requires the external 'ispec' library for reading linelists and abundances.
    - Requires a locally defined function 'SynthSpec(spec, linemask, model_atmospheres, abundance)'.

    Output
    ------
    A PDF file saved to: 'ispec_dir + mySamOut_dir + objName + "_FittedLines[WithSynth].pdf"'
    containing one page per spectral line fit.
    """
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    if mode=='tweak':
        addon = "WithSynth"
    else:
        addon = ""
    pdf = PdfPages(ispec_dir+mySamOut_dir+objName+"_FittedLines"+addon+".pdf")
    atomic_linelist_file = ispec_dir+mySamOut_dir+objName+"_LineList.txt"
    atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file, wave_base=np.min(star_spectrum['waveobs']), \
                wave_top=np.max(star_spectrum['waveobs']))
    for i in range(len(linemasks)):
        logging.info('PLOTTING LINE %i/%i' % (i+1, len(linemasks)))
        waveGap = 0.35 # +/- in nm
        spec = star_spectrum[(star_spectrum['waveobs']>linemasks['wave_peak'][i]-waveGap) & \
                    (star_spectrum['waveobs']<linemasks['wave_peak'][i]+waveGap)]
        gauss = 1.+linemasks['A'][i]*np.exp(-(spec['waveobs']-linemasks['mu'][i])**2/(2.*linemasks['sig'][i]**2))
        if (spec['waveobs'][-1]-spec['waveobs'][0])<waveGap:
            continue
        step = spec['waveobs'][1]-spec['waveobs'][0]
        from_x = linemasks['wave_peak'][i] - 6.*linemasks['sig'][i]
        to_x = linemasks['wave_peak'][i] + 6.*linemasks['sig'][i]
        wave_filter_custom = (spec['waveobs'] >= linemasks['wave_base'][i]) & \
                    (spec['waveobs'] <= linemasks['wave_top'][i])
        specline = spec[wave_filter_custom]
        lineEndsX = [specline['waveobs'][0], specline['waveobs'][-1]]
        lineEndsY = [1., 1.]
        cont_a = 0.; cont_b = 1.
        continuum_custom = cont_a*spec['waveobs'] + cont_b
        
        solar_abundances = ispec.read_solar_abundances(ispec_dir + \
                    "/input/abundances/Asplund.2021/stdatom.dat")
        chemical_elements = ispec.read_chemical_elements(ispec_dir + \
                    "/input/abundances/chemical_elements_symbols.dat")
        ID = chemical_elements['atomic_num'][chemical_elements['symbol']==linemasks['element'][i][:-2]]
        global initial_MH
        abund = solar_abundances['Abund'][solar_abundances['code']==ID]+12.036+initial_MH
        
        fig = plt.figure(figsize=(11,8))
        ax = fig.add_subplot(1,1,1)
        plt.xlim([linemasks['wave_peak'][i]-0.3,linemasks['wave_peak'][i]+0.3])
        plt.ylim([0.2,1.2])
        plt.xlabel('$\\lambda$ (nm)')
        plt.title('%s (%i/%i), EW = %.1f mA, initial guess EW = %.1f mA' % (linemasks['element'][i], \
                    i+1,len(linemasks), linemasks['ew'][i], linemasks['theoretical_ew'][i]))
        ax.plot(spec['waveobs'], spec['flux'], '-k', label='Observed spectrum')
        
        # Add pointers with elements to lines with EW>5 mA
        significant_linelist = atomic_linelist[atomic_linelist['wave_nm']>spec['waveobs'][0]]
        significant_linelist = significant_linelist[significant_linelist['wave_nm']<spec['waveobs'][-1]]
        significant_linelist = significant_linelist[significant_linelist['theoretical_ew']/linemasks['ew'][i]>0.05]
        for line in significant_linelist:
            plt.annotate(line['element'], xy=(line['wave_nm'], 1.02), xytext=(line['wave_nm'], 1.12), rotation=90, 
                        ha='center', fontsize=15, arrowprops=dict(arrowstyle="-", facecolor='black', lw=2))
        
        if mode=='tweak':
            synth0 = SynthSpec(spec, linemasks[i], model_atmospheres, -10.)
            synth = SynthSpec(spec, linemasks[i], model_atmospheres, abund)
            ax.plot(synth['waveobs'], synth['flux'], '--g', label='A(%s)=%.2f' % 
                        (linemasks['element'][i][:-2], abund))
            ax.plot(synth0['waveobs'], synth0['flux'], '-.m', label='A(%s)=-10.00' % 
                        linemasks['element'][i][:-2])
        
        mask4Gauss = (spec['waveobs']>from_x) & (spec['waveobs']<to_x)
        ax.plot(spec['waveobs'][mask4Gauss], gauss[mask4Gauss], '-r', label='Gaussian fit')
        ax.axvline(linemasks['wave_nm'][i], ymin=0., ymax=0.8, c='red', ls=':', zorder=1)
        ax.axhline(1., c='gray', ls=':', zorder=0, label='Continuum')
        ax.fill_between(lineEndsX, y1=0., y2=1., color='none', edgecolor='olive', hatch='\\\\\\', 
                    zorder=0., label='Line region')
        ax.fill_between(spec['waveobs'][mask4Gauss], gauss[mask4Gauss], continuum_custom[mask4Gauss], 
                    color='none', edgecolor='cyan', hatch='///', zorder=0, label='Line area (EW)')
        ax.legend(ncol=2, loc='lower left')
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
        plt.close()
    pdf.close()

def SynthSpec(star_spectrum, regions, model_atmospheres, abund=0., code='turbospectrum'):
    """
    Generate a synthetic spectrum for a given star using atmospheric parameters and a model atmosphere grid.

    This function synthesizes a stellar spectrum over the same wavelength range as the observed input
    spectrum using the iSpec library. It interpolates the model atmosphere, reads relevant linelists and
    abundance files, and optionally adjusts the abundance of a specified element before synthesizing
    the final spectrum.

    Parameters
    ----------
    star_spectrum : dict
        A dictionary containing the observed stellar spectrum. It must include the key 'waveobs'
        with an array of observed wavelengths.

    regions : dict
        A dictionary describing the spectral region and target element. It must include the key 'element',
        which should be the name of the element to be adjusted (e.g., 'Fe I') if 'abund' is specified.

    model_atmospheres : str
        Path to the grid of precomputed model atmospheres to be used for interpolation.

    abund : float, optional
        The absolute abundance (A(X)) to be fixed for the element specified in 'regions['element']'. 
        If 'abund' is 0 (default), the solar-scaled abundance is used.

    code : str, optional
        The radiative transfer code to be used for spectral synthesis. Supported values typically
        include 'turbospectrum', 'spectrum', etc. Default is 'turbospectrum'.

    Returns
    -------
    synth_spectrum : dict
        A dictionary containing the synthesized spectrum. It includes the keys:
        - 'waveobs': array of wavelengths (copied from input),
        - 'flux': array of synthesized fluxes corresponding to the wavelengths.

    Notes
    -----
    - Requires global variables to be defined prior to execution:
        - initial_teff (float): Effective temperature of the star.
        - initial_logg (float): Surface gravity.
        - initial_MH (float): Metallicity [M/H].
        - initial_vmic (float): Microturbulence velocity.
        - initial_R (float): Resolving power of the instrument.
        - initial_alpha (float): Alpha-element enhancement.
    - The 'mySamOut_dir', 'objName', and 'ispec_dir' variables must be defined in the global scope.
    - If a fixed abundance is specified, it is converted to a differential value relative to the
      solar abundance of the same element.
    - The function uses hardcoded limb darkening and vsini parameters (limb_darkening_coeff=0.6, vsini=0).

    Example
    -------
    >>> synth = SynthSpec(obs_spectrum, {'element': 'Fe I'}, 'marcs_grid', abund=7.5)
    >>> plt.plot(synth['waveobs'], synth['flux'])
    """
    global initial_teff, initial_logg, initial_MH, initial_vmic, initial_R, initial_alpha
    macroturbulence = ispec.estimate_vmac(initial_teff, initial_logg, initial_MH)
    max_iterations = 10
    modeled_layers_pack = ispec.load_modeled_layers_pack(model_atmospheres)
    atmosphere_layers = ispec.interpolate_atmosphere_layers(modeled_layers_pack, \
                {'teff':initial_teff, 'logg':initial_logg, 'MH':initial_MH, 'alpha':initial_alpha})
    isotopes = ispec.read_isotope_data(ispec_dir + "/input/isotopes/SPECTRUM.lst")
    solar_abundances = ispec.read_solar_abundances(ispec_dir + "/input/abundances/Asplund.2021/stdatom.dat")
    atomic_linelist_file = ispec_dir+mySamOut_dir+objName+"_LineList.txt"
    atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file, wave_base=np.min(star_spectrum['waveobs']), \
                wave_top=np.max(star_spectrum['waveobs']))
    chemical_elements_file = ispec_dir + "/input/abundances/chemical_elements_symbols.dat"
    chemical_elements = ispec.read_chemical_elements(chemical_elements_file)
    if abund!=0.:
        fixed_abundances = ispec.create_free_abundances_structure([regions['element'][:-2]], \
                    chemical_elements, solar_abundances)
        fixed_abundances['Abund'] = [abund-12.036]
    else:
        fixed_abundances=None
    # Synthesis
    synth_spectrum = ispec.create_spectrum_structure(star_spectrum['waveobs'])
    synth_spectrum['flux'] = ispec.generate_spectrum(synth_spectrum['waveobs'], \
            atmosphere_layers, initial_teff, initial_logg, initial_MH, initial_alpha, 
            atomic_linelist, isotopes, solar_abundances, \
            fixed_abundances=fixed_abundances, microturbulence_vel = initial_vmic, \
            macroturbulence=macroturbulence, vsini=0., limb_darkening_coeff=0.6, \
            R=initial_R, regions=None, verbose=1, code=code)
    return(synth_spectrum)
    
def EWparam(star_spectrum, star_continuum_model, linemasks, model_atmospheres, code='moog', 
            mode='default', nlte=False):
    """
    Estimate stellar atmospheric parameters using an equivalent width (EW)-based method.

    This function filters input spectral line data based on physical criteria and quality control,
    prepares input for spectral modeling using EWs, performs parameter inference via iSpec's
    'model_spectrum_from_ew', and optionally outputs intermediate and final results to files.
    It also updates global initial atmospheric parameters used across different stages of modeling.

    Parameters
    ----------
    star_spectrum : dict
        Dictionary containing observed stellar spectrum, typically with keys like 'waveobs' and 'flux'.
    
    star_continuum_model : array-like
        Array or structure representing the continuum-normalized model of the star’s spectrum.

    linemasks : numpy structured array or pandas DataFrame
        Table of spectral line information, including:
        - 'ewr': log of reduced equivalent width
        - 'lower_state_eV': excitation potential of the lower energy state
        - 'rms': residual of the fit for the line
        - 'fwhm': full width at half maximum of the line
        - 'peak', 'wave_nm', 'wave_peak', 'wave_base', 'wave_top', 'element': other line properties.

    model_atmospheres : str
        Path to the atmospheric model grid compatible with iSpec.

    code : str, optional
        Radiative transfer code to use for synthesis, e.g., 'moog', 'turbospectrum'. Default is 'moog'.

    mode : str, optional
        Reserved for future options controlling the mode of operation. Currently unused. Default is 'default'.

    nlte : bool, optional
        Whether to output results to an NLTE-specific filename. Default is False.

    Returns
    -------
    params : dict
        Dictionary of inferred stellar atmospheric parameters:
        - 'teff': effective temperature (K)
        - 'logg': surface gravity (dex)
        - 'MH'  : metallicity [M/H] (dex)
        - 'vmic': microturbulent velocity (km/s)

    errors : dict
        Dictionary of uncertainties for the inferred parameters, with the same keys as 'params'.

    Notes
    -----
    - Filters lines based on reduced EW, excitation potential, fit quality, and flux availability.
    - Writes selected linemasks to file for transparency and reproducibility.
    - Uses 'iSpec.model_spectrum_from_ew' to derive stellar parameters from EWs.
    - Handles outlier rejection during fitting via sigma clipping.
    - Updates global variables for initial atmospheric parameters for potential reuse.
    - Generates plots via 'CnstrPlot()' to visualize element abundance constraints.
    - Applies special error limits for extremely metal-poor stars like 'CCLyr'.
    - Results are saved to disk with filenames dependent on whether 'nlte' is True or False.

    Global Variables
    ----------------
    initial_teff, initial_logg, initial_MH, initial_vmic, initial_alpha : float
        These variables must be defined globally prior to calling this function. They are used
        to validate and initialize the atmospheric model fitting.

    Raises
    ------
    Prints a warning if the initial parameters fall outside the supported model atmosphere grid.

    See Also
    --------
    iSpec.model_spectrum_from_ew : Function that performs the actual fitting of EW data.
    CnstrPlot : External plotting function used to visualize the element constraints.
    """
    #--- Model spectra from EW -------------------------------------------------
    global initial_teff, initial_logg, initial_MH, initial_vmic, initial_alpha
    max_iterations = 15
    #--- Model spectra ---------------------------------------------------------
    logging.info("Last initializations...")
    # Load SPECTRUM abundances
    solar_abundances_file = ispec_dir + "input/abundances/Asplund.2021/stdatom.dat"
    solar_abundances = ispec.read_solar_abundances(solar_abundances_file)
    # Load model atmospheres
    modeled_layers_pack = ispec.load_modeled_layers_pack(model_atmospheres)
    # Validate parameters
    if not ispec.valid_atmosphere_target(modeled_layers_pack, {'teff':initial_teff, \
                'logg':initial_logg, 'MH':initial_MH, 'alpha':initial_alpha}):
        msg = "The specified effective temperature, gravity (log g) and metallicity [M/H] \
                fall out of the atmospheric models."
        print(msg)
    # Reduced equivalent width
    # Filter too weak/strong lines
    # * Criteria presented in paper of GALA
    #efilter = np.logical_and(linemasks['ewr'] >= -5.8, linemasks['ewr'] <= -4.65)
    efilter = np.logical_and(linemasks['ewr'] >= -6.0, linemasks['ewr'] <= -4.3)
    # Filter high excitation potential lines
    # * Criteria from Eric J. Bubar "Equivalent Width Abundance Analysis In Moog"
    efilter = np.logical_and(efilter, linemasks['lower_state_eV'] <= 5.0)
    efilter = np.logical_and(efilter, linemasks['lower_state_eV'] >= 0.5)
    ## Filter also bad fits
    efilter = np.logical_and(efilter, linemasks['rms'] < 1.00)
    # no flux
    noflux = star_spectrum['flux'][linemasks['peak']] < 1.0e-10
    efilter = np.logical_and(efilter, np.logical_not(noflux))
    # unfitted
    unfitted = linemasks['fwhm'] == 0
    efilter = np.logical_and(efilter, np.logical_not(unfitted))
    linemasks = linemasks[efilter]
    print(len(linemasks))

    if nlte:
        f = open(ispec_dir+mySamOut_dir+objName+"_Litlast_NLTE.txt", "w")
    else:
        f = open(ispec_dir+mySamOut_dir+objName+"_Litlast.txt", "w")
    for l in linemasks:
        f.write('%.4f\t%.4f\t%.4f\t%.4f\t%s\n' % (l['wave_nm'], l['wave_peak'], l['wave_base'], 
                    l['wave_top'], l['element']))
    f.close()

    results = ispec.model_spectrum_from_ew(linemasks, modeled_layers_pack, \
                        solar_abundances, initial_teff, initial_logg, initial_MH, initial_alpha, \
                        initial_vmic, free_params=["teff", "logg", "vmic"], \
                        adjust_model_metalicity=True, \
                        max_iterations=max_iterations, \
                        enhance_abundances=True, \
                        #outliers_detection = "robust", \
                        #outliers_weight_limit = 0.90, \
                        outliers_detection = "sigma_clipping", \
                        sigma_level = 50., \
                        tmp_dir = None, \
                        code=code) #"teff", "logg", "vmic"
    params, errors, status, x_over_h, selected_x_over_h, fitted_lines_params, used_linemasks = results
    initial_teff = params['teff']; initial_logg = params['logg']
    initial_MH = params['MH']; initial_vmic = params['vmic']
    ##--- Plotting the constraints' plots --------------------------------------
    CnstrPlot(linemasks, x_over_h, nlte)
    
    ##--- Uncertainty fix for extremely metal-poor targets ---------------------
    if objName in {'CCLyr'}:
        max_errors = {'teff': 250., 'logg': 0.5, 'vmic': 1.}
        errors = {key: min(value, max_errors[key]) if key in max_errors else value for key, 
                    value in errors.items()}
    
    ##--- Save results ---------------------------------------------------------
    logging.info("Saving results...")
    if nlte:
        f = open(ispec_dir+mySamOut_dir+objName+"_res_StePar_NLTE.txt", "w")
    else:
        f = open(ispec_dir+mySamOut_dir+objName+"_res_StePar.txt", "w")
    f.write('Teff\teTeff\tlogg\telogg\tMH\teMH\tvmic\tevmic\n')
    f.write('%4.0f\t%4.0f\t%1.2f\t%1.2f\t%1.2f\t%1.2f\t%1.2f\t%1.2f\n' % (params['teff'], errors['teff'], 
                params['logg'], errors['logg'], params['MH'], errors['MH'], params['vmic'], errors['vmic']))
    f.close()
    
    return(params, errors)

def CnstrPlot(linemasks, x_over_h, nlte=False):
    """
    Generates diagnostic plots and output files to analyze iron abundance trends 
    with respect to reduced equivalent width and excitation potential for a set 
    of spectral lines.

    The function filters the provided spectral line data to include only Fe I and 
    Fe II lines, applies validity checks, and then produces two plots:
        1. [Fe/H] vs log(EW/λ) to assess microturbulence velocity correction.
        2. [Fe/H] vs lower state excitation potential to check excitation equilibrium.

    Parameters:
    ----------
    linemasks : list of structured arrays or dictionaries
        List of spectral line data. Each entry must contain at least the keys:
        'element', 'wave_peak', 'lower_state_eV', 'ewr', 'ew', 'theoretical_ew'.

    x_over_h : array-like
        Corresponding array of derived [Fe/H] values for the spectral lines. 
        Must align with the valid entries in 'linemasks'.

    nlte : bool, optional (default=False)
        If True, labels and filenames will include an '_NLTE' suffix to indicate 
        that non-LTE abundances were used.

    Outputs:
    -------
    - Two PDF plots are saved in the output directory (defined externally):
        * One for the [Fe/H] vs log(EW/λ) trend.
        * One for the [Fe/H] vs excitation potential.
    
    - A text file "<objName>_FinalValuesforPlotting" is saved with key parameters 
      for each Fe line: wavelength, excitation potential, reduced equivalent width, 
      theoretical EW, and element ID.

    Notes:
    -----
    - The function uses several global variables assumed to be defined externally 
      in the environment: 'ispec_dir', 'mySamOut_dir', and 'objName'. These specify 
      the output directory and object name for labeling and file saving.
    
    - The function prints the line-by-line diagnostics and the mean [Fe/H] derived 
      separately from Fe I and Fe II lines.

    - Requires 'numpy' and 'matplotlib'.

    """
    import matplotlib.pyplot as plt
    Fe_line_regions = [line for line in linemasks if (line[0]=='Fe 1' or line[0]=='Fe 2')]
    Fe_line_regions = np.array(Fe_line_regions)
    idx = np.isfinite(x_over_h)
    Fe_line_regions = Fe_line_regions[idx]; x_over_h = x_over_h[idx]
    ind1 = np.where(Fe_line_regions['element']=='Fe 1')[0]
    ind2 = np.where(Fe_line_regions['element']=='Fe 2')[0]

    f = open(ispec_dir+mySamOut_dir+objName+"_FinalValuesforPlotting", "w")
    for l in Fe_line_regions:
        f.write('%.4f\t%.4f\t%.2f\t%.2f\t%s\n' % (l['wave_peak'], l['lower_state_eV'], l['ewr'], 
                    l['theoretical_ew'], l['element']))
    f.close()
    for i in range(len(x_over_h)):
        print('%s: CWL=%.5f, [Fe/H]=%.2f, EW=%.2f, thEW=%.2f, rEW=%.2f, eV=%.2f' % (Fe_line_regions['element'][i], 
                    Fe_line_regions['wave_peak'][i], x_over_h[i], Fe_line_regions['ew'][i], 
                    Fe_line_regions['theoretical_ew'][i], Fe_line_regions['ewr'][i], 
                    Fe_line_regions['lower_state_eV'][i]))
    
    ## v_mic correction
    plt.xlim([-6.,-4.2])
    plt.ylim([np.nanmean(x_over_h)-1,np.nanmean(x_over_h)+1])
    plt.xlabel("$\\log$(EW/$\\lambda$)")
    plt.ylabel("[Fe/H]")
    plt.title('%s, $v_{mic}$ correction' % objName)
    plt.grid(True, zorder=0)
    coef = np.polyfit(Fe_line_regions['ewr'],x_over_h,1)
    poly1d_fn = np.poly1d(coef)
    x = np.linspace(-6., -4.2, 10)
    plt.plot(x, poly1d_fn(x), '-k', zorder=1)
    plt.scatter(Fe_line_regions['ewr'][ind1], x_over_h[ind1], s=10, c='olive', zorder=2, 
                label='Fe 1: %3i lines' % len(x_over_h[ind1]))
    plt.scatter(Fe_line_regions['ewr'][ind2], x_over_h[ind2], s=10, c='cyan', zorder=2, 
                label='Fe 2: %3i lines' % len(x_over_h[ind2]))
    plt.legend()
    if not nlte:
        nlteAdd=''
    else:
        nlteAdd='_NLTE'
    plt.savefig(ispec_dir+mySamOut_dir+objName+"_res_SlopeEqu"+nlteAdd+".pdf", bbox_inches='tight', pad_inches=0)
    plt.close()
    
    ## Excitation equilibrium
    plt.xlim([.5,5.])
    plt.ylim([np.nanmean(x_over_h)-1,np.nanmean(x_over_h)+1])
    plt.xlabel("lower state (eV)")
    plt.ylabel("[Fe/H]")
    plt.title('%s, excitation equilibrium' % objName)
    plt.grid(True, zorder=0)
    coef = np.polyfit(Fe_line_regions['lower_state_eV'],x_over_h,1)
    poly1d_fn = np.poly1d(coef)
    x = np.linspace(.5, 5., 10)
    plt.plot(x, poly1d_fn(x), '-k', zorder=1)
    plt.scatter(Fe_line_regions['lower_state_eV'][ind1], x_over_h[ind1], s=10, c='olive', zorder=2, 
                label='Fe 1: %3i lines' % len(x_over_h[ind1]))
    plt.scatter(Fe_line_regions['lower_state_eV'][ind2], x_over_h[ind2], s=10, c='cyan', zorder=2, 
                label='Fe 2: %3i lines' % len(x_over_h[ind2]))
    plt.legend()
    if not nlte:
        nlteAdd=''
    else:
        nlteAdd='_NLTE'
    plt.savefig(ispec_dir+mySamOut_dir+objName+"_res_ExcitBal"+nlteAdd+".pdf", bbox_inches='tight', pad_inches=0)
    plt.close()
    print('[Fe/H] by Fe 1: %.2f; [Fe/H] by Fe 2: %.2f' % (np.nanmean(x_over_h[Fe_line_regions['element']=='Fe 1']), 
                np.nanmean(x_over_h[Fe_line_regions['element']=='Fe 2'])))
    
def EWabund(star_spectrum, star_continuum_model, linemasks, model_atmospheres, code='moog', 
            mode='default', rewrite_abund_file=True, nlte=False):
    """
    Estimate chemical abundances of elements in a stellar spectrum using equivalent width (EW)
    or synthetic line-by-line (L2L) analysis.

    This function supports both EW-based abundance determination and synthetic spectral 
    modeling depending on the 'mode' parameter. It uses a specified radiative transfer code 
    (e.g., MOOG) and interpolates the appropriate model atmospheres to match the star's 
    atmospheric parameters. It supports NLTE corrections and optionally saves the results 
    into output files.

    Parameters
    ----------
    star_spectrum : structured array or dict-like
        Observed stellar spectrum data. Must include 'waveobs' and 'flux' arrays.

    star_continuum_model : structured array or dict-like
        Continuum-normalized model spectrum used for synthetic fitting.

    linemasks : structured array
        Line list with elemental and line properties (e.g., element name, wavelength, EW, loggf, etc.)
        used for abundance analysis.

    model_atmospheres : str
        Path to the model atmosphere grid package used for interpolation.

    code : str, optional
        Radiative transfer code to use (default is 'moog'). Compatible options depend on 'ispec'.

    mode : str, optional
        Operating mode:
        - 'default': Perform EW-based abundance analysis.
        - 'ssfl2l': Perform synthetic spectral fitting for each line (L2L approach).
        Default is 'default'.

    rewrite_abund_file : bool, optional
        Whether to overwrite output abundance result files. Default is True.

    nlte : bool, optional
        Whether to apply NLTE corrections (if available). Default is False.

    Returns
    -------
    abundArray : list of tuples
        List of tuples per element in the form:
        (element, mean [X/H], std deviation [X/H], number of lines used)

    Notes
    -----
    - Requires several global variables to be pre-defined:
        initial_teff, initial_logg, initial_MH, initial_vmic, initial_alpha
    - Output files are generated in the directories determined by the global variables
        'ispec_dir', 'mySamOut_dir', and 'objName'.
    - The function enforces a lower metallicity limit of [M/H] = -2.5 due to model grid boundaries.
    - In 'ssfl2l' mode, the function performs full spectrum synthesis using ispec's 
      'model_spectrum' method and stores synthetic spectra for each line in a subdirectory.
    - Relies on iSpec’s functions for atmosphere interpolation, abundance computation, 
      and spectral modeling.

    Raises
    ------
    RuntimeError
        If the atmospheric parameters fall outside the boundaries of the provided model grid.
    
    FileNotFoundError
        If required input files (abundances, atomic linelist, etc.) are not found.

    Examples
    --------
    >>> abundances = EWabund(spectrum, continuum, line_mask, model_grid, mode='default')
    >>> for elem, mean_abund, std_abund, n_lines in abundances:
    ...     print(f"{elem}: [{elem}/H] = {mean_abund:.2f} ± {std_abund:.2f} from {n_lines} lines")
    """
    global initial_teff, initial_logg, initial_MH, initial_vmic, initial_alpha
    max_iterations = 10
    #--- Metallicity grid limitation
    if initial_MH < -2.5:
        initial_MH = -2.5
    
    abundTargets = []
    for elem in linemasks['element']:
        if (elem not in abundTargets and elem != ''):
            abundTargets.append(elem)
    logging.info("HERE ARE OUR CONTESTANTS: %s" % abundTargets)
    
    # Load SPECTRUM abundances
    solar_abundances_file = ispec_dir + "input/abundances/Asplund.2021/stdatom.dat"
    solar_abundances = ispec.read_solar_abundances(solar_abundances_file)
    
    # Load model atmospheres
    modeled_layers_pack = ispec.load_modeled_layers_pack(model_atmospheres)

    # Validate parameters
    if not ispec.valid_atmosphere_target(modeled_layers_pack, {'teff':initial_teff, \
                'logg':initial_logg, 'MH':initial_MH, 'alpha':initial_alpha}):
        msg = "The specified effective temperature, gravity (log g) and metallicity [M/H] \
                fall out of the atmospheric models."
        print(msg)

    ##--- EW abund -------------------------------------------------------------
    abundArray = []
    if rewrite_abund_file:
        if nlte:
            fMean = open(ispec_dir+mySamOut_dir+objName+"_res_Abund_NLTE.txt", "w")
            fIndv = open(ispec_dir+mySamOut_dir+objName+"_res_IndivAbund_NLTE.txt", "w")
        else:
            fMean = open(ispec_dir+mySamOut_dir+objName+"_res_Abund.txt", "w")
            fIndv = open(ispec_dir+mySamOut_dir+objName+"_res_IndivAbund.txt", "w")
        fMean.write('element\tN\t[X/H]\te[X/H]\t[X/Fe]\te[X/Fe]\tA(X)array\tA(X)+12.036array\t[X/H]array\n')
        fIndv.write('element\twave_nm\tloggf\tlower_state_eV\tew_mA\tlog(eps)\tlog(eps)sol\tx_over_h\tx_over_fe\n')
    for elem in abundTargets: #Calculate abundance for each element
        element_name = elem[:-2] # Fe 1 -> Fe, Sc 2 -> Sc, etc.
        logging.info("CALCULATING [" + element_name + "/H].")
        elem_line_regions = [line for line in linemasks if line[0]==elem]
        elem_line_regions = np.array(elem_line_regions)
        atmosphere_layers = ispec.interpolate_atmosphere_layers(modeled_layers_pack, \
                    {'teff':initial_teff, 'logg':initial_logg, 'MH':initial_MH, 'alpha':initial_alpha}, \
                    code=code)
        spec_abund, normal_abund, x_over_h, x_over_fe = ispec.determine_abundances(atmosphere_layers, \
                        initial_teff, initial_logg, initial_MH, initial_alpha, elem_line_regions, \
                        solar_abundances, microturbulence_vel = initial_vmic, \
                        verbose=1, code=code)
        if rewrite_abund_file:
            ##--- Save results -----------------------------------------------------
            logging.info("Saving results...")
            fMean.write('%2s\t%i\t%1.2f\t%1.2f\t%1.2f\t%1.2f\t%s\t%s\n' % (elem, len(elem_line_regions), 
                        np.nanmean(x_over_h), np.nanstd(x_over_h), np.nanmean(x_over_fe), np.nanstd(x_over_fe), 
                        str(normal_abund), str(x_over_h)))
            for i in range(len(x_over_h)):
                elem_row_sol = solar_abundances['code']==int(float(elem_line_regions['spectrum_moog_species'][i]))
                log_eps_sol = solar_abundances['Abund'][elem_row_sol] + 12.036
                fIndv.write('%s\t%.4f\t%.3f\t%.3f\t%.1f\t%.2f\t%.2f\t%.2f\t%.2f\n' % (elem, 
                            elem_line_regions['wave_nm'][i], elem_line_regions['loggf'][i], 
                            elem_line_regions['lower_state_eV'][i], elem_line_regions['ew'][i], 
                            log_eps_sol[0]+x_over_h[i], log_eps_sol[0], x_over_h[i], x_over_fe[i]))
            print('%s: [%s/H] = %s' % (elem, elem[:-2], x_over_h))
        abundArray.append((elem,np.nanmean(x_over_h),np.nanstd(x_over_h),len(elem_line_regions)))
    if rewrite_abund_file:
        fMean.close()
        fIndv.close()
    
    if mode=='ssfl2l':
        ###--- SS L2L abund ----------------------------------------------------
        initial_vmac = ispec.estimate_vmac(initial_teff, initial_logg, initial_MH)
        initial_vsini = 1.6; initial_limb_darkening_coeff = 0.6; initial_vrad = 0.
        atomic_linelist_file = ispec_dir+mySamOut_dir+objName+"_LineList.txt"
        atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file, \
                    wave_base=np.min(star_spectrum['waveobs']), \
                    wave_top=np.max(star_spectrum['waveobs']))
        #atomic_linelist = atomic_linelist[atomic_linelist['theoretical_depth'] >= 0.01]
        chemical_elements_file = ispec_dir + "input/abundances/chemical_elements_symbols.dat"
        chemical_elements = ispec.read_chemical_elements(chemical_elements_file)
        isotope_file = ispec_dir + "input/isotopes/SPECTRUM.lst"
        isotopes = ispec.read_isotope_data(isotope_file)
        linelist_free_loggf = None
        free_params = []
        f = open(ispec_dir+mySamOut_dir+objName+"_res_L2LSynthAbund.txt","w")
        f.write('element\twl (nm)\tTurbocode\t[X/H]\te[X/H]\t[X/Fe]\te[X/Fe]\n')
        for i, line in enumerate(linemasks): #Calculate abundance for each line
            if line['element'][-1] == ' ' or int(line['element'][-1]) > 2:
                continue
            element_name = line['element'][:-2] # Fe 1 -> Fe, Sc 2 -> Sc, etc.
            free_abundances = ispec.create_free_abundances_structure([element_name], \
                        chemical_elements, solar_abundances)
            free_abundances['Abund'] += initial_MH # Scale to metallicity
            logging.info("CALCULATING SYNTHETIC [" + element_name + "/Fe] at " + 
                        str(np.round(line['wave_peak'],3)) + " nm (line #" + str(i+1) + 
                        "/"+str(len(linemasks))+").")
            individual_line_regions = linemasks[i:i+1]
            segments = ispec.create_segments_around_lines(individual_line_regions, margin=0.25)
            wfilter = ispec.create_wavelength_filter(star_spectrum, regions=segments) # Only use the segment
            if len(star_spectrum[wfilter]) == 0 or np.any(star_spectrum['flux'][wfilter] == 0):
                continue
            obs_spec, modeled_synth_spectrum, params, errors, abundances_found, loggf_found, \
                    status, stats_linemasks = ispec.model_spectrum(star_spectrum[wfilter], \
                    star_continuum_model, modeled_layers_pack, atomic_linelist, isotopes, \
                    solar_abundances, free_abundances, linelist_free_loggf, initial_teff, \
                    initial_logg, initial_MH, initial_alpha, initial_vmic, initial_vmac, initial_vsini, \
                    initial_limb_darkening_coeff, initial_R, initial_vrad, free_params, segments=segments, \
                    linemasks=individual_line_regions, enhance_abundances=True, use_errors = False, \
                    vmic_from_empirical_relation = False, vmac_from_empirical_relation = False, \
                    max_iterations=max_iterations, tmp_dir = None, code=code)
            logging.info("SAVING SYNTHETIC [%s/Fe] = %.2f at %.3f nm (line #%i/%i)." % (element_name, 
                        abundances_found[0][5], line['wave_peak'], i+1, len(linemasks)))
            if abundances_found[0][7] > 5.: #Bad uncertainties
                abundances_found[0][7] = -1.
                abundances_found[0][9] = -1.
            f.write('%2s\t%4.3f\t%s\t%1.2f\t%1.2f\t%1.2f\t%1.2f\n' % (abundances_found[0][2], line['wave_peak'], 
                        line['turbospectrum_species'], abundances_found[0][3], abundances_found[0][7], 
                        abundances_found[0][5], abundances_found[0][9]))
            ##--- Save synthetic spectrum --------------------------------------
            synthSpecName = ispec_dir+mySamOut_dir+"L2LSynthSpec/"
            if not os.path.exists(synthSpecName):
                os.makedirs(synthSpecName)
            fOut = open(synthSpecName+objName+"_"+element_name+"_"+
                        str(np.round(line['wave_peak'],2))+"nm.txt", "w")
            for s in modeled_synth_spectrum:
                fOut.write('%.5f\t%.7f\t%.7f\n' % (s['waveobs'], s['flux'], s['err']))
            fOut.close()
        f.close()
    return(abundArray)

def SaveSpec(star_spectrum, spectrum_type):
    """
    Saves a star's spectral data to a text file in a tab-delimited format.

    This function writes the observed wavelength ('waveobs'), flux ('flux'), and 
    error ('err') from a list of spectral data dictionaries to a text file. 
    The file is named based on the global 'objName' and 'spectrum_type', and is 
    saved in the directory specified by the global variables 'ispec_dir' and 
    'mySamOut_dir'.

    Parameters
    ----------
    star_spectrum : list of dict
        A list where each element is a dictionary representing a spectral data 
        point. Each dictionary must contain the keys:
            - 'waveobs': float, the observed wavelength
            - 'flux'   : float, the measured flux at that wavelength
            - 'err'    : float, the uncertainty in the flux measurement

    spectrum_type : str
        A string indicating the type of spectrum (e.g., 'obs', 'model', etc.), 
        used to differentiate output filenames.

    Notes
    -----
    - This function depends on the global variables 'ispec_dir', 'mySamOut_dir', 
      and 'objName' to determine the output file path.
    - The output file will be overwritten if it already exists.
    - The output format is:
        waveobs    flux    err
        <float>    <float> <float>
    - All numeric values are written with fixed precision:
        waveobs: 5 decimal places
        flux, err: 7 decimal places

    Examples
    --------
    >>> star_spectrum = [
    ...     {'waveobs': 5000.12345, 'flux': 1.2345678, 'err': 0.0001234},
    ...     {'waveobs': 5001.54321, 'flux': 1.3456789, 'err': 0.0002345}
    ... ]
    >>> SaveSpec(star_spectrum, 'obs')
    # Creates a file named like '.../objName_obs.txt' with tab-delimited columns
    """
    f = open(ispec_dir+mySamOut_dir+objName+"_"+spectrum_type+".txt", "w")
    f.write('waveobs\tflux\terr\n')
    for s in star_spectrum:
        f.write('%.5f\t%.7f\t%.7f\n' % (s['waveobs'], s['flux'], s['err']))
    f.close()



def interpolate_atmosphere(code="turbospectrum"):
    """
    Interpolates a stellar atmosphere model using predefined global stellar parameters.

    This function loads a grid of precomputed model atmospheres (specifically ATLAS9 Kurucz ODFNEW models),
    validates whether the specified global stellar parameters fall within the bounds of the available model grid,
    and interpolates the atmospheric layers accordingly using the iSpec library. The resulting interpolated model
    is then saved to a file.

    Parameters:
    -----------
    code : str, optional
        The spectral synthesis code to use for formatting the atmospheric model.
        Supported values typically include 'turbospectrum', 'sme', etc.
        Default is "turbospectrum".

    Global Variables:
    -----------------
    initial_teff : float
        The effective temperature (Teff) of the star in Kelvin.

    initial_logg : float
        The surface gravity (log g) of the star in cgs units.

    initial_MH : float
        The metallicity [M/H] of the star, i.e., the logarithmic abundance of metals
        relative to hydrogen compared to the Sun.

    initial_R : float
        The resolving power of the instrument (not directly used in this function).

    initial_vmic : float
        The microturbulence velocity in km/s (not directly used in this function).

    initial_alpha : float
        The alpha-element enhancement [α/Fe] of the star.

    iSpec-dependent Globals:
    ------------------------
    ispec_dir : str
        Base directory where iSpec resources and outputs are stored.

    mySamOut_dir : str
        Subdirectory name for storing output files.

    objName : str
        Identifier for the target star or observation, used in the output file name.

    Returns:
    --------
    None
        The function does not return any object. It writes the interpolated atmosphere
        model to a text file in the specified output directory.

    Notes:
    ------
    - If the input stellar parameters fall outside the range covered by the atmospheric
      model grid, a warning message is printed.
    - The atmospheric model grid is assumed to be in iSpec-compatible format.
    - The output file name is constructed using the 'objName' and saved with a suffix "_InterpModel.txt".
    """
    global initial_teff, initial_logg, initial_MH, initial_R, initial_vmic, initial_alpha
    model = ispec_dir + "/input/atmospheres/ATLAS9.KuruczODFNEW/"
    # Load model atmospheres
    modeled_layers_pack = ispec.load_modeled_layers_pack(model)

    # Validate parameters
    if not ispec.valid_atmosphere_target(modeled_layers_pack, {'teff':initial_teff, \
                'logg':initial_logg, 'MH':initial_MH, 'alpha':initial_alpha}):
        msg = "The specified effective temperature, gravity (log g) and metallicity [M/H] \
                fall out of theatmospheric models."
        print(msg)

    # Prepare atmosphere model
    atmosphere_layers = ispec.interpolate_atmosphere_layers(modeled_layers_pack, \
                {'teff':initial_teff, 'logg':initial_logg, 'MH':initial_MH, 'alpha':initial_alpha}, \
                code=code)
    atmosphere_layers_file = ispec_dir + mySamOut_dir + objName + "_InterpModel.txt"
    atmosphere_layers_file = ispec.write_atmosphere(atmosphere_layers, initial_teff, initial_logg, \
                initial_MH, atmosphere_filename=atmosphere_layers_file, code=code)



def WriteErrCalcBest(linemasks, model_atmospheres, nlte=False):
    """
    Calculates and writes the best-fit elemental abundances and associated uncertainties 
    to a formatted text file for a given stellar spectrum using equivalent width (EW) analysis.

    This function computes the elemental abundances from a stellar spectrum by using line masks 
    and model atmospheres, and then saves the results (including the element name, abundance, 
    line-to-line scatter, and number of lines used) to a file. The output is directed to a specific 
    subdirectory named "ErrCalc" within the analysis path.

    Parameters:
    ----------
    linemasks : list or dict
        A list or dictionary specifying spectral line masks to be used in the abundance analysis.
        Each mask typically includes wavelength regions to isolate lines of specific elements.
    
    model_atmospheres : object
        A model atmosphere object or data structure compatible with the abundance calculation 
        function, representing stellar atmospheric parameters (e.g., temperature, gravity, metallicity).
    
    nlte : bool, optional (default=False)
        If True, indicates that NLTE (Non-Local Thermodynamic Equilibrium) abundances are to be computed 
        and the output file will be named accordingly (appending "_NLTE" to the filename). 
        If False, LTE (Local Thermodynamic Equilibrium) is assumed.

    Outputs:
    -------
    A tab-separated text file is saved in the "ErrCalc" subdirectory. The filename is based on the 
    global variable 'objName', and includes a "_best" suffix. If 'nlte=True', the filename will include 
    "_best_NLTE" instead. The file contains the following columns:
        - element: Chemical element symbol.
        - [X/H]: Mean abundance relative to hydrogen.
        - l2l: Line-to-line scatter (standard deviation among lines).
        - Nlines: Number of lines used in the calculation.

    Notes:
    -----
    - The function assumes the existence of several global variables:
        - 'ispec_dir' and 'mySamOut_dir': Base directories for saving output.
        - 'objName': Name identifier for the target object/spectrum.
        - 'star_spectrum': The observed stellar spectrum.
        - 'star_continuum_model': The model used to normalize or fit the stellar continuum.
    - The abundance calculation is performed by the function 'EWabund', 
      which is assumed to be defined elsewhere and supports MOOG as the backend.

    Raises:
    ------
    This function does not explicitly raise exceptions, but relies on the existence of 
    valid global variables and a properly configured file system.
    """
    ErrCalc_dir = ispec_dir+mySamOut_dir+"ErrCalc/"
    if not os.path.exists(ErrCalc_dir):
         os.makedirs(ErrCalc_dir)
    if nlte:
        fnorm = open(ErrCalc_dir+objName+"_best_NLTE.txt", "w")
    else:
        fnorm = open(ErrCalc_dir+objName+"_best.txt", "w")
    abunds = EWabund(star_spectrum, star_continuum_model, linemasks, model_atmospheres, code="moog") #, mode="ssfl2l"
    fnorm.write('element\t[X/H]\tl2l\tNlines\n')
    for a in abunds:
        fnorm.write("%s\t%.2f\t%.2f\t%i\n" % a) # elem, mean, l2l scatter, Nlines
    fnorm.close()

def ErrStud(star_spectrum, star_continuum_model, model_atmospheres, linemasks, rv, params, errors, nlte=False):
    """
    Performs an error analysis on derived stellar abundances by varying individual stellar parameters 
    (e.g., effective temperature, surface gravity, metallicity, microturbulence) within their respective 
    uncertainties, and computing the resulting changes in elemental abundances.

    For each parameter, this function perturbs the parameter by ±1σ (as given in 'errors') while holding the 
    others constant, runs the EWabund abundance determination routine, and writes the resulting elemental 
    abundances and line-to-line scatter to output files.

    Parameters:
    ----------
    star_spectrum : object
        The observed stellar spectrum to be analyzed.
    
    star_continuum_model : object
        The continuum model applied to the stellar spectrum.
    
    model_atmospheres : object
        Atmospheric models used for abundance analysis.
    
    linemasks : dict
        Dictionary of line masks used to select spectral lines for abundance determination.
    
    rv : float
        Radial velocity of the star (not used directly in this function but passed for consistency).
    
    params : dict
        Dictionary of stellar parameters with keys including:
            'teff' (effective temperature),
            'logg' (surface gravity),
            'MH' (metallicity),
            'vmic' (microturbulence),
            and optionally 'alpha'.
    
    errors : dict
        Dictionary of uncertainties corresponding to each stellar parameter in 'params'.
    
    nlte : bool, optional
        If True, output filenames include an '_NLTE' suffix indicating non-LTE conditions.
        Default is False (assumes LTE).

    Outputs:
    -------
    For each varied parameter (excluding 'alpha'), two files are created:
        - '<objName>_<param>+error(_NLTE).txt'
        - '<objName>_<param>-error(_NLTE).txt'

    Each file contains tab-separated columns:
        - element: Element symbol (e.g., Fe, Mg)
        - [X/H]: Derived abundance relative to solar
        - l2l: Line-to-line scatter (standard deviation)
        - Nlines: Number of lines used

    Notes:
    -----
    - Uses global variables: 'initial_teff', 'initial_logg', 'initial_MH', 'initial_vmic', 'objName',
      'ispec_dir', 'mySamOut_dir'.
    - Relies on external function 'EWabund' to perform abundance calculations using MOOG.
    - Modifies global stellar parameters temporarily for each test; restores them after each run.
    - Skips 'alpha' parameter in error study.
    - Assumes presence of required directories and write permissions for output.

    """
    global initial_teff, initial_logg, initial_MH, initial_vmic
    teff_copy = initial_teff; logg_copy = initial_logg; MH_copy = initial_MH; vmic_copy = initial_vmic
    for (key1, param), (key2, error) in zip(params.items(), errors.items()):
        if key1=='alpha':
            continue
        params_test = params; errors_test = errors
        params_test[key1] = param+error
        if nlte:
            fdevs = open(f'{ispec_dir}{mySamOut_dir}ErrCalc/{objName}_{key1}+error_NLTE.txt', "w")
        else:
            fdevs = open(f'{ispec_dir}{mySamOut_dir}ErrCalc/{objName}_{key1}+error.txt', "w")
        fdevs.write('element\t[X/H]\tl2l\tNlines\n')
        initial_teff = params_test['teff']; initial_logg = params_test['logg']
        initial_MH = params_test['MH']; initial_vmic = params_test['vmic']
        abunds = EWabund(star_spectrum, star_continuum_model, linemasks, model_atmospheres, 
                    code="moog", rewrite_abund_file=False) #, mode="ssfl2l"
        for a in abunds:
            fdevs.write("%s\t%.2f\t%.2f\t%i\n" % a)
        fdevs.close()
        
        params_test[key1] = param-error
        if nlte:
            fdevs = open(f'{ispec_dir}{mySamOut_dir}ErrCalc/{objName}_{key1}-error_NLTE.txt', "w")
        else:
            fdevs = open(f'{ispec_dir}{mySamOut_dir}ErrCalc/{objName}_{key1}-error.txt', "w")
        fdevs.write('element\t[X/H]\terr\tNlines\n')
        initial_teff = params_test['teff']; initial_logg = params_test['logg']
        initial_MH = params_test['MH']; initial_vmic = params_test['vmic']
        if initial_logg<0.:
            initial_logg=0.
        abunds = EWabund(star_spectrum, star_continuum_model, linemasks, model_atmospheres, 
                    code="moog", rewrite_abund_file=False) #, mode="ssfl2l"
        for a in abunds:
            fdevs.write("%s\t%.2f\t%.2f\t%i\n" % a)
        params_test[key1] = param
        initial_teff = teff_copy; initial_logg = logg_copy; initial_MH = MH_copy; initial_vmic = vmic_copy
        fdevs.close()

def TotalErrCalc(nlte=False):
    """
    Calculates total abundance errors for stellar elemental analysis by combining
    line-to-line scatter with perturbation-based uncertainties in atmospheric parameters.
    
    This function reads in abundance data from a main result file and multiple
    perturbed model output files (varying Teff, logg, vmic), computes the absolute
    differences between the baseline abundances and those from the perturbed models,
    calculates average perturbation effects, and combines all uncertainties to produce
    a total error for each element. The results are saved to summary files.

    Parameters:
    -----------
    nlte : bool, optional
        If True, uses NLTE-corrected input/output files.
        If False (default), uses LTE files.

    Input Files:
    ------------
    Located in: {ispec_dir}{mySamOut_dir}ErrCalc/
    - {objName}_best[(_NLTE)].txt: Main abundance result file.
    - {objName}_<param>[+/-]error[(_NLTE)].txt: Files corresponding to
      perturbed values of Teff, logg, and vmic.

    Output Files:
    -------------
    - {objName}_ErrMatrix[(_NLTE)].txt:
        Contains all element-wise data with perturbation errors and total combined errors.
    - {objName}_FinalErr[(_NLTE)].txt:
        Contains a simplified summary with element name, abundance, and total error.

    Computation Details:
    --------------------
    - The second column (abundance) from each perturbation file is compared to the baseline,
      and the absolute difference is taken.
    - Average errors are calculated for Teff, logg, and vmic (e.g., mean of +error and -error).
    - The total uncertainty is computed using quadrature:
          total_error = sqrt(l2l^2 + dTeff^2 + dLogg^2 + dVmic^2)
    - For lines with only one measurement (Nlines == 1), a minimum line-to-line scatter
      (l2l) of 0.10 dex is enforced.

    Notes:
    ------
    - Assumes global variables 'ispec_dir', 'mySamOut_dir', and 'objName' are defined.
    - Assumes NumPy is imported as 'np'.

    Raises:
    -------
    FileNotFoundError:
        If any expected input file is missing.
    ValueError:
        If file formats are incorrect or data cannot be parsed as expected.
    """
    data = []
    if nlte:
        fileBest = open(f'{ispec_dir}{mySamOut_dir}ErrCalc/{objName}_best_NLTE.txt', 'r')
    else:
        fileBest = open(f'{ispec_dir}{mySamOut_dir}ErrCalc/{objName}_best.txt', 'r')
    for line in fileBest:
        if line.startswith('element'):
            headers = line.strip().split('\t')
        else:
            data.append(line.strip().split('\t'))
    fileBest.close()

    # Read other files and append second columns
    elements = ['teff+error', 'teff-error', 'logg+error', 'logg-error', 'vmic+error', 'vmic-error']
    for element in elements:
        if nlte:
            filename = f'{ispec_dir}{mySamOut_dir}ErrCalc/{objName}_{element}_NLTE.txt'
        else:
            filename = f'{ispec_dir}{mySamOut_dir}ErrCalc/{objName}_{element}.txt'
        with open(filename, 'r') as file:
            headers.extend([f'{element}'])
            for idx, line in enumerate(file):
                if idx == 0:
                    continue  # Skip header line
                parts = line.strip().split('\t')
                diff = np.abs(float(parts[1]) - float(data[idx-1][1]))
                data[idx-1].extend([f'{diff:.2f}'])

    # Compute average values for 'dteff', 'dlogg', and 'dvmic'
    dteff_values = [(float(data[i][4]) + float(data[i][5])) / 2 for i in range(len(data))]
    dlogg_values = [(float(data[i][6]) + float(data[i][7])) / 2 for i in range(len(data))]
    dvmic_values = [(float(data[i][8]) + float(data[i][9])) / 2 for i in range(len(data))]
    
    # Set l2l=0.10 dex where Nlines=1
    [row.__setitem__(2, '0.10') for row in data if row[3]=='1']
    dtotal_values = [np.sqrt(float(data[i][2])**2 + float(dteff_values[i])**2 + 
                float(dlogg_values[i])**2 + float(dvmic_values[i])**2) for i in range(len(data))]

    # Add the computed values to headers and data
    headers.extend(['dteff', 'dlogg', 'dvmic', 'dtotal'])
    for i in range(len(data)):
        data[i].extend([f'{dteff_values[i]:.2f}', f'{dlogg_values[i]:.2f}', f'{dvmic_values[i]:.2f}', 
                    f'{dtotal_values[i]:.2f}'])

    # Write all data to "..._ErrMatrix(_NLTE).txt"
    if nlte:
        fileMtrx = open(f'{ispec_dir}{mySamOut_dir}ErrCalc/{objName}_ErrMatrix_NLTE.txt', 'w')
    else:
        fileMtrx = open(f'{ispec_dir}{mySamOut_dir}ErrCalc/{objName}_ErrMatrix.txt', 'w')
        # Write headers
        fileMtrx.write('\t'.join(headers) + '\n')
        # Write data rows
        for row in data:
            fileMtrx.write('\t'.join(row) + '\n')
    fileMtrx.close()

    # Write only element, abund, and total error to "..._FinalErr(_NLTE).txt"
    if nlte:
        fileFnl = open(f'{ispec_dir}{mySamOut_dir}ErrCalc/{objName}_FinalErr_NLTE.txt', 'w')
    else:
        fileFnl = open(f'{ispec_dir}{mySamOut_dir}ErrCalc/{objName}_FinalErr.txt', 'w')
    # Write headers
    fileFnl.write(f'{headers[0]}\t{headers[1]}\t{headers[-1]}\n')
    # Write data rows
    for row in data:
        fileFnl.write(f'{row[0]}\t{row[1]}\t{row[-1]}\n')
    fileFnl.close()

def custom_sort(element):
    """
    Returns the index of the given element in a predefined sorting order (atomic number).

    This function is intended for use as a key in sorting operations (e.g., with 'sorted()' or 'list.sort()'),
    where elements should be ordered according to a specific sequence of chemical species and ionization states.
    The sorting order is defined based on domain-specific priorities, such as commonality in astrophysical
    or spectroscopic applications.

    Parameters:
    -----------
    element : dict
        A dictionary containing at least the key 'element', whose value is a string representing a chemical 
        element and its ionization state in the format "Symbol Ionization", e.g., "Fe 1" for neutral iron
        or "Mg 2" for singly ionized magnesium.

    Returns:
    --------
    int
        The index of the element in the predefined custom sorting list.

    Raises:
    -------
    ValueError
        If the element string is not found in the sorting order.

    Example:
    --------
    >>> custom_sort({'element': 'Fe 1'})
    22

    >>> sorted(elements, key=custom_sort)
    # Sorts a list of dictionaries by the custom element order
    """
    sorting_order = ['C 1', 'N 1', 'O 1', 'Na 1', 'Mg 1', 'Mg 2', 'Al 1', 'Si 1', 'Si 2', 'S 1', 
                'K 1', 'Ca 1', 'Ca 2', 'Sc 2', 'Ti 1', 'Ti 2', 'V 1', 'V 2', 'Cr 1', 'Cr 2', 'Mn 1', 
                'Mn 2', 'Fe 1', 'Fe 2', 'Co 1', 'Ni 1', 'Ni 2', 'Cu 1', 'Zn 1', 'Sr 2', 'Y 2', 'Zr 2', 
                'Ba 2', 'La 2', 'Ce 2', 'Nd 2', 'Sm 2', 'Eu 2', 'Gd 2', 'Dy 2']
    return sorting_order.index(element['element'])

def MasterList(mode, nlte=False): #line/abund
    """
    Generates a master summary file containing spectroscopic line or abundance data
    for a list of target stars, aggregated by wavelength ('wave_nm'). The function 
    reads individual output files, extracts relevant information, and compiles a 
    master line or abundance list as a tab-separated values (TSV) file.

    Parameters:
    -----------
    mode : str
        Mode of operation, must be either ''line'' or ''abund''.
        - ''line'': Extracts equivalent widths ('ew') from line fit files.
        - ''abund'': Extracts elemental abundances ('x_over_h') from abundance result files.
    
    nlte : bool, optional (default=False)
        Indicates whether to use NLTE (Non-Local Thermodynamic Equilibrium) results.
        - If 'True', reads from files with "_NLTE" in the filename for abundance mode.
        - Has no effect on file paths in ''line'' mode.

    Behavior:
    ---------
    1. Reads line fit or abundance result files for all targets defined in 'master_list_targets'.
    2. Extracts header columns and corresponding data entries from each file.
    3. Aggregates all entries into a single dictionary keyed by 'wave_nm' (wavelength).
    4. Ensures that each wavelength entry contains values for all targets (or '-' if missing).
    5. Writes a master output file with columns:
        ['element', 'wave_nm', 'loggf', 'lower_state_eV', <target1>, <target2>, ..., <targetN>]
       where the target columns contain 'ew' or 'x_over_h' values depending on the mode.

    Output:
    -------
    A TSV file is saved to:
        - "mySampleAll/output/MasterLineList.txt" if mode is ''line'' and 'nlte' is False
        - "mySampleAll/output/MasterLineList_NLTE.txt" if mode is ''line'' and 'nlte' is True
        - "mySampleAll/output/MasterAbundList.txt" if mode is ''abund'' and 'nlte' is False
        - "mySampleAll/output/MasterAbundList_NLTE.txt" if mode is ''abund'' and 'nlte' is True

    Raises:
    -------
    ValueError:
        If a file contains invalid formatting or cannot be parsed correctly.
    
    Notes:
    ------
    - The function relies on the global variable 'master_list_targets', which should
      be a list of target names (strings) that correspond to the directories in the file paths.
    - Uses 'custom_sort()' for ordering rows in the final output; this function must be 
      defined elsewhere in the program.
    - Assumes all input files are tab-delimited and located in subdirectories under 
      "mySampleAll/output/<target>/".

    Example Usage:
    --------------
    MasterList('line')
    MasterList('abund', nlte=True)
    """
    def read_arrays_from_files(targets):
        arrays = {}
        for target in targets:
            if mode=='line':
                file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                            f"mySampleAll/output/{target}/LineFitOutput/{target}_LineFit.txt")
            elif mode=='abund':
                if nlte:
                    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                                f"mySampleAll/output/{target}/{target}_res_IndivAbund_NLTE.txt")
                else:
                    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                                f"mySampleAll/output/{target}/{target}_res_IndivAbund.txt")
            arrays[target] = {'columns': [], 'data': []}
            with open(file_path, 'r') as file:
                try:
                    lines = file.readlines()
                    keys = lines[0].strip().split('\t')
                    arrays[target]['columns'] = keys
                    for line in lines[1:]:
                        data = line.strip().split('\t')
                        array_data = {}
                        for i, value in enumerate(data):
                            try:
                                array_data[keys[i]] = float(value)
                            except ValueError:
                                array_data[keys[i]] = value
                        arrays[target]['data'].append(array_data)
                except ValueError:
                    print(f"Error reading file '{file_path}': invalid data format.")
        return(arrays)

    targets = master_list_targets
    arrays = read_arrays_from_files(targets)

    # Combine all the shortlist files by 'wave_nm' into a master dictionary
    master_dict = {}
    for target, data_dict in arrays.items():
        for entry in data_dict['data']:
            wave_nm = entry['wave_nm']
            if wave_nm not in master_dict:
                master_dict[wave_nm] = {'element': entry['element'], 'loggf': entry['loggf'], 
                            'lower_state_eV': entry['lower_state_eV']}
                for f in targets:
                    master_dict[wave_nm][f] = "-"
            if mode=='line':
                master_dict[wave_nm][target] = entry['ew']
            elif mode=='abund':
                master_dict[wave_nm][target] = entry['x_over_h']

    # Write the master dictionary into the "MasterLineList.txt" file
    if nlte:
        if mode=='line':
            filenameOut = "mySampleAll/output/MasterLineList_NLTE.txt"
        elif mode=='abund':
            filenameOut = "mySampleAll/output/MasterAbundList_NLTE.txt"
    else:
        if mode=='line':
            filenameOut = "mySampleAll/output/MasterLineList.txt"
        elif mode=='abund':
            filenameOut = "mySampleAll/output/MasterAbundList.txt"
    with open(filenameOut, "w") as master_file:
        # Write header
        master_file.write("element\twave_nm\tloggf\tlower_state_eV\t")
        master_file.write("\t".join(f"{target}" for target in targets))
        master_file.write("\n")

        # Write sorted data
        for wave_nm, data in sorted(master_dict.items(), key=lambda x: (custom_sort(x[1]), x[0])):
            master_file.write(f"{data['element']}\t{wave_nm:.4f}\t{data['loggf']:.3f}\t{data['lower_state_eV']:.3f}\t")
            master_file.write("\t".join(f"{data.get(target, '-'):.2f}" if 
                        isinstance(data.get(target), float) else f"{data.get(target, '-')}" for target in targets))
            master_file.write("\n")



def MasterAbundPack(nlte=False):
    """
    Aggregates and organizes stellar elemental abundance data from multiple targets into a master file.

    This function reads elemental abundance data for a predefined list of stellar targets 
    (defined globally as 'master_list_targets') from individual output files, and compiles 
    them into a single master file named 'MasterAbundPack[OptionalNLTE].txt'. It handles both 
    LTE and NLTE modes, controlled by the 'nlte' parameter.

    The function processes input files formatted as tab-separated values with a header and 
    three columns: element name, abundance ([X/H]), and total uncertainty (dtotal). 
    It expects files to be located at:
        mySampleAll/output/{target}/ErrCalc/{target}_FinalErr[OptionalNLTE].txt

    The final output file contains each target as a row, with abundances and uncertainties 
    listed in a fixed order of elements (defined in 'sorting_order'). If an element is 
    missing for a target, "NaN" values are recorded in the output.

    Parameters:
    ----------
    nlte : bool, optional (default=False)
        If True, the function reads and writes files in NLTE mode, modifying filenames 
        accordingly by appending '_NLTE'. If False, LTE mode is assumed.

    Output:
    ------
    A master file is written to:
        mySampleAll/output/MasterAbundPack[OptionalNLTE].txt

    The format of the file is:
        Target <element1> <element1_err> <element2> <element2_err> ... 

    Example:
    -------
    MasterAbundPack()          # Generate master file using LTE data
    MasterAbundPack(nlte=True) # Generate master file using NLTE data

    Notes:
    -----
    - The function assumes 'master_list_targets' is defined in the global scope.
    - If data files are missing or have formatting issues, errors are printed to the console.
    - Elements are written in a consistent order for all targets, even if some values are missing.
    """
    def read_arrays_from_files(targets):
        arrays = {}
        for target in targets:
            if not nlte:
                nlteAdd = ''
            else:
                nlteAdd = '_NLTE'
            file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                        f"mySampleAll/output/{target}/ErrCalc/{target}_FinalErr{nlteAdd}.txt")
            arrays[target] = {'columns': [], 'data': []}
            with open(file_path, 'r') as file:
                try:
                    lines = file.readlines()
                    keys = lines[0].strip().split('\t')
                    arrays[target]['columns'] = keys
                    for line in lines[1:]:
                        data = line.strip().split('\t')
                        array_data = {}
                        array_data['element'] = data[0]
                        array_data['[X/H]'] = float(data[1])
                        array_data['dtotal'] = float(data[2])
                        arrays[target]['data'].append(array_data)
                except (ValueError, IndexError):
                    print(f"Error reading file '{file_path}': invalid data format.")
        return(arrays)

    targets = master_list_targets
    arrays = read_arrays_from_files(targets)

    # Write the master dictionary into the "MasterAbundList(_NLTE).txt" file
    if not nlte:
        nlteAdd = ''
    else:
        nlteAdd = '_NLTE'
    master_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                f"mySampleAll/output/MasterAbundPack{nlteAdd}.txt")
    with open(master_file_path, "w") as master_file:
        # Write header
        sorting_order = ['C 1', 'N 1', 'O 1', 'Na 1', 'Mg 1', 'Mg 2', 'Al 1', 'Si 1', 'Si 2', 'S 1', 
                    'K 1', 'Ca 1', 'Ca 2', 'Sc 2', 'Ti 1', 'Ti 2', 'V 1', 'V 2', 'Cr 1', 'Cr 2', 'Mn 1', 
                    'Mn 2', 'Fe 1', 'Fe 2', 'Co 1', 'Ni 1', 'Ni 2', 'Cu 1', 'Zn 1', 'Sr 2', 'Y 2', 'Zr 2', 
                    'Ba 2', 'La 2', 'Ce 2', 'Nd 2', 'Sm 2', 'Eu 2', 'Gd 2', 'Dy 2']
        master_file.write("Target")
        for el in sorting_order:
            master_file.write(f"\t{el}\terr")
        master_file.write("\n")

        # Write sorted data
        for target, data_dict in arrays.items():
            master_file.write(f"{target}")
            for el in sorting_order:
                found = False
                for entry in data_dict['data']:
                    if entry['element'] == el:
                        master_file.write(f"\t{entry['[X/H]']:.2f}\t{entry['dtotal']:.2f}")
                        found = True
                        break
                if not found:
                    master_file.write("\tNaN\tNaN")
            master_file.write("\n")



def DoAllLists(nlte=False): #To be called with any ONE target
    """
    Executes a complete data aggregation pipeline for stellar spectroscopy results, 
    generating master summary files for both line data and elemental abundances.

    This function serves as a wrapper to sequentially run:
        1. 'MasterList(mode='line', nlte=nlte)'
        2. 'MasterList(mode='abund', nlte=nlte)'
        3. 'MasterAbundPack(nlte=nlte)'

    The output consists of:
        - A master line list file containing equivalent widths ('ew') for all targets.
        - A master abundance list file containing abundance ratios ('[X/H]') for all targets.
        - A final abundance pack summarizing key elemental abundances and associated uncertainties.

    Parameters:
    -----------
    nlte : bool, optional (default=False)
        Indicates whether to use NLTE (Non-Local Thermodynamic Equilibrium) results.
        - If 'True', NLTE-specific filenames are used for reading and writing.
        - If 'False', the pipeline uses standard LTE-based filenames.

    Output:
    -------
    The function produces the following output files in the 'mySampleAll/output/' directory:
        - MasterLineList[Optional_NLTE].txt
        - MasterAbundList[Optional_NLTE].txt
        - MasterAbundPack[Optional_NLTE].txt

    Notes:
    ------
    - This function assumes the presence of a global variable 'master_list_targets', 
      which defines the list of stellar targets to be included in the aggregation.
    - All intermediate files must exist and follow the expected tab-delimited structure.
    - Use this function when you want to regenerate **all** master files in one step.

    Example Usage:
    --------------
    DoAllLists()           # Run the full aggregation pipeline in LTE mode
    DoAllLists(nlte=True)  # Run the full aggregation pipeline in NLTE mode
    """
    MasterList(mode='line', nlte=nlte)
    MasterList(mode='abund', nlte=nlte)
    MasterAbundPack(nlte=nlte)



def NLTECorrAddition(): # Version without parameter re-estimation
    """
    Applies Non-Local Thermodynamic Equilibrium (NLTE) corrections to elemental abundance results
    for a given object and saves the updated data to a new file.

    This function reads previously computed elemental abundance results for spectral lines,
    corrects these abundances by adding corresponding NLTE correction values (if available),
    and writes the updated values to a new output file. Lines without available corrections
    are retained with their original values.

    Assumptions:
    - The global variable 'objName' must be defined prior to calling this function. It is used to 
      locate object-specific input and output files.
    - The NumPy module ('np') must be imported and available in the global scope.

    File Operations:
    - Reads the NLTE correction values from 'mySampleAll/output/result_all.txt'.
    - Reads original abundance results from 'mySampleAll/output/{objName}/{objName}_res_IndivAbund.txt'.
    - Writes corrected abundance results to 'mySampleAll/output/{objName}/{objName}_res_IndivAbund_NLTE.txt'.

    NLTE corrections are only applied when:
    - The element identifier matches (ignoring the last two characters of the element string).
    - The wavelength in the data file matches the entry in the NLTE correction file within a 0.01 nm tolerance.

    Notes:
    - Missing or invalid NLTE correction values (-1000.0) are replaced with NaN and ignored.
    - Only the last column in the original abundance file (assumed to be the abundance value) is updated.
    - The header line of the original abundance file is preserved.

    Logs:
    - Logs a message indicating the start of the NLTE correction process.

    Returns:
    None
    """
    logging.info("CORRECTING ABUNDANCES FOR NLTE EFFECTS")
    import pandas as pd
    df = pd.read_fwf(f"mySampleAll/output/result_all.txt", sep='\t', header=0, index_col=False)
    df.replace([-1000.0], np.nan, inplace=True)

    with open(f"mySampleAll/output/{objName}/{objName}_res_IndivAbund.txt", 'r') as file:
        lines = file.readlines()

    updated_lines = []
    for line in lines[1:]:
        parts = line.split('\t')
        elem_array = df[df['el']==parts[0][:-2]]
        if elem_array.empty:
            updated_lines.append(line)
            continue
        line_array = elem_array[abs(elem_array['wavenm']-float(parts[1]))<0.01]
        if line_array.empty:
            updated_lines.append(line)
            continue
        abund = float(line_array[objName])
        updated_value = float(parts[-1]) + abund
        updated_line = '\t'.join(parts[:-1]) + f'\t{updated_value:.2f}\n'
        updated_lines.append(updated_line)

    with open(f"mySampleAll/output/{objName}/{objName}_res_IndivAbund_NLTE.txt", "w") as f:
        f.writelines(lines[0])
        f.writelines(updated_lines)

def NLTEErrUpdate():
    """
    Recalculates and updates the total error of NLTE (Non-Local Thermodynamic Equilibrium) abundances 
    for each chemical element in a stellar abundance analysis.

    This function reads two input files:
    1. An error matrix file containing preliminary error estimates for each element.
    2. A file with individual NLTE abundance values per element.

    For each element listed in the error matrix file (excluding the header), the function:
    - Extracts corresponding abundance values from the individual abundances file.
    - Computes the mean abundance and the standard deviation (used as the statistical error).
    - Recalculates the total error by combining the original total error, subtracting the fixed 
      systematic component, and adding the squared standard deviation.
    - Stores the updated abundance and recalculated total error in a new output file.

    Output:
    Writes the updated results to a file named '{objName}_FinalErr_NLTE.txt' located in the 
    'mySampleAll/output/{objName}/ErrCalc/' directory. This file contains:
    - A header line: 'element\t[X/H]\tdtotal'
    - One line per element with updated abundance and error in tab-separated format.

    Notes:
    - Assumes that 'objName' is a globally defined variable representing the name of the object being analyzed.
    - Uses 'np.nanmean' and 'np.nanstd' to handle potential NaN values in abundance arrays.
    - Logs the operation with an informational message.

    Requirements:
    - The NumPy and logging libraries must be imported.
    - The 'objName' variable must be defined in the global scope prior to calling this function.

    Example file formats:
    - '{objName}_ErrMatrix.txt': <element>\t<...>\t<fixed error>\t...\t<total error>
    - '{objName}_res_IndivAbund_NLTE.txt': <element>\t<...>\t<...>\t<...>\t<abundance>

    Raises:
    - FileNotFoundError if input files are missing.
    - ValueError if file contents are not in expected format or contain non-numeric values.
    """
    logging.info("RE-CALCULATING THE ERRORS OF NLTE ABUNDANCES")
    with open(f"mySampleAll/output/{objName}/ErrCalc/{objName}_ErrMatrix.txt", 'r') as file:
        lines = file.readlines()
    with open(f"mySampleAll/output/{objName}/{objName}_res_IndivAbund_NLTE.txt", 'r') as file:
        abunds = file.readlines()

    updated_lines = []
    for line in lines[1:]:
        parts = line.split('\t')
        elem_name = parts[0]
        elem_array = [float(abund.split('\t')[4]) for abund in abunds if abund.split('\t')[0] == elem_name]
        if elem_array:  # Check if the array is not empty
            abund = np.nanmean(elem_array)
            error = np.nanstd(elem_array)
            total_error = np.sqrt(float(parts[-1])**2 - float(parts[2])**2 + error**2)
            updated_line = f'{parts[0]}\t{abund:.2f}\t{total_error:.2f}\n'
            updated_lines.append(updated_line)

    with open(f"mySampleAll/output/{objName}/ErrCalc/{objName}_FinalErr_NLTE.txt", "w") as f:
        f.writelines('element\t[X/H]\tdtotal\n')
        f.writelines(updated_lines)

def NLTECorrAdditionPars(linemasks): # Version with parameter re-estimation
    """
    Applies NLTE (Non-Local Thermodynamic Equilibrium) abundance corrections to a list of spectral line masks.

    This function modifies the input 'linemasks' by updating their equivalent widths ('ew') and 
    logarithmic reduced equivalent widths ('ewr') based on NLTE abundance corrections. It reads 
    correction data from an external fixed-width formatted file ('mySampleAll/output/result_fe.txt') 
    and adjusts the 'ewr' value by adding the appropriate NLTE abundance. The corresponding 'ew' is 
    recalculated using the updated 'ewr'.

    Parameters:
    -----------
    linemasks : list of dict
        A list of dictionaries where each dictionary represents a spectral line and contains at least
        the following keys:
            - 'element' : str
                The chemical element symbol (e.g., 'Fe1', 'Fe2') with a trailing charge state.
            - 'wave_nm' : float
                The wavelength of the line in nanometers.
            - 'wave_A' : float
                The wavelength of the line in Angstroms.
            - 'ewr' : float
                The logarithmic reduced equivalent width.
            - 'ew' : float
                The equivalent width (in mÅ) to be updated.

    Returns:
    --------
    list of dict
        The modified list of line masks with NLTE-corrected 'ewr' and recalculated 'ew'.

    Notes:
    ------
    - Lines for which no matching element or wavelength is found in the correction file are skipped.
    - A global variable 'objName' is expected to be defined elsewhere in the code. It is used to extract
      the abundance correction value from the correction file for a specific object.
    - The correction file must contain columns: 'el' (element symbol), 'wavenm' (wavelength in nm), 
      and a column corresponding to 'objName' which holds the abundance correction.
    - The function prints diagnostic information before and after applying corrections.

    Raises:
    -------
    KeyError:
        If any of the expected dictionary keys are missing from a 'linemasks' entry.
    FileNotFoundError:
        If the file 'mySampleAll/output/result_fe.txt' cannot be found.
    """
    logging.info("CORRECTING LINEMASKS FOR NLTE EFFECTS")
    import pandas as pd
    df = pd.read_fwf("mySampleAll/output/result_fe.txt", sep='\t', header=0, index_col=False)
    df.replace([-1000.0], np.nan, inplace=True)

    for line in linemasks:
        elem_array = df[df['el']==line['element'][:-2]]
        if elem_array.empty:
            continue
        line_array = elem_array[abs(elem_array['wavenm']-line['wave_nm'])<=0.01]
        if line_array.empty:
            continue
        abund = float(line_array[objName])
        print(f"BEFORE: {line['wave_nm']}, {line['ew']}, {line['ewr']}")
        line['ewr'] += abund
        line['ew'] = line['wave_A']*1000.*10**(line['ewr'])
        print(f"AFTER: {line['wave_nm']}, {line['ew']}, {line['ewr']}")
    return(linemasks)

def CorrectLoggfValues(linemasks): #To update with recent NIST data, to correct for self-blends (effective loggf)
    """
    Applies empirical corrections to log(gf) values for specific spectral lines based on known 
    self-blend effects shown in atomic data from NIST.

    This function iterates through a list of spectral line dictionaries and adjusts their 'loggf' 
    values if a matching element and wavelength (within a specified tolerance) is found in the 
    correction database. These corrections help account for discrepancies due to line blending or 
    outdated oscillator strength (log(gf)) values.

    Parameters:
    -----------
    linemasks : list of dict
        A list of dictionaries, each representing a spectral line. Each dictionary must contain:
            - 'element': str, chemical element and ionization stage (e.g., 'Fe 1', 'Ca 2')
            - 'wave_nm': float, central wavelength of the spectral line in nanometers
            - 'loggf' : float, oscillator strength (log(gf)) value to be corrected

    Returns:
    --------
    list of dict
        The updated list of spectral line dictionaries, with corrected 'loggf' values for lines 
        that match the correction criteria.

    Notes:
    ------
    - The correction database includes empirically determined offsets for specific lines, intended 
      to reflect effective log(gf) values in the presence of self-blends or to match updated atomic data.
    - Wavelength matches are determined within a fixed tolerance of ±0.01 nm.
    - Corrections are additive: corrected_loggf = original_loggf + correction.

    Example:
    --------
    >>> linemasks = [{'element': 'O 1', 'wave_nm': 615.5971, 'loggf': -0.45}]
    >>> corrected = CorrectLoggfValues(linemasks)
    >>> corrected[0]['loggf']
    -0.103  # (i.e., -0.45 + 0.347)
    """
    corrections = {
        ('C 1', 711.5170): 0.114,
        ('O 1', 553.0741): 0.113,
        ('O 1', 615.5971): 0.347,
        ('O 1', 615.6778): 0.252,
        ('O 1', 615.8187): 0.113,
        ('Na 1', 568.8205): 0.046,
        ('S 1', 458.9261): -0.087,
        ('S 1', 469.4113): 0.061,
        ('S 1', 469.5443): 0.048,
        ('S 1', 469.6252): 0.039,
        ('S 1', 527.8700): 0.200,
        ('S 1', 605.2656): 0.157,
        ('S 1', 674.8790): 0.103,
        ('S 1', 869.4710): 0.312,
        ('Cu 1', 570.0237): 0.253,
        ('Na 1', 568.8205): 0.046,
        ('Mg 1', 517.2684): 0.057,
        ('Mg 1', 518.3604): 0.072,
        ('Si 1', 864.8465): -0.088,
        ('Si 2', 637.1371): -0.042,
        ('Ca 1', 428.3011): -0.088,
        ('Ca 1', 431.8652): -0.070,
        ('Ca 1', 558.8749): -0.148,
        ('Ca 1', 559.8480): -0.143,
        ('Ca 1', 643.9075): 0.080,
        ('Co 1', 481.3476): 0.010,
        ('Eu 2', 664.5094): 0.282
    }
    wave_delta = 0.01
    for line in linemasks:
        key = (line['element'], line['wave_nm'])
        for (element, wave_nm), correction in corrections.items():
            if element == key[0] and abs(wave_nm - key[1]) < wave_delta:
                line['loggf'] += correction
    return(linemasks)



def StepReduc(objName, linelist_created=1):
    """
    Perform the initial reduction of a stellar spectrum including continuum fitting,
    cosmic ray filtering, and radial velocity correction.

    Parameters:
    ----------
    objName : str
        The name of the object/star whose spectrum is to be processed.
    linelist_created : int, optional (default=1)
        Flag indicating whether the line list has already been created (1) or not (0).
        If not, a new list will be created based on the spectral range.

    Returns:
    -------
    tuple
        - star_spectrum : dict
            Dictionary containing the processed spectral data.
        - star_continuum_model : dict
            Continuum model used during spectrum normalization.
        - model_atmospheres : str
            Path to the model atmosphere directory used for the analysis.
        - rv : float
            Measured radial velocity of the star.
        - rv_err : float
            Estimated error in the radial velocity measurement.
    """
    star_spectrum = ispec.read_spectrum(f'{mySamIn_dir}{objName}.txt')
    model_atmospheres = ispec_dir + "input/atmospheres/ATLAS9.KuruczODFNEW/"
    if objName=="CCLyr":
        model_atmospheres = ispec_dir + "input/atmospheres/MARCS.GES/"
    if linelist_created == 0:
        ListCreation(np.min(star_spectrum['waveobs']), np.max(star_spectrum['waveobs']), model_atmospheres)
    #star_spectrum = ContFitAndNorm(star_spectrum)
    star_continuum_model = ispec.fit_continuum(star_spectrum, fixed_value=1.0, model="Fixed value")
    star_spectrum = CosmicFilter(star_spectrum, star_continuum_model)
    star_spectrum, rv, rv_err = RVCorr(star_spectrum)
    #star_spectrum, snr = SNRErrCalc(star_spectrum)
    SaveSpec(star_spectrum, "CleanSpec")
    return(star_spectrum, star_continuum_model, model_atmospheres, rv, rv_err)

def StepFind(star_spectrum, star_continuum_model, model_atmospheres, rv, rv_err, FeCNO=1):
    """
    Identify suitable absorption lines in the spectrum for abundance analysis.

    Parameters:
    ----------
    star_spectrum : dict
        Dictionary containing the processed stellar spectrum.
    star_continuum_model : dict
        Continuum model fitted to the stellar spectrum.
    model_atmospheres : str
        Path to the model atmosphere grid used in analysis.
    rv : float
        Radial velocity of the star.
    rv_err : float
        Error in the radial velocity measurement.
    FeCNO : int, optional (default=1)
        Flag for including Fe, C, N, and O lines during line selection.

    Returns:
    -------
    linemasks : list
        List of dictionaries representing spectral lines identified for analysis.
    """
    linemasks = LineFit(star_spectrum, star_continuum_model, model_atmospheres, rv, rv_err, FeCNO, mode="seek")
    return(linemasks)

def StepFilter(star_spectrum, star_continuum_model, model_atmospheres, rv, rv_err):
    """
    Refine and correct the line list by adjusting line parameters and estimating initial abundances.

    Parameters:
    ----------
    star_spectrum : dict
        Dictionary containing the processed stellar spectrum.
    star_continuum_model : dict
        Continuum model fitted to the stellar spectrum.
    model_atmospheres : str
        Path to the model atmosphere grid used in analysis.
    rv : float
        Radial velocity of the star.
    rv_err : float
        Error in the radial velocity measurement.

    Returns:
    -------
    linemasks : list
        Updated line list with corrected oscillator strengths and refined line parameters.
    """
    linemasks = LineFit(star_spectrum, star_continuum_model, model_atmospheres, rv, rv_err, mode="tweak")
    linemasks = CorrectLoggfValues(linemasks)
    params, errors = EWparam(star_spectrum, star_continuum_model, linemasks, model_atmospheres, code="moog")
    abunds = EWabund(star_spectrum, star_continuum_model, linemasks, model_atmospheres, code="moog")
    return(linemasks)

def StepStud(star_spectrum, star_continuum_model, model_atmospheres, rv, rv_err):
    """
    Finalize line selection and compute stellar parameters and elemental abundances.

    Parameters:
    ----------
    star_spectrum : dict
        Dictionary containing the processed stellar spectrum.
    star_continuum_model : dict
        Continuum model fitted to the stellar spectrum.
    model_atmospheres : str
        Path to the model atmosphere grid used in analysis.
    rv : float
        Radial velocity of the star.
    rv_err : float
        Error in the radial velocity measurement.

    Returns:
    -------
    tuple
        - linemasks : list
            Final list of selected and corrected lines.
        - params : dict
            Estimated stellar atmospheric parameters (e.g., Teff, logg, [Fe/H], etc.).
        - errors : dict
            Estimated errors on the stellar parameters.
        - abunds : dict
            Computed elemental abundances for the selected lines.
    """
    linemasks = LineFit(star_spectrum, star_continuum_model, model_atmospheres, rv, rv_err, mode="pick")
    linemasks = CorrectLoggfValues(linemasks)
    params, errors = EWparam(star_spectrum, star_continuum_model, linemasks, model_atmospheres, code="moog")
    WriteErrCalcBest(linemasks, model_atmospheres)
    abunds = EWabund(star_spectrum, star_continuum_model, linemasks, model_atmospheres, code="moog")
    return(linemasks, params, errors, abunds)

def Step4NLTE():
    """
    Prepare model atmosphere interpolation for NLTE (Non-Local Thermodynamic Equilibrium) corrections.

    This function should be run prior to applying NLTE corrections to ensure
    appropriate atmospheric models are available.
    """
    interpolate_atmosphere()

def StepNLTE(star_spectrum, star_continuum_model, linemasks, abunds):
    """
    Apply NLTE corrections to the line masks and update associated errors.

    Parameters:
    ----------
    star_spectrum : dict
        Dictionary containing the processed stellar spectrum.
    star_continuum_model : dict
        Continuum model fitted to the stellar spectrum.
    linemasks : list
        List of spectral lines to be corrected for NLTE effects.
    abunds : dict
        Initial elemental abundances prior to NLTE correction.

    Notes:
    -----
    This step modifies line masks and updates internal data structures to reflect NLTE-corrected quantities.
    """
    NLTECorrAddition()
    NLTEErrUpdate()

def StepErr(star_spectrum, star_continuum_model, model_atmospheres, linemasks, rv, params, errors, nlte=False):
    """
    Perform error estimation on the final derived parameters and abundances.

    Parameters:
    ----------
    star_spectrum : dict
        Dictionary containing the processed stellar spectrum.
    star_continuum_model : dict
        Continuum model fitted to the stellar spectrum.
    model_atmospheres : str
        Path to the model atmosphere grid used in analysis.
    linemasks : list
        Final line list used for parameter and abundance derivation.
    rv : float
        Radial velocity of the star.
    params : dict
        Estimated stellar atmospheric parameters.
    errors : dict
        Errors associated with the stellar parameters.
    nlte : bool, optional (default=False)
        Flag indicating whether NLTE corrections should be included in the error estimation.

    Returns:
    -------
    None
        This function performs calculations and saves output but does not return data directly.
    """
    linemasksErr = ErrStud(star_spectrum, star_continuum_model, model_atmospheres, linemasks, rv, 
                params, errors, nlte=nlte)
    TotalErrCalc(nlte=nlte)

def main():
    """
    Executes the main pipeline for stellar spectral analysis, including spectral modeling, 
    radial velocity correction, line mask generation, abundance determination, and error estimation.
    
    This function orchestrates the overall workflow to analyze a star's observed spectrum 
    using various processing steps. It optionally includes non-local thermodynamic equilibrium (NLTE) 
    corrections based on the 'nlte' flag.

    Workflow:
        1. Calls 'StepReduc' to reduce and prepare the stellar spectrum and continuum model, 
           retrieve model atmospheres, and compute radial velocity (RV) and its error.
        2. Calls 'StepStud' to study the stellar spectrum, extracting line masks, atmospheric 
           parameters, errors, and elemental abundances.
        3. Based on the 'nlte' flag:
            - If 'False', calls 'StepErr' to compute uncertainties in the parameters, and 
              'DoAllLists' to finalize and compile results.
            - If 'True', applies NLTE corrections using 'StepNLTE', and then recomputes errors 
              and finalizes the analysis with 'DoAllLists'.

    Notes:
        - Commented-out functions ('StepFind', 'StepFilter', 'Step4NLTE') suggest alternate or 
          additional analysis routes that are not active in the current workflow.
        - Assumes all necessary modules and data structures ('objName', 'StepReduc', etc.) are 
          correctly defined and imported in the surrounding context.

    Returns:
        None
    """
    nlte=False
    star_spectrum, star_continuum_model, model_atmospheres, rv, rv_err = StepReduc(objName, linelist_created=1)
    #linemasks = StepFind(star_spectrum, star_continuum_model, model_atmospheres, rv, rv_err, FeCNO=1)
    #linemasks = StepFilter(star_spectrum, star_continuum_model, model_atmospheres, rv, rv_err)
    linemasks, params, errors, abunds = StepStud(star_spectrum, star_continuum_model, model_atmospheres, rv, rv_err)
    #Step4NLTE()
    if not nlte:
        StepErr(star_spectrum, star_continuum_model, model_atmospheres, linemasks, rv, params, errors)
        DoAllLists() #To be called with any ONE target
    else:
        StepNLTE(star_spectrum, star_continuum_model, linemasks, model_atmospheres, abunds)
        StepErr(star_spectrum, star_continuum_model, model_atmospheres, linemasks, rv, params, errors, nlte)
        DoAllLists(nlte) #To be called with any ONE target

if __name__ == '__main__':
    main()
    pass
