#!/usr/bin/env python
# The pySME showcasing functions were written by Mingjie Jian. Later, this code was completely re-structured
# by Maksym Mohorian to be used as part of E-iSpec (Mohorian et al. 2024, 2025a, 2025b).

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import gc

from pysme.sme import SME_Structure
from pysme.abund import Abund
from pysme.linelist.vald import ValdFile
from pysme.synthesize import synthesize_spectrum
from pysme.solve import solve

def integrate_EW(wave, flux):
    profile_sum = 0.
    for i in range(len(wave)-1):
        profile_sum += (1-flux[i])*(wave[i+1]-wave[i])
    return profile_sum*1000. # converting A into mA

def read_atmo_params(target_name, target_elem):
    df = pd.read_csv("MasterParamList_Extreme.txt", sep='\t', header=0, index_col=False)
    teff = float(df["Teff"][df["Name"]==target_name].iloc[0])
    logg = float(df["logg"][df["Name"]==target_name].iloc[0])
    monh = float(df["[Fe/H]"][df["Name"]==target_name].iloc[0])
    vmic = float(df["Vmic"][df["Name"]==target_name].iloc[0])
    vmac = 10.
    vsini = 0.
    if target_elem=='Fe':
        elemonh = monh
    else:
        elemonh = 0.
    del df
    return(teff, logg, monh, vmic, vmac, vsini, elemonh)

def synthesise_lte(target_elem, target_wav, target_ew, teff, logg, monh, vmic, vmac, vsini, \
                   elemonh, linelist, margin, resol, s_n):
    # Set up the SME structure for spectral synthesis
    sme = SME_Structure()
    if target_elem=='Fe':
        print(f"A(Fe) = {sme.abund['Fe']:.2f} dex <--- Solar")
    else:
        print(f"A({target_elem}) = {sme.abund[target_elem]:.2f} dex, A(Fe) = {sme.abund['Fe']:.2f} dex <--- Solar")
    # Input stellar parameters
    sme.teff, sme.logg, sme.monh, sme.vmic, sme.vmac, sme.vsini = teff, logg, monh, vmic, vmac, vsini
    # Input instrumental broadening
    sme.iptype = 'gauss'
    sme.ipres = resol
    # Input wavelength grid
    sme.wave = np.arange(target_wav - margin, target_wav + margin, 0.02)
    # Input linelist
    #print(help(ValdFile))
    #print(linelist_abridged)
    sme.linelist = linelist
    # These two parameters control the synthetic accuracy, would be useful for very weak lines 
    #       (but still they may go wrong).
    sme.accwi = 0.00001
    sme.accrt = 0.00001
    # Set the individual abundances.
    if target_elem=='Fe':
        print(f"A(Fe) = {sme.abund['Fe']:.2f} dex <--- Metallicity-Scaled Solar")
    else:
        print(f"A({target_elem}) = {sme.abund[target_elem]:.2f} dex, " + \
              f"A(Fe) = {sme.abund['Fe']:.2f} dex <--- Metallicity-Scaled Solar")
    # +[X/Fe] = +[X/H]-[Fe/H]; -monh is necessary to counteract the automatically adding +monh
    # Synthesize the LTE spectrum.
    sme = synthesize_spectrum(sme)

    # Correct for observed EW.
    summed_ew = integrate_EW(sme.wave[0], sme.synth[0]); print(f'First guess LTE EW = {summed_ew:.1f} mA')
    is_converged = False
    while (not is_converged):
        if target_elem=='Fe':
            sme.monh += np.round(np.log10(target_ew/summed_ew), 4)
        else:
            sme.abund[target_elem] += np.round(np.log10(target_ew/summed_ew) -sme.monh, 4)
        sme1 = synthesize_spectrum(sme)
        summed_ew = integrate_EW(sme1.wave[0], sme1.synth[0]); print(f'Iterating LTE EW = {summed_ew:.1f} mA')
        del sme1; gc.collect()
        if abs(target_ew-summed_ew)<5e-2:
            if target_elem=='Fe':
                print(f"Converged for [Fe/H] = {sme.monh:.2f} dex")
            else:
                print(f"Converged for [X/Fe] = {sme.abund:.2f} dex")
            is_converged = True

    #wave_lte, flux_lte = sme.wave[0].copy(), sme.synth[0].copy()
    # Extract the LTE spectra and add some error according to S/N.
    wave_obs, flux_obs = sme.wave[0], sme.synth[0] * (1+np.random.randn(len(sme.synth[0])) / s_n)
    
    # Do the same thing but in NLTE.
    #sme.nlte.set_nlte(target_elem)
    #sme = synthesize_spectrum(sme)
    #wave_nlte, flux_nlte = sme.wave[0].copy(), sme.synth[0].copy()

    del sme; gc.collect()

    return(wave_obs, flux_obs)

def fit_nlte(wave_obs, flux_obs, target_elem, target_wav, teff, logg, monh, vmic, vmac, vsini, linelist, margin, \
             s_n, resol):
    # Set up the SME structure for abundance fitting.
    sme_fit = SME_Structure()
    sme_fit.teff, sme_fit.logg, sme_fit.monh, sme_fit.vmic, sme_fit.vmac, sme_fit.vsini = teff, logg, monh, vmic, vmac, vsini
    sme_fit.iptype = 'gauss'
    sme_fit.ipres = resol
    # Input the 'observed' wavelength grid, flux and flux error.
    sme_fit.wave = wave_obs
    sme_fit.spec = flux_obs
    sme_fit.uncs = flux_obs / s_n
    sme_fit.linelist = linelist
    # Set up NLTE grid.
    # Note: if you have your own NLTE grid (named as 'your_nlte.grd'), 
    #       simply use sme_fit.nlte.set_nlte(target_elem, 'your_nlte.grid') to specify it.
    #       by default we will use the grids listed in https://pysme-astro.readthedocs.io/en/latest/usage/lfs.html.
    if target_elem=='Ti':
        sme_fit.nlte.set_nlte(target_elem, '/media/max/Data/CallofPhDuty/pySME_grids/nlte_Ti_ama51_Aug2024_pysme.grd')
    elif target_elem=='S':
        sme_fit.nlte.set_nlte(target_elem, '/media/max/Data/CallofPhDuty/pySME_grids/nlte_S_ama51_Sep2024_pysme.grd')
    elif target_elem=='Cu':
        sme_fit.nlte.set_nlte(target_elem, '/media/max/Data/CallofPhDuty/pySME_grids/nlte_Cu_caliskan_Oct2024_pysme.grd')
    else:
        sme_fit.nlte.set_nlte(target_elem)
    sme_fit.accwi = 0.00001
    sme_fit.accft = 0.000005

    # Fit the spectra
    if target_elem=='Fe':
        sme_fit = solve(sme_fit, ['monh'])
    else:
        sme_fit = solve(sme_fit, [f'Abund {target_elem}'])

    #with open(f'flags/{"".join(target_elem.split())}_NLTE.txt', 'w') as flags:
    #    [flags.write(f'{sme_fit.linelist.wlcent[i]:.3f}\t{sme_fit.nlte.flags[i]}\n') for i in range(len(sme_fit.linelist.wlcent))]
    
    #wave_nlte, flux_nlte = sme_fit.wave[0].copy(), sme_fit.synth[0].copy()

    #print('Wavelength, NLTE available?')
            #[print(f'{sme_fit.linelist.wlcent[i]}, {sme_fit.nlte.flags[i]}') for i in range(len(sme_fit.nlte.flags))]

    if sme_fit.nlte.flags[np.argmin(abs(sme_fit.linelist.wlcent-target_wav))]:
        print(f'Line {target_wav:.3f} has a non-LTE correction')
        status = 'YES'
    else:
        print(f'Line {target_wav:.3f} does not have a non-LTE correction')
        status = 'NO'
        fit_results = 99.999

    output = [sme_fit.wave[0], sme_fit.synth[0], sme_fit.monh, sme_fit.abund[target_elem], 
              sme_fit.fitresults["fit_uncertainties"][0], status]

    del sme_fit; gc.collect()

    return(output)

def correct_lte(wave_nlte, flux_nlte, target_elem, target_wav, teff, logg, monh, vmic, vmac, vsini, linelist, margin, \
             s_n, resol):
    # Set up the SME structure for abundance fitting.
    sme_corr = SME_Structure()
    sme_corr.teff, sme_corr.logg, sme_corr.monh, sme_corr.vmic, sme_corr.vmac, sme_corr.vsini = teff, logg, monh, vmic, vmac, vsini
    sme_corr.iptype = 'gauss'
    sme_corr.ipres = resol
    # Input the 'observed' wavelength grid, flux and flux error.
    sme_corr.wave = wave_nlte
    sme_corr.spec = flux_nlte
    sme_corr.uncs = flux_nlte / s_n
    sme_corr.linelist = linelist
    sme_corr.accwi = 0.00001
    sme_corr.accft = 0.000005

    # Fit the spectra
    if target_elem=='Fe':
        sme_corr = solve(sme_corr, ['monh'])
    else:
        sme_corr = solve(sme_corr, [f'Abund {target_elem}'])

    #plt.plot(sme_corr.wave[0], sme_corr.synth[0], label='LTE (fitted)')
    #plt.plot(wave_nlte, flux_nlte, label='NLTE')
    #plt.legend()
    #plt.show()

    output = [sme_corr.wave[0], sme_corr.synth[0], sme_corr.monh, sme_corr.abund[target_elem]]

    del sme_corr; gc.collect()

    return(output)

def plot_spectra(monh, fit_monh, abund, fit_abund, fit_fitresults, target_elem_ion, target_wav, \
                 target_ew, target_name, wave_lte, flux_lte, wave_nlte, flux_nlte, wave_corr, flux_corr, \
                 fOut, status):
    target_elem = target_elem_ion.split()[0]
    plt.figure(figsize=(10, 6))

    if target_elem=='Fe':
        plt.title(rf'{target_elem_ion} line @{target_wav:.3f}$\mathrm{{\AA}}$; '+
                  rf'[Fe/H]$_\mathrm{{input}}$={monh:.2f}, [Fe/H]$_\mathrm{{fitted}}$='+
                  rf'{fit_monh:.2f}$\pm${fit_fitresults:.3f}, '+
                  rf'[Fe/H]$_\mathrm{{fitted}}$ - [Fe/H]$_\mathrm{{input}}$={fit_monh-monh:.4f}')
    else:
        plt.title(rf'{target_elem_ion} line @{target_wav:.3f}$\mathrm{{\AA}}$; '+
                  rf'$A_\mathrm{{input}}$({target_elem})={abund:.2f}, '+
                  rf'$A_\mathrm{{fitted}}$({target_elem})={fit_abund:.2f}$\pm$'+
                  rf'{fit_fitresults:.3f}, $A_\mathrm{{fitted}} - '
                  rf'A_\mathrm{{input}}=${fit_abund-abund:.4f}')

    # Do a quick plot for comparison.
    plt.plot(wave_lte, flux_lte, 'k-', label='LTE (original)')
    plt.plot(wave_nlte, flux_nlte, 'r-', label='NLTE fitting of LTE spectrum')
    plt.plot(wave_corr, flux_corr, 'b:,', label='LTE fitting of NLTE spectrum')
    #plt.plot(wave_nlte, flux_nlte, label='NLTE spectrum')

    plt.legend()
    plt.xlabel(r'Wavelength ($\mathrm{\AA}$)');
    plt.ylabel('Normalized flux');
    plt.savefig(f'PNGs/{target_name}/{"".join(target_elem_ion.split())}_'+
                f'{target_wav:.3f}_{target_name}.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    if target_elem=='Fe':
        fOut.write(f"{target_name}\t{target_wav/10.:.4f}\t{target_ew:.1f}\t{fit_monh-monh:.4f}\t{status}\n")
    else:
        fOut.write(f"{target_name}\t{target_wav/10.:.4f}\t{target_ew:.1f}\t{fit_abund-abund:.4f}\t{status}\n")

def main():
    import tracemalloc
    tracemalloc.start()

    df = pd.read_csv("MasterLineList_Extreme.txt", sep='\t', header=0, index_col=False) #full #transition #dustpoor
    target_name = sys.argv[1] # target_names = df_abridged.columns[4:].tolist()
    target_elem_ion = sys.argv[2] # Target element with its ionization stage
    target_elem = target_elem_ion.split()[0]

    linelist_filename = {
            'C 1': "636",
            'N 1': "637",
            'O 1': "638",
            'Na 1': "639",
            'Mg 1': "640",
            'Al 1': "641",
            'Si 1': "642",
            'Si 2': "643",
            'S 1': "644",
            'K 1': "645",
            'Ca 1': "646",
            'Ca 2': "648",
            'Ti 1': "649",
            'Ti 2': "650",
            'Mn 1': "651",
            'Mn 2': "652",
            'Fe 1': "653",
            'Fe 2': "654",
            'Cu 1': "769",
            'Ba 2': "655"
    }

    linelist = ValdFile("lines/MaksymMohorian.002"+linelist_filename[target_elem_ion]) # 'vald_na_4982.list'

    margin = 0.2 # Set the range of wavelength for synthesize (+-margin)
    s_n = 1000. # S/N
    resol = 57000. # Resolution

    df_abridged = df[df['element']==target_elem_ion].reset_index(drop=True)
    del df

    with open(f'corrs/{target_name}/{"".join(target_elem_ion.split())}_{target_name}_NLTE_corrections.txt', 'w') as fOut:
        for i in range(len(df_abridged['wave_nm'])):
            if df_abridged[target_name][i] == "-":
                continue
            target_wav = float(df_abridged['wave_nm'][i])*10.
            target_ew = float(df_abridged[target_name][i])

            linelist_abridged = linelist.trim(target_wav-margin, target_wav+margin)
            if np.min(np.abs(linelist['wlcent']-target_wav))>5.e-2:
                continue

            teff, logg, monh, vmic, vmac, vsini, elemonh = read_atmo_params(target_name, target_elem) # Stellar parameters
            if target_elem_ion[:-2]=='Fe':
                print(f'>>>Now analysing {target_name} at {target_wav:.3f} A (obs EW = {target_ew:.1f} mA)\n' + \
                      f'(Teff = {teff:.0f} K, logg = {logg:.2f} dex, [Fe/H] = {monh:.2f} dex, Vmic = {vmic:.1f} km/s)')
            else:
                print(f'>>>Now analysing {target_name} at {target_wav:.3f} A (obs EW = {target_ew:.1f} mA)\n' + \
                      f'(Teff = {teff:.0f} K, logg = {logg:.2f} dex, [Fe/H] = {monh:.2f} dex, Vmic = {vmic:.1f} km/s, [X/Fe] = {elemonh:.2f} dex)')

            wave_obs, flux_obs = synthesise_lte(target_elem, target_wav, target_ew, teff, logg, monh, vmic, vmac, vsini, elemonh, \
                       linelist_abridged, margin, resol, s_n)
            wave_nlte, flux_nlte, fit_monh, fit_abund, fit_results, status = fit_nlte(wave_obs, flux_obs, target_elem, target_wav, \
                       teff, logg, monh, vmic, vmac, vsini, linelist_abridged, margin, s_n, resol)
            wave_corr, flux_corr, corr_monh, corr_abund = correct_lte(wave_nlte, flux_nlte, target_elem, target_wav, teff, logg, \
                       monh, vmic, vmac, vsini, linelist_abridged, margin, s_n, resol)

            plot_spectra(corr_monh, fit_monh, corr_abund, fit_abund, fit_results, target_elem_ion, target_wav, target_ew, \
                         target_name, wave_obs, flux_obs, wave_nlte, flux_nlte, wave_corr, flux_corr, fOut, status)

            del linelist_abridged; gc.collect()

            #snapshot = tracemalloc.take_snapshot()
            #top_stats = snapshot.statistics('lineno')
            #print("[ Top Memory Consumers ]")
            #for stat in top_stats[:5]:
            #    print(stat)

if __name__=='__main__':
    main()
