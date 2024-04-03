# E-iSpec
A modified version of iSpec to derive elemental abundances and isotopic ratios of evolved stars. For the full description, please see Mohorian et al. (2024).

###########################################################################    

___TO RUN A NEW STAR YOU NEED TO:___

    a) add the source's spectrum ("waveobs\tflux\terr\n") and the line regions ("wave_peak\twave_base\twave_top\tnote\n") to '.../mySample/input/' folder,
    
    b) add literature stellar parameters for your object in 'Spoiler.txt' (first seven symbols was enough for star IDs in my samples, e.g. "J004441"),
    
    c) open terminal in the root folder and type 'python3 E-iSpec_atomic_script.py 7-symbol-star-ID' (for example, "J004441").

###########################################################################

___ROOT FOLDER STRUCTURE___
***iSpec has two versions: GUI and Python. They will be addressed as GUISpec and PySpec, respectively.***
Root folder:

    a) images/                         = iSpec logos (PNG and GIF),
    b) input/                          = iSpec input files: solar abundances, model atmospheres, model atmosphere grid and minigrid (SPECTRUM, MARCS, GES), Yonsei-Yale isochrones, list of isotopes, line lists, and template spectra,
    c) isochrones/                     = Yonsei-Yale isochrone plotter,
    d) ispec/                          = all PySpec essentials,
    e) synthesizer/                    = radiative transfer codes,
    f) .git/                           = iSpec git-related folder,
    g) mySample/                       = input and output files for E-iSpec,
    h) dev-requirements.txt            = Python requirements,
    i) example.py                      = different examples of working with the PySpec functions,
    j) E-iSpec_manual_normalisation.py = a script to normalise spectrum manually,
    k) E-iSpec_atomic_script.py        = the main E-iSpec script,
    l) interactive.py                  = GUISpec essential,
    m) ispec.log                       = a concatenation of all the terminal logs (only iSpec-specific logs),
    n) iSpec.command                   = a bash script invoking GUISpec,
    o) LICENSE                         = iSpec license,
    p) Makefile                        = subroutines initial builder (check iSpec installation manual),
    q) pytest.ini                      = Python path testing,
    r) README.md                       = iSpec readme,
    s) requirements.txt                = Python-related requirements,
    t) Spoiler.txt                     = tabulated literature data: 7-symbol-IDs, stellar parameters, and comments,
    u) test.command                    = testing iSpec script for Procyon, the Sun, etc.
    v) .gitignore                      = iSpec git-related list of folders and files.
