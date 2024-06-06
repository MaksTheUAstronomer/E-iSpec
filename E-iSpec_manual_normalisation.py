# -*- coding: utf-8 -*-
# 
# Interactive python code to first point interactively to the continuum points in the 
# graphical display, fit a 5D polynomial through the datapoints and save the file
# with underscore _norm.fits
# 
# python NormaliseManually.py -i <filename.fits>
#
#
import matplotlib
matplotlib.use('TkAgg')

from matplotlib import pylab as plt
import astropy.io.fits as pyfits
import math
import numpy
import scipy.interpolate
import sys
import os

# START INPUT. When used not in folder, give the full path
def write_txt_file(file_path, data):
    with open(file_path, 'w') as f:
        f.write('Wavelength (nm)\tFlux\t Error\n')
        # Assuming each row of data is a space-separated string
        for row in data:
            f.write('\t'.join(map(str, row)) + '\n')

ar = len(sys.argv) - 1
if(ar > 0):
    args = sys.argv          
    for k in numpy.arange(1, len(args)):
        if(args[k] == "-i"):
            inputfile = sys.argv[k+1]
            outputfileflux = inputfile.strip('.txt') + "_norm.txt"
            outputfilecurve = inputfile.strip('.txt') + "_response.txt"
            print(inputfile, outputfileflux)
else:
   #   no parameters received from the command line, give help.
   print("USAGE  : python E-iSpec_manual_normalisation.py -i inputfile.txt")
   print(" ")


#-- switch on interactive mode (for plots)
#plt.ion()

#This is a function definition to fit a 5D polynomial through the chosen continuum points.
#The polynomial will have the same range and sampling as the inputspectrum so the 
# normalisation can be defined.

def Polyfit5(wavlist,fluxlist):
  poly = numpy.polyfit(wavlist,fluxlist,5)
  
  #Filling in the indices of the fit
  curve = []
  for i in range(len(wavlist)):
    g = poly[0]*wavlist[i]**5 + poly[1]*wavlist[i]**4 + poly[2]*wavlist[i]**3 + poly[3]*wavlist[i]**2 + poly[4]*wavlist[i] + poly[5]
    curve.append(g)
  
  return(curve)

# Function definition to rebin on a wavelength-array.
def rebin_spectrum(wvl_org, flux_org, wvl_new):
  s = scipy.interpolate.interp1d(wvl_org, flux_org, kind='linear', bounds_error=False, fill_value=1.0)
  flux_new = s(wvl_new)
  return(flux_new)

def WriteTxtFile(wave, normflux, outputfile):
    # Stack the arrays horizontally
    error = numpy.zeros_like(wave)
    output_data = numpy.column_stack((wave, normflux, error))
    
    # Write to text file
    output_txt_file = outputfile.strip('.txt') + ".txt"
    write_txt_file(output_txt_file, output_data)
    
    print(f"{output_txt_file} saved")

#{ Tools
def fit_splines():

    coords = variables['coords']
    
    #-- sort coordinates according to x value, and make separate arrays from the
    #   x and y coordinate
    xs = numpy.array([coord[0] for coord in coords])
    sortarr = numpy.argsort(xs)
    ys = numpy.array([coord[1] for coord in coords])[sortarr]
    xs = xs[sortarr]
    
    #-- fit fifth order polynomial through the points
    curve = Polyfit5(xs,ys)
    
    #-- interpolate the polynomial to the ral wavelength region
    normcurve = rebin_spectrum(xs,curve,wave)    
        
    #-- Normalise the spectrum by dividing the fluxes by the continuum curve.
    normflux = []
    for i in range(len(wave)):
      x = flux[i]/normcurve[i]
      normflux.append(x)
    
    #-- plot the polynomial
    plt.figure(1)
    plt.plot(wave,normcurve,'r-')
    
    w = plt.get_fignums()
    if(2 in w):
      plt.figure(2)
      plt.clf()
    
    plt.figure(2)
    plt.title('Normalised flux')
    plt.xlabel('Wavelength (Angstrom)')
    plt.ylabel('Arbitrary Flux')
    plt.plot(wave,normflux,'k-')
    plt.axhline(y=1,color='r')
    
    plt.show()
    
    return(normflux,normcurve)
#}

#Global parameter used to plot when a coordinate is removed
normc = []

#{ User interaction
def ontype(event):
    """
    Actions linked to key strokes (only normalisation).
    """

    global normc
    
    #-- if the user presses enter, the polynomial is fitted to the marked coordinates and the spectrum is normalised and shown
    if event.key=='enter':
        normflux,normcurve = fit_splines()
        normc = normcurve
        print('No files are saved with this key')
    
    #-- if the user presses g, the normalised flux and normalisation curve are both saved to the desired output fitsfiles    
    if event.key=='g':
       if(outputfileflux == ''):
           print('No outputfile name for the normalised flux hence no outputfile created')
       else:
           normflux,normcurve = fit_splines()
           #error = numpy.ones_like(inputfile)  # Replace this with your error array
           WriteTxtFile(wave, normflux, outputfileflux)
           print('Normalised output flux written to ' + str(outputfileflux))
       if(outputfilecurve == ''):
           print('No outputfile name for the normalisation curve hence no outputfile created')
       else:
           normflux,normcurve = fit_splines()
           WriteTxtFile(wave, normcurve, outputfilecurve)
           print('Normalisation output curve written to ' + str(outputfilecurve))
           plt.figure(1)
           plt.show()
      
    #-- if the user presses q, the program ends. 
    if event.key=='q':
        print('End of program')
        sys.exit()

    #-- if the user presses r, the plot is refreshed and the user can start over
    if event.key=='r':
        print('Refreshing the original flux plot, all previous coordinate data is deleted')
        variables['coords'] = []
        normc = []
        plt.figure(1)
        plt.clf()
        plt.title('Original flux')
        plt.xlabel('Wavelength (Angstrom)')
        plt.ylabel('Arbitrary flux')
        plt.plot(variables['wave'],variables['flux'],'k-')

        plt.show()

    #-- if the user presses j, the output fitsfiles are plotted. If no files are yet outputed, the program gives a message
    if event.key=='j':
       just = os.path.isfile(outputfileflux)
       if(just == True):
         spectrum_file4 = numpy.loadtxt(outputfileflux)
         spec_flux4 = data[:, 1]
         spec_wvl4 = data[:, 0]
         
         normwavje = spec_wvl4
         normfluxje = spec_flux4
         
         w = plt.get_fignums()
         if(4 in w):
           plt.figure(4)
           plt.clf()
    
         plt.figure(4)
         plt.title('Checkup Normalised flux')
         plt.xlabel('Wavelength (Angstrom)')
         plt.ylabel('Arbitrary Flux')
         plt.plot(normwavje,normfluxje,'b-')
         plt.axhline(y=1,color='r')         
          
         plt.show()
       
       else:
          print('No outputfiles have been saved yet. Press "f" to create the outputfiles.')
       
       just = os.path.isfile(outputfilecurve)
       if(just == True):
         spectrum_file3 = numpy.loadtxt(outputfilecurve)
         spec_flux3 = data[:, 1]
         spec_wvl3 = data[:, 0]
         
         curvwavje = spec_wvl3
         curvfluxje = spec_flux3
         
         w = plt.get_fignums()
         if(3 in w):
           plt.figure(3)
           plt.clf()
         
         plt.figure(3)
         plt.title('Checkup Original flux')
         plt.xlabel('Wavelength (Angstrom)')
         plt.ylabel('Arbitrary flux')
         plt.plot(variables['wave'],variables['flux'],'b-')
         plt.plot(curvwavje,curvfluxje,'r-')
        
         plt.show()
        
       plt.figure(1) 
         
       plt.show()   
        
    #-- if the user presses h, the help commentary is printed
    if event.key=='m':
        print('This is an overview of the approved event keys:')
        print('')
        print('"m"      gives the help menu')
        print('"g"      saves the normalised flux and normalisation curve to the desired output fitsfiles ')
        print('"q"      ends program')
        print('"r"      refreshes the original flux plot (figure 1)')
        print('"enter"  fits 5D the polynomial to the marked coordinates and the spectrum is normalised and shown')
        print('"j"      plots the output fitsfiles. If no files are yet outputed, the program gives a message')
        print('\n')
        print('MOUSE CLICKS')
        print('"Left"   select coordinates for the 5D polynomial fitting')
        print('"Middle" fits 5D the polynomial to the marked coordinates and the spectrum is normalised and shown')
        print('"Right"  remove a selected coordinate')
        
        plt.show()

def onclick(event):
    """
    Actions on mouse clicks (selected and deselecting coordinates, and fitting).
    The actions work as follows: 
	the left button of the mouse is used to select points of the continuum = event.button 1
	the right button of the mouse is used to remove selected points = event.button 3
	the middle button of the mouse is used to fit the spline = event.button 2
    """
    
    global normc
    
    tb = plt.get_current_fig_manager().toolbar
    base_width = 1. # for median filtering
    x,y = event.xdata,event.ydata # extract clicked points

    #-- select spline-fitting points.
    if event.button==1 and event.inaxes and tb.mode == '':
        wave,flux = variables['wave'],variables['flux']
         
        variables['coords'].append((x,y))
        print('Added coordinate(%.3f,%.3f)'%(x,y))
        
        #-- plot the selected points in blue squares
        plt.plot([x],[y],'rs',ms=5)
        plt.show()
    
    #-- deselect spline-fitting points
    if event.button==3 and event.inaxes and tb.mode == '':
        #-- remove points if the user clicks on them. "Clicking on" actually
        #   means "Clicking near", which is relative to the zoom level
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        xlimr = xlim[1]-xlim[0]
        ylimr = ylim[1]-ylim[0]
        
        #-- run through all clicked coordinates, and see if any of them are near
        #   the clicked coordinates
        i = 0
        while i < len(variables['coords']):
            coord = variables['coords'][i]
            xi,yi = tuple(coord)
            distance = math.sqrt( (x-xi)**2/xlimr**2 + (y-yi)**2/ylimr**2)
            if distance<=0.01:
                print('Removed coordinate(%.3f,%.3f)'%(xi,yi))
                variables['coords'].remove(coord)
                plt.clf()
                plt.plot(variables['wave'],variables['flux'],'k-')
                for i in range(len(variables['coords'])):
                   coord = variables['coords'][i]
                   xi,yi = tuple(coord)
                   plt.plot([xi],[yi],'rs',ms=5)
                if(normc != []):
                   plt.plot(variables['wave'],normc,'r-')
                   plt.xlim(xlim[0],xlim[1])
                   plt.ylim(ylim[0],ylim[1])
                   plt.show()     
                break
            i+=1
        else:
            print("No close coordinate found")
    
    #-- fit and plot the splines
    if event.button==2:
        normflux,normcurve = fit_splines()
        normc = normcurve
        print('Not yet saved !!')
#}

if __name__=="__main__":
    """
    Interpret the command line argument as a file name, build the GUI and let
    the user do all the work...
    """
    #-- read in the file if possible, otherwise exit the program
     
    try:
        data = numpy.loadtxt(inputfile, skiprows=1)
        spec_wvl = data[:, 0]
        spec_flux = data[:, 1]
    except Exception as e:
        print(f"Error reading the file: {e}")
        sys.exit()    
    
    wave,flux = spec_wvl,spec_flux
    
    
    #-- This part selects all wavelength points
    region = abs(wave-wave[0])>=0
    #-- Here we define the variables for wavelengths and fluxes.
    variables = {'wave':wave[region],'flux':flux[region]}

    #-- keep track of points to use for spline fitting (none so far)
    variables['coords'] = []

    #-- plot the spectrum!
    plt.figure(1)
    plt.title('Original flux')
    plt.xlabel('Wavelength (Angstrom)')
    plt.ylabel('Arbitrary flux')
    
    plt.plot(variables['wave'],variables['flux'],'k-')

    print('##################################################################################################')
    print('This is an overview of the approved event keys:')
    print('')
    print('"m"     gives the help menu')
    print('"g"     saves the normalised flux and normalisation curve to the desired output fitsfiles ')
    print('"q"     ends program')
    print('"r"     refreshes the original flux plot (figure 1)')
    print('"enter" fits 5D the polynomial to the marked coordinates and the spectrum is normalised and shown')
    print('"j"     plots the output files. If no files are yet outputed, the program gives a message'  )   
    print('##################################################################################################')

    ###-- add functionality
    plt.gcf().canvas.mpl_connect('key_press_event',ontype)
    plt.gcf().canvas.mpl_connect('button_press_event',onclick)
    
    plt.show()
