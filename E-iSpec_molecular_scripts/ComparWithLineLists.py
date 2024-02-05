import os
import sys
import numpy as np
from scipy import interpolate as ip
from matplotlib import pyplot as plt

objName = sys.argv[1]
delta_nm = 0.05
elems = ['He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ge', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cs', 'Ba', 'Ce', 'Nd', 'Yb', '13C', '15N', '17O', '18O']
###############################################################################
### Populating arrays (abundances to plot, spectral windows and spectra)
idAPOGEE,lAPOGEE,ewAPOGEE,tAPOGEE,cAPOGEE = np.loadtxt("/home/max/CallofPhDuty/iSpec_v20230804/mySample/outputAPOGEE/%7s/%7s_LineList_APOGEE.txt" % (objName, objName), dtype=np.dtype([('element', 'U5'), ('wave_nm',float), ('ew_mA',float), ('type','U1'), ('code',float)]), delimiter='\t', usecols=[0,2,22,25,29], skiprows=1, unpack=True)
idMASSER,lMASSER,ewMASSER,tMASSER,cMASSER = np.loadtxt("/home/max/CallofPhDuty/iSpec_v20230804/mySample/outputAPOGEE/%7s/%7s_LineList.txt" % (objName, objName), dtype=np.dtype([('element', 'U5'), ('wave_nm',float), ('ew_mA',float), ('type','U1'), ('code',float)]), delimiter='\t', usecols=[0,2,22,25,29], skiprows=1, unpack=True)
###############################################################################
### Do the thing
for elem in elems:
    if not os.path.exists("Abunds/Abunds_%s/%s_SynthLines_%s.txt" % (objName, objName, elem)):
        continue
    f = open("Abunds/Abunds_%s/%s_SynthLines_%s.txt" % (objName, objName, elem))
    fLines = f.readlines()
    Lines = []
    for line in fLines:
        if line[0]=='1':
            Lines.append((float(line[0:10]), float(line[10:20])))
    f.close()
    matchL = []; matchID = []
    with open("Abunds/Abunds_%s/%s_Match_%s.txt" % (objName, objName, elem), "w") as outfile:
        outfile.write('SpecWind (nm)\tMatched (nm)\tSource\tType\tID\tIsotopic code\tMSSA EW\n')
        outfile.write('--------------------------------------------------------------------------------\n')
        for line in Lines:
            counter = 0
            print("Checking range %.5f-%.5f nm of %s (%s)" % (float(line[0]), float(line[1]), objName, elem))
            ###################################################################
            ### APOGEE (DR14)
            for i in range(len(lAPOGEE)):
                if lAPOGEE[i]>line[0]-delta_nm and lAPOGEE[i]<line[1]+delta_nm and ewAPOGEE[i]>1.:
                    counter += 1
                    if tAPOGEE[i]=='T':
                        Type='Mol'
                    else:
                        Type='Ato'
                    outfile.write('%.4f\t%.4f\tA\t%s\t%s\t%f\t%.3f\n' % ((line[0]+line[1])/2.,lAPOGEE[i],Type,idAPOGEE[i],cAPOGEE[i],ewAPOGEE[i]))
            ###################################################################
            ### Masseron (DR17)
            for i in range(len(lMASSER)):
                if lMASSER[i]>line[0]-delta_nm and lMASSER[i]<line[1]+delta_nm and ewMASSER[i]>1.:
                    counter += 1
                    if tMASSER[i]=='T':
                        Type='Mol'
                    else:
                        Type='Ato'
                    outfile.write('%.4f\t%.4f\tT\t%s\t%s\t%f\t%.3f\n' % ((line[0]+line[1])/2.,lMASSER[i],Type,idMASSER[i],cMASSER[i],ewMASSER[i]))
            if counter==0:
                outfile.write('%.4f\t-\tA\t-\t-\t-\n' % ((line[0]+line[1])/2.))
            outfile.write('--------------------------------------------------------------------------------\n')
    outfile.close()
