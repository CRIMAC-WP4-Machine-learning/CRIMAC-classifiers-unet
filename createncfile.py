import pickle
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib as npm
from skimage import measure
import h5py
import datetime
import pandas as pd
import pdb
import glob

def createncfile(F, seg, labels, r, mattime, heave, trdepth):
    # Convert from matlab time to milliseconds
    t = time2NTtime(mattime)
    
    #
    # Tidy up data
    #
    
    # Apply logic operatioin to the data to get the binary image
    bin_labels = (labels > 0) * 1   # get either 0 or 1 in the array
    
    # plt.imshow(bin_labels, cmap=plt.cm.gray)  # use appropriate colormap here
    # plt.show()
    
    # Connect pixels to generate unique schools
    all_labels = measure.label(bin_labels)
    
    # Get the school numbers rom the labelling
    schools = [schools for schools in np.unique(all_labels) if schools > 0]
    
    # Open the hdf file
    with h5py.File(F+".nc", "w") as f:
        # Global attributes
        f.attrs['mask_convention_version'] = "0.1"
        f.attrs['date_created'] = "20190819T134900Z"
        f.attrs['mask_convention_name'] = "SONAR-netCDF4"
        f.attrs['mask_convention_authority'] = "ICES, IMR"
        f.attrs['rights'] = "Unrestricted rights"
        f.attrs['license'] = "CC-BY 4.0"
        f.attrs['Conventions'] = "CF-1.7, ACDD-1.3, SONAR-netCDF4-2.0"
        f.attrs['keywords'] = "scrutinisation mask, echosounder"
        f.attrs['summary'] = "Contains definitions of echogram scrutiny masks"
        f.attrs['title'] = "Echogram scrutiny masks"
        
        # Create interpolation group
        intepretation = f.create_group("Interpretation")
        # group v1
        # Subsequent versions of this interpretation get put in new subgroups,
        # using the numbering system v1, v2, etc.
        v1 = f.create_group("Interpretation/v1")
        v1.attrs['Interpretation/v1/version'] = 1
        v1.attrs['Interpretation/v1/version_save_date'] = datetime\
          .datetime.now().isoformat()  # // ISO8601 format
        v1.attrs['Interpretation/v1/version_author'] = "NOH"
        v1.attrs['Interpretation/v1/version_comment'] \
            = "UNET predictions from Brautaset et al. (2010)"
        v1.attrs['Interpretation/v1/mask_times/long_name'] \
            = "Timestamp of each mask point"
        v1.attrs['Interpretation/v1/mask_times/units'] \
            = "milliseconds since 1601-01-01 00:00:00Z"
        v1.attrs['Interpretation/v1/mask_times/axis'] = "T"
        v1.attrs['Interpretation/v1/mask_times/calendar'] = "gregorian"
        v1.attrs['Interpretation/v1/mask_times/standard_name'] = "time"
        v1.attrs['Interpretation/v1/mask_depths/long_name'] \
            = "Depth pairs of mask"
        v1.attrs['Interpretation/v1/mask_depths/units'] = "m"
        v1.attrs['Interpretation/v1/mask_depths/valid_min'] = float(0)
        
        # Create empty dataset for dimension scale definitions
        # (this does not work when testing ncdump)
        # f.create_dataset("Interpretation/v1/channels", dtype=float)
        # f["Interpretation/v1/channels"].make_scale('channels')
        # f.create_dataset("Interpretation/v1/regions", dtype=float)
        # f["Interpretation/v1/regions"].make_scale('regions')
        # f.create_dataset("Interpretation/v1/categories", dtype=float)
        # f["Interpretation/v1/categories"].make_scale('categories')
        # Add if not empty
        
        length_schools = len(schools)
        if length_schools > 0:
            # Create the mask_time data set
            dt = h5py.vlen_dtype(np.dtype('float'))
            mt = f.create_dataset("Interpretation/v1/mask_times",
                                  (length_schools,), dtype=dt)
            md = f.create_dataset("Interpretation/v1/mask_depths",
                                  (length_schools,), dtype=dt)
            
            # Loop over all schools and get the start and stop depths
            for i, school in enumerate(schools):
                # Initialize empty numpy arra for the time variable
                sub_school = all_labels == school
                # Get the time indices for school i
                timeinds = np.where(np.sum(sub_school, 0) > 0)[0]
                # Loop over time
                for j, timeind in enumerate(timeinds):
                    # Find pairs of start and end depths for time i
                    diffs = (np.diff(np.sign(bin_labels[:, timeind])) != 0)*1
                    diffinds = np.where(diffs)[0]
                    # Extract range and add transducer depth and heave
                    nilz = r[diffinds] + trdepth[timeind] + heave[timeind]
                    # Number of equal timestamps
                    ki = int(len(nilz)/2)
                    # Append time vector and duplicate the time steps
                    if(j == 0):
                        T = npm.repmat(t[timeind], ki, 1)
                    else:
                        T = np.append(T, npm.repmat(t[timeind], ki, 1))
                        # Append range vector
                    if(j == 0):
                        R = nilz
                    else:
                        R = np.append(R, nilz)
                    # Store the range and time vector as a ragged array for each school
                md[i] = R
                mt[i] = np.reshape(T,len(T))


def time2NTtime(matlabSerialTime):
    # offset in days between ML serial time and NT time
    ML_NT_OFFSET = 584755  # datenum(1601, 1, 1, 0, 0, 0);
    # convert the offset to 100 nano second intervals
    # 60 * 60 * 24 * 10000000 = 864000000000
    ML_NT_OFFSET = ML_NT_OFFSET * 864000000000
    # Convert your MATLAB serial time to 100 nano second intervals
    matlabSerialTime = matlabSerialTime * 864000000000
    # Now subtract
    ntTime = matlabSerialTime - ML_NT_OFFSET
    return ntTime


# Loop over temporary files
Fs = glob.glob("/datawork/*.pkl")
for F in Fs:
    with open(F, 'rb') as f:
        print(F)
        [seg, labels, r, mattime, heave, trdepth] = pickle.load(f)
        createncfile(F, seg, labels, r, mattime, heave, trdepth)
