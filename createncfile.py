import pickle
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
import h5py
import datetime

# Read the tempoary data
F = "/datawork/2016837-D20160427-T221032.pkl"
with open(F, 'rb') as f:
    [seg, labels, r, t] = pickle.load(f)

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
        dt1 = h5py.vlen_dtype(np.dtype('float'))
        mt = f.create_dataset("Interpretation/v1/mask_time",
                              (length_schools,), dtype=dt1)

        # This requires 2 depths only, but this needs to be scalable
        rowtype = np.dtype([('f0', '<f4', (2, ))])
        dt2 = h5py.special_dtype(vlen=np.dtype(rowtype))
        md = f.create_dataset("Interpretation/v1/mask_depths",
                              (length_schools,), dtype=dt2)
        # https://stackoverflow.com/questions/41465480/h5py-how-to-store-many-2d-arrays-of-different-dimensions
        
        # Loop over all schools and get the start and stop depths
        k = 0
        for school in schools:
            sub_school = all_labels == school
            # Get the time indices for the school
            timeinds = np.where(np.sum(sub_school, 0) > 0)[0]
            # Add time data to the nc file
            mt[k] = t[timeinds]
            
            # Add the range data to the nc file
            testarray = np.ones((len(timeinds),), dtype=rowtype)
            for i, timeind in enumerate(timeinds):
                # Find start and end depths
                diffs = (np.diff(np.sign(bin_labels[:, timeind])) != 0)*1
                diffinds = np.where(diffs)[0]
                testarray[i] = r[diffinds][0:1]
            # print(testarray)
            md[k] = testarray
            # Book keeping for the next school
            k = k + 1
