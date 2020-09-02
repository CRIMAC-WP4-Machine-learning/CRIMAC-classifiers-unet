import pickle
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
import h5py
import datetime

F = "/datawork/2016837-D20160427-T221032.pkl"
with open(F, 'rb') as f:
    [seg, labels, r, t] = pickle.load(f)

#
# Tidy up data
#

# apply some logic operatioin to the data
bin_labels = (labels > 0) * 1   # get either 0 or 1 in the array

# plt.imshow(bin_labels, cmap=plt.cm.gray)  # use appropriate colormap here
# plt.show()

# Connect pixels for unique schools
all_labels = measure.label(bin_labels)

# Get the school numbers
schools = [schools for schools in np.unique(all_labels) if schools > 0]

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
        = "UNET predictions from Brautaset et al."

    #		types:
    #			// Note: empty_water == LSSS erased; no_data == LSSS excluded
    #			byte enum region_t {empty_water = 0, no_data = 1, analysis = 2, track = 3, marker = 4};
    #			// Storing 3D regions is not yet done, but we include the region dimension here anyway
    #			byte enum region_dim_t {twoD = 0, threeD = 1};
    #			float(*) mask_depth_t;
    #			mask_depth_t(*) mask_depths_t;
    #			uint64(*) mask_time_t; // ragged array for region ping times
    #		dimensions:
    #			regions = 3; // varies to suit data. Could also be unlimited
    #			channels = 3; // varies to suit data
    #			categories = 5; // varies to suit data.
    #		variables:
    #			float sound_speed;
    #				sound_speed:long_name = "Sound speed used to convert echo time into range";
    #				sound_speed:standard_name = "speed_of_sound_in_sea_water";
    #				sound_speed:units = "m/s";
    #				sound_speed:valid_min = 0.0f;
    #
    dset.attrs['Interpretation/v1/version_comment'] = "UNET";
    #			// The bounding box of each region
    #			float min_depth(regions);
    #				min_depth:long_name = "Minimum depth for each region";
    #				min_depth:units = "m";
    #				min_depth:valid_min = 0.0f;
    #			float max_depth(regions);
    #				max_depth:long_name = "Maximum depth for each regions";
    #				max_depth:units = "m";
    #				max_depth:valid_min = 0.0f;
    #			uint64 start_time(regions);
    #				start_time:long_name = "Timestamp of the earliest data point in each region";
    #				start_time:units = "milliseconds since 1601-01-01 00:00:00Z";
    #				start_time:axis = "T";
    #				start_time:calendar = "gregorian";
    #				start_time:standard_name = "time";
    #			uint64 end_time(regions);
    #				end_time:long_name = "Timestamp of the latest data point in each region";
    #				end_time:units = "milliseconds since 1601-01-01 00:00:00Z";
    #				end_time:axis = "T";
    #				end_time:calendar = "gregorian";
    #				end_time:standard_name = "time";
    #				
    #			region_dim_t region_dimension; 
    #				region_dimension:long_name = "Region dimension";
    #
    #			int region_id(regions);
    #				region_id:long_name = "Dataset-unique identification number for each region";
    #			string region_name(regions);
    #				region_name:long_name = "Name of each region";
    #				region_name:_Encoding = "utf-8";
    #			string region_provenance(regions);
    #				region_provenance:long_name = "Provenance of each region"; 
    #				region_provenance:_Encoding = "utf-8";
    #			string region_comment(regions);
    #				region_comment:long_name = "Comment for each region";
    #				region_comment:_Encoding = "utf-8";
    #			int region_order(regions);
    #				region_order:long_name = "The stacking order of the region";
    #				region_order:comment = "Regions of the same order cannot overlap";
    #			region_t region_type(regions);
    #				region_type:long_name = "Region type";
    #			
    #			// The acosutic categories. Each layer may have several categories and proportions.
    #			string region_category_names(categories);
    #				region_category_names:long_name = "Categorisation name";
    #				region_category_names:_Encoding = "utf-8";
    #			float region_category_proportions(categories);
    #				region_category_proportions:long_name = "Proportion of backscatter for the categorisation";
    #				region_category_proportions:value_range = 0.0f, 1.0f;
    #			int region_category_ids(categories);
    #				region_category_ids:long_name = "region_id of this categorisation and proportion";
    #			
    #			string channel_names(channels);
    #				channel_names:long_name = "Echosounder channel names";
    #				channel_names:_Encoding = "utf-8";
    #			uint region_channels(regions);
    #				region_channels:long_name = "Echosounder channels that this region applies to";
    #				region_channels:description = "Bit mask derived from channel_names (index 1 of channel_names = bit 1, index 2 = bit 2, etc). Set bits in excess of the number of channels are to be ignored.";
    #				region_channels:_FillValue = 4294967295; // 2^32-1
    #				
    #			mask_time_t mask_times(regions);
    #				mask_times:long_name = "Timestamp of each mask point";
    #				mask_times:units = "milliseconds since 1601-01-01 00:00:00Z";
    #				mask_times:axis = "T";
    #				mask_times:calendar = "gregorian";
    #				mask_times:standard_name = "time";
    #			mask_depths_t mask_depths(regions);
    #				mask_depths:long_name = "Depth pairs of mask";
    #				mask_depths:units = "m";
    #				mask_depths:valid_min = 0.0f;




#// simple example regions
    #region_dimension = 'twoD'
    #sound_speed = 1496
    #min_depth =  [0.0, 20.5, 55.0]
    #max_depth = [10.0, 42.0, 125.2]
    #start_time = [13189164120001, 13189164121000, 13189164124000]
    #end_time =   [13189164123004, 13189164124000, 13189164131000]
    #region_id = [1, 5, 234]
    #region_name = "region1", "region2", "";
    #region_provenance = "KORONA-2.6.0;LSSS", "Echoview - template ABC", "Manual inspection";
    #region_comment = "", "", "whale!";
    #region_category_names = "herring", "krill", "seal", "lion", "platypus";
    #region_category_proportions = [0.9, 0.1, 0.45, 0.40, 0.10]
    #region_category_ids = [1, 1, 234, 234, 234]
    #region_type = analysis, empty_water, analysis;
    #channel_names = "18kHz WBT ABC", "38kHz WBT ZYX", "120kHz GPT 123";
    #region_channels = [5, 7, 7]
    #mask_times = {13189164120001, 13189164121002, 13189164122003, 13189164123004}, {13189164121000, 13189164122000, 13189164123000, 13189164124000}, {13189164124000, 13189164125000, 13189164126000, 13189164127000, 13189164128000, 13189164129000, 13189164131000}
    #mask_depths = {[0.0, 15.0], [0.0, 4.0, 5.0, 10.0], [0.0, 10.0], [0.0, 10.0]}, {[20.5, 25.0], [30.5, 35.0], [35.5, 40.0], [40.0, 42.0]}, {[55.0, 105.0], [60.0, 80.2, 100.6, 115.0], [55.0, 107.0], [55.0, 110.0], [55.0, 115.6], [55.0, 125.2], [60, 115]}
            

    

# Loop over all schools and get the start and stop depths
for school in schools:
    
    sub_school = all_labels == school
    # Get the time indices for the school
    timeinds = np.where(np.sum(sub_school, 0) > 0)[0]
    # Loop over the time indices and get the depths
    for timeind in timeinds:
        print(bin_labels[:, timeind])
        # Find start and end depths
        diffs = (np.diff(np.sign(bin_labels[:, timeind])) != 0)*1
        diffinds = np.where(diffs)[0]
        startdepths = r[diffinds[0::2]]
        enddepths = r[diffinds[1::2]]
        timeping = t[timeind]

