% Run the generate_metadata and generate_metadata2 prior to this script
%
% This script reads the metadata, raw acoustic data and the labels, convert the % input data to a mat file that include both raw data and labels. The script
% also interpolates the data into a common grid.
%
% It is recommended to keep a separate scratch and data directory. If the
% reprocessing the data is required, simply deleting the scratch directory
% is preferable. If not the indidivual datastatus.mat files needs to be
% deleted (one per survey per year). If not the files that already have
% been processed will be skipped.
%
% Dependencies:
% https://github.com/nilsolav/LSSSreader/src
% https://github.com/nilsolav/NMDAPIreader
% https://github.com/nilsolav/readEKraw
%
% Required inpout data files (example for the sand eel case):
% <datafolder>/dataoverviews/DataOverview/dataoverviews/DataOverview/DataOverview_North Sea NOR Sandeel cruise in Apr_May.mat
% <datafolder>/dataoverviews/DataOverview/dataoverviews/DataOverview/DataOverview_North Sea NOR Sandeel cruise in Apr_May.bot
% <datafolder>/data/DataOverview_North Sea NOR Sandeel cruise in Apr_May/<year>/<pairs of snap and raw files>

% Output data files:
% <scratchfolder>/data/DataOverview_North Sea NOR Sandeel cruise in Apr_May/<year>/<png file per raw/snap combination>
% <scratchfolder>/data/DataOverview_North Sea NOR Sandeel cruise in Apr_May/<year>/<index png file per raw/snap combination>
% <scratchfolder>/data/DataOverview_North Sea NOR Sandeel cruise in Apr_May/<year>/<mat file per raw/snap combination>
% <scratchfolder>/data/DataOverview_North Sea NOR Sandeel cruise in Apr_May/<year>/datastatus.mat (matrix to determine progress; delete this if you would like to reprocess all the data)

% COGMAR project, Nils Olav Handegard

% Plotting frequency
% Which frequency to use when generating the plots
par.plot_frequency='200';

% Which range vector to use when interpolating into the common grid
par.range_frequency = 200;

par.bottomoutlier = 95; % Assumes less than 5 percent outliers in depthdata
par.depthoffset = 15;% Add 10 m below seafloor

par.dz = 0.188799999523326;
par.dzdiff = .05;
% Initialisation and local environment

% Create a matlab file setenvironment.m for setting local environments
% and edit this to fit your env:

%function fld = setenvironment
% % This file sets the matlab environment for the specific server
% %
% % This setup is for the IMR GPU server
%
%fld.scracth = '/scratch/nilsolav/deep/data/echosounder/akustikk_all/';
%fld.data  = '/gpfs/gpfs0/deep/data/echosounder/akustikk_all/';
%addpath('/nethome/nilsolav/repos/github/LSSSreader/src/')
%addpath('/nethome/nilsolav/repos/hg/matlabtoolbox/echolab/readEKRaw')

% Set the local environment or use the default
if exist('setenvironment')
    fld=setenvironment;
else
    fld.scracth = '/scratch/nilsolav/deep/data/echosounder_scratch/akustikk_all/';
    fld.data    = '/scratch/nilsolav/deep/data/echosounder/akustikk_all/data';
    addpath('/nethome/nilsolav/repos/github/LSSSreader/src/')
    addpath('/nethome/nilsolav/repos/hg/matlabtoolbox/echolab/readEKRaw')
end

cruises = {...
    'North Sea NOR Sandeel cruise in Apr_May','2007','S2007205_PJOHANHJORT_1019';...
    'North Sea NOR Sandeel cruise in Apr_May','2008','S2008205_PJOHANHJORT_1019';...
    'North Sea NOR Sandeel cruise in Apr_May','2009','S2009107_PGOSARS_4174';...
    'North Sea NOR Sandeel cruise in Apr_May','2010','S2010205_PJOHANHJORT_1019';...
    'North Sea NOR Sandeel cruise in Apr_May','2011','S2011206_PJOHANHJORT_1019';...
    'North Sea NOR Sandeel cruise in Apr_May','2012','S2012837_PBRENNHOLM_4405';...
    'North Sea NOR Sandeel cruise in Apr_May','2013','S2013842_PEROS_3317';...
    'North Sea NOR Sandeel cruise in Apr_May','2014','S2014807_PEROS_3317';...
    'North Sea NOR Sandeel cruise in Apr_May','2015','S2015837_PEROS_3317';...
    'North Sea NOR Sandeel cruise in Apr_May','2016','S2016837_PEROS_3317';...
    'North Sea NOR Sandeel cruise in Apr_May','2017','S2017843_PEROS_3317';...
    'North Sea NOR Sandeel cruise in Apr_May','2018','S2018823_PEROS_3317';...
    'North Sea NOR Sandeel cruise in Apr_May','2019','S2019847_PEROS_3317'};

%%
for k=1:size(cruises,1)
    % Survey data directory per year
    dd_data_year = fullfile(fld.data,cruises{k,3});
    disp(dd_data_year)
    % Scratch directory per year
    dd_scratch_year = fullfile(fld.scracth,cruises{k,1},cruises{k,2});
    disp(dd_scratch_year)
    % Create it if it does not exist
    if ~exist(dd_scratch_year)
        mkdir(dd_scratch_year)
    end
    
    % Get the file list
    raw0 = dir(fullfile(dd_data_year,'ACOUSTIC','EK60','EK60_RAWDATA','*.raw'));
    
    % Generate status file if it is missing
    statusfile = fullfile(dd_scratch_year,'datastatus.mat');
    if ~exist(statusfile)
        status = zeros(length(raw0),1);
        save(statusfile,'status')
    end
    
    % Loop over file pairs
    for f=1:length(raw0)
        load(statusfile)
        % Create file names (in and out)
        [~,fn,~]=fileparts(raw0(f).name);
        % Run files that have not been run earlier
        qrun = status(f)<=0;
        % Get files
        bot = fullfile(dd_data_year,'ACOUSTIC','EK60','EK60_RAWDATA',[fn,'.bot']);
        raw = fullfile(dd_data_year,'ACOUSTIC','EK60','EK60_RAWDATA',[fn,'.raw']);
        snap = fullfile(dd_data_year,'ACOUSTIC','LSSS','WORK',[fn,'.snap']);
        % Output files
        mat = fullfile(dd_scratch_year,[fn,'.mat']);
        png = fullfile(dd_scratch_year,[fn,'.png']);
        png_I = fullfile(dd_scratch_year,[fn,'_I.png']);
        png_I2 = fullfile(dd_scratch_year,[fn,'_I2.png']);
        if qrun
            disp([datestr(now),'; running ; ',fullfile(dd_data_year,fn)])
            % Generate figures and save clean data file
            try
                generate_mat_files(snap,raw,bot,png,png_I,png_I2,mat,par)
                close gcf
                disp([datestr(now),'; success ; ',fn])
                status(f)=now;
            catch ME
                disp([datestr(now),'; failed  ; ',fn])
                status(f)=-now;
                disp([ME.identifier]);
                disp([ME.message]);
                for MEi = 1:length(ME.stack)
                    disp(ME.stack(MEi))
                end
            end
        else
            disp([datestr(now),'; exists ; ',fn])
        end
        save(statusfile,'status')
    end
end
