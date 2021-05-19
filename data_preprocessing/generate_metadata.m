%% generate_metadata.m
% This script reads the NMDAPI to get the list of cruiseseries and search
% through the data storage to get lists of all the raw files and snap
% (classification) files. After running this scrip run the
% generate_metadata2.m script to copy cruise series to the data disk.
%
% File output:
% \\ces.imr.no\deep\data\echsosounder\akustikk_all\dataoverviews\DataOverview_<surveytimeseries>.mat
% contains the variable 'DataStatus' 
% \\ces.imr.no\deep\data\echsosounder\akustikk_all\dataoverviews\DataPairedFiles_<surveytimeseries>.mat
% contains the variable 'pairedfiles' 
%
% Variables:
% DataStatus(:,1) : CruiseSeries name
% DataStatus(:,2) : Year
% DataStatus(:,3) : CruiseNr
% DataStatus(:,4) : ShipName
% DataStatus(:,5) : DataPath
% DataStatus(:,6) : Problem, i.e. if the data is not at the standard location
% DataStatus(:,7) : raw, number of raw files in standard location
% DataStatus(:,8) : Snap, number of snap files in standard location
% DataStatus(:,1) : Workfiles
% DataStatus(:,1) : RawfilesNotStdLocation
% DataStatus(:,1) : SnapfilesNotStdLocation
% DataStatus(:,1) : WorkfilesNotStdLocation
%
% pairedfiles.F       : List of unique file combinations.
% pairedfiles.F{i,1}  : Full path to snap file
% pairedfiles.F{i,2}  : Full path to work file
% pairedfiles.F{i,3}  : Full path to raw file
%
% NB: This script reads the IMR data structure and does not work outisde
% the firewall.
%
% Dependencies:
% https://github.com/nilsolav/LSSSreader/src
% https://github.com/nilsolav/NMDAPIreader
% 

if isunix
    %cd /nethome/nilsolav/repos/nilsolav/MODELS/MLprosjekt/
    addpath('/nethome/nilsolav/repos/github/LSSSreader/src/')
    addpath('/nethome/nilsolav/repos/github/NMDAPIreader/')
    addpath('/nethome/nilsolav/repos//nilsolav/MODELS/matlabtools/Enhanced_rdir')
    dd='/data/cruise_data/';
    dd_out = '/gpfs/gpfs0/deep/data/echosounder/akustikk_all/dataoverviews/';
else
    cd D:\repos\Github\acoustic_private\data_preprocessing
    dd='\\ces.imr.no\cruise_data\';
    dd_out = 'D:\DATA\deep\echosounder\akustikk_all\dataoverviews\';
end

%% Get survey time series structure
if ~(exist('D.mat')==2)
  disp('REading NMD api')
  D = NMDAPIreader_readcruiseseries;
  save('D','D')
else
  disp('loading D.mat')
  load('D.mat')
end
disp('Finished reading the API')


%% Get information and data statistics per survey
if false
DataStatus = cell(1,8);
DataStatus(1,:) ={'CruiseSeries','Year','CruiseNr','ShipName','DataPath','Problem','raw','Snap'};
l=2;
for i = 1:length(D)
    disp([D(i).name])
    for j=1:length(D(i).sampletime)
        ds = fullfile(dd,D(i).sampletime(j).sampletime);
        disp(['   ',D(i).sampletime(j).sampletime])
        for k=1:length(D(i).sampletime(j).Cruise)
            DataStatus{l,1} = D(i).name;
            DataStatus{l,2} = D(i).sampletime(j).sampletime;
            DataStatus{l,3} = D(i).sampletime(j).Cruise(k).cruisenr;
            DataStatus{l,4} = D(i).sampletime(j).Cruise(k).shipName;
            if ~isempty(D(i).sampletime(j).Cruise(k).cruise)
                if isfield(D(i).sampletime(j).Cruise(k).cruise.datapath,'Text')
                    DataStatus{l,5} = D(i).sampletime(j).Cruise(k).cruise.datapath.Text;
                end
                DataStatus{l,6} = D(i).sampletime(j).Cruise(k).cruise.datapath.Comment;
                if isfield(D(i).sampletime(j).Cruise(k).cruise.datapath,'rawfiles')
                    DataStatus{l,7} = D(i).sampletime(j).Cruise(k).cruise.datapath.rawfiles;
                end
                if isfield(D(i).sampletime(j).Cruise(k).cruise.datapath,'snapfiles')
                    DataStatus{l,8} = D(i).sampletime(j).Cruise(k).cruise.datapath.snapfiles;
                end
            end
            l=l+1;
        end
    end
end

%% Save summary data
fid=fopen([fullfile(dd_out,'DataOverview.csv')],'wt');
for i=1:size(DataStatus,1)
    for j=1:size(DataStatus,2)
        if i>1&&ismember(j,[7 8])
            st='%i;';
            str = (DataStatus{i,j});
        else
            st = '%s;';
            str=DataStatus{i,j};
        end
        fprintf(fid,st,str);
    end
    fprintf(fid,'\n');
end
fclose(fid);

end

%% Crunch data - count files per series and get list of files

for i = 11%1:length(D)
    tic
    DataStatus = cell(1,12);
    DataStatus(1,:) ={'CruiseSeries','Year','CruiseNr','ShipName','DataPath','Problem','Rawfiles','Snapfiles','Workfiles','RawfilesNotStdLocation','SnapfilesNotStdLocation','WorkfilesNotStdLocation'};
    l=2;
    disp([D(i).name])
    for j=1:length(D(i).sampletime)
        ds = fullfile(dd,D(i).sampletime(j).sampletime);
        disp(['   ',D(i).sampletime(j).sampletime])
        for k=1:length(D(i).sampletime(j).Cruise)
            DataStatus{l,1} = D(i).name;
            DataStatus{l,2} = D(i).sampletime(j).sampletime;
            DataStatus{l,3} = D(i).sampletime(j).Cruise(k).cruisenr;
            DataStatus{l,4} = D(i).sampletime(j).Cruise(k).shipName;
            DataStatus{l,5} = D(i).sampletime(j).Cruise(k).datapath.path;
            DataStatus{l,6} = D(i).sampletime(j).Cruise(k).cruise.datapath.Comment;
            
            % Go into the directory and check the files
            if exist(DataStatus{l,5},'dir')==7

                % Get information per cruise
                [filecount,files]   = NMDAPIreader_getLSSSdatastatus(DataStatus{l,5});
                
                % Pair files
                pairedfiles{l-1}=LSSSreader_pairfiles(files);
                
                % Combine the different files
                DataStatus{l,7}  = filecount(1);
                DataStatus{l,8}  = filecount(2);
                DataStatus{l,9}  = filecount(3);
                DataStatus{l,10} = filecount(4);
                DataStatus{l,11} = filecount(5);
                DataStatus{l,12} = filecount(6);
            else
                % Combine the different files
                DataStatus{l,7}  = NaN;
                DataStatus{l,8}  = NaN;
                DataStatus{l,9}  = NaN;
                DataStatus{l,10} = NaN;
                DataStatus{l,11} = NaN;
                DataStatus{l,12} = NaN;
            end
            l=l+1;
        end
    end
    % Write data status
   save([dd_out,'DataPairedFiles_',D(i).name,'.mat'],'pairedfiles');
   save([dd_out,'DataOverview_',D(i).name,'.mat'],'DataStatus');
   clear pairedfiles
    fid=fopen([dd_out,'DataOverview_',D(i).name,'.csv'],'w');
    for q=1:size(DataStatus,1)
        for r=1:size(DataStatus,2)
            fprintf(fid,'%s;',DataStatus{q,r});
        end
        fprintf(fid,'\n');
    end
    fclose(fid)
    toc
end





