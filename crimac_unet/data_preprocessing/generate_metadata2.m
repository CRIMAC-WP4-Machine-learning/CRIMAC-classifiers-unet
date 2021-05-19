%% MLproject_main2.m
% This script reads the metadata, copies pairs of snap and raw data into:
% /data/deep/data/echsosounder/akustikk_all/data/<cruise_series>/<year> 
%


%% Init
if isunix
    cd /nethome/nilsolav/repos/github/acoustic_private/data_preprocessing/
    dd_out = '/gpfs/gpfs0/deep/data/echosounder/akustikk_all/';
else
    cd D:/repos/github/acoustic_private/data_preprocessing/
    dd_out = 'D:\DATA\deep\echosounder\akustikk_all\';
end

DataOverview = dir(fullfile(dd_out,'dataoverviews','DataOverview*.mat'));


%% Start loop over cruise series
warning off
for k=11%1:length(DataOverview)
    dd_data = fullfile(dd_out,'data',DataOverview(k).name(1:end-4));
    if ~exist(dd_data)
        mkdir(dd_data)
    end
    
    % Load the paired files
    dat = load(fullfile(dd_out,'dataoverviews',['DataPairedFiles',DataOverview(k).name(13:end)]));
    dat2 = load(fullfile(dd_out,'dataoverviews',['DataOverview',DataOverview(k).name(13:end)]));
    
    % Loop over years
    for i=14%1:length(dat.pairedfiles)
        % Create year directory
        dd_data_year = fullfile(dd_out,'data',DataOverview(k).name(1:end-4),dat2.DataStatus{i+1,2});
        if ~exist(dd_data_year)
            mkdir(dd_data_year)
        end
        
        % I need column one and three (snap and raw)
        disp(dat2.DataStatus{i+1,2})
        if size(dat.pairedfiles{i}.F,2)==3
            for j=1:size(dat.pairedfiles{i}.F,1)
                if ~isempty(dat.pairedfiles{i}.F{j,3}) &&~isempty(dat.pairedfiles{i}.F{j,1})

                    % Create file names (in and out)
                    snap=dat.pairedfiles{i}.F{j,1};
                    raw=dat.pairedfiles{i}.F{j,3};
                    bot = [dat.pairedfiles{i}.F{j,3}(1:end-4),'.bot'];
                    [~,fn,~]=fileparts(dat.pairedfiles{i}.F{j,3});
                    raw_out = fullfile(dd_data_year,[fn,'.raw']);
                    bot_out = fullfile(dd_data_year,[fn,'.bot']);
                    snap_out = fullfile(dd_data_year,[fn,'.snap']);
                    
                    % Copy files to scratch disk
                    if exist(bot)
                    [~,msg,~] = copyfile(bot,bot_out); 
                    end
                    disp(msg)
                    [~,msg,~] = copyfile(snap,snap_out); 
                    disp(msg)
                    [~,msg,~] = copyfile(raw,raw_out); 
                    disp(msg)
                end
            end
        end
    end
end
