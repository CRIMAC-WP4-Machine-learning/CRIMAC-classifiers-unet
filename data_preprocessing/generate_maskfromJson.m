% Reads the LSSS json data from the LSSS API call 	/lsss/regions/deletion
% and generates polygons for each deleted regions.
%
% First the Korona module needs to be run with the correct parameters. The 
% file 'ConfigFileSettings.cfs' point to the files used by korona. First
% generate a new directory called 'LSSSkoronarun' by copying the 'LSSS'
% directory for each survey. Then replace the files 
% 'LSSS_koronarun/LSSS_FILES/plankton.xml'
% 'LSSS_koronarun/LSSS_FILES/categorization.xml'
% 'LSSS_koronarun/LSSS_FILES/KoronaModuleSetup_example_FULL_MULTIFREQUENCY_2015_03_28.cds'
% 'LSSS_koronarun/LSSS_FILES/categories/' (whole direcotry)
% and edit the 'ConfigFileSettings.cfs' file to point to these files. 
%
% After this step, open the LSSS from the LSSS_koronarun folder and start korona. 
%
% Reopen the LSSS project and use the Korona files instead of the original raw files.
% Start with a new interpretation (Regions - Reset interpretation),
% push the Korona buttom to use the Korona files. Right Click on the right bar showing
% the intensity legend and  selsect "Conditional masking". Select the class you would
% like to "erase", i.e. "sandeel" and store to database. Then read the erased 
% region using the LSSS api to get the json string that codes the erased resgions
% (lsss/regions/deletion). In python simply pass this code (after enabling the LSSS aPI):
%
% Added functionality to extract transducerdepth from raw files.
% 
%erasedRegionsasSandEel = lsss.get('/lsss/regions/deletion')
%with open('data.json', 'w') as f:
%    json.dump(erasedRegionsasSandEel, f, sort_keys=True)
%
% And place the json file in the data directory

%%
if exist('setenvironment')
   fld=setenvironment;
else
  fld.scracth = '/scratch/nilsolav/deep/data/echosounder/akustikk_all/';
  fld.data    = '/scratch/nilsolav/deep/data/echosounder/akustikk_all/';
  addpath('/nethome/nilsolav/repos/github/LSSSreader/src/')
  addpath('/nethome/nilsolav/repos/hg/matlabtoolbox/echolab/readEKRaw')
end

year = {'2018'};
i=1;
data = fullfile(fld.data,'data','DataOverview_North Sea NOR Sandeel cruise in Apr_May',year{i});
scratch = fullfile(fld.scracth,'data','DataOverview_North Sea NOR Sandeel cruise in Apr_May',year{i});

%% Read the json file
file = fullfile(data,['erasedmask',year{i},'.json']);
fid = fopen(file); 
raw = fread(fid,inf); 
str = char(raw'); 
fclose(fid); 
% Parse the json file
val = jsondecode(str);

%% List raw files
rf = dir(fullfile(data,'*.raw'));
tstr = 'yyyy-mm-ddTHH:MM:SS.FFFZ';

%% Loop over files
ti=1;%Initilize index for Korona labels
ch=4;
for j = 100%:11%1:length(rf)-1
    % Set up new figure
    clf
    hold on
    axis ij
    
    fn = rf(j).name(10:end-4);
    fn_mat = fullfile(scratch,[rf(j).name(1:end-4),'.mat']);
    
    % Read raw file to get the offset
    [Dheader,Ddata]=readEKRaw(fullfile(data,[rf(j).name]));
        
    % Load mat file for plotting depth
    RAW = load(fn_mat);
    imagesc(1:length(RAW.t),RAW.range,squeeze(10*log10(RAW.sv(:,:,ch))))
    %plot(RAW.t,RAW.depths(:,3),'k')
    plot(RAW.depths(:,3),'k')
    %imagesc(squeeze(10*log10(RAW.sv(:,:,4))))
    %plot(interp1(RAW.range,1:length(RAW.range),RAW.depths(:,3)),'r')
    
    starttime = datenum(fn,'YYYYmmdd-THHMMSS');
    datestr(starttime)
    fn = rf(j+1).name(10:end-4);
    endtime = datenum(fn,'YYYYmmdd-THHMMSS');
    datestr(endtime)

    
    t = datenum(val(ti).time,tstr);
    while t<endtime|ti>length(val)
        t = datenum(val(ti).time,tstr);
        if t>starttime
            tind = interp1(RAW.t,1:length(RAW.t),t);
            for ri = 1:length(val(ti).depthRanges)
%                plot([tind tind],interp1(RAW.range,1:length(RAW.range),[val(ti).depthRanges(ri).min val(ti).depthRanges(ri).max]),'r')
                plot([tind tind],[val(ti).depthRanges(ri).min val(ti).depthRanges(ri).max]-Ddata.pings(ch).transducerdepth(1),'r')
            end
        end
        ti=ti+1;
    end
    %datetick('x','HH:MM')
    title(datestr(starttime))
    png=fullfile(scratch,[rf(j).name(1:end-4),'_Ikorona.png']);
    print(png,'-dpng')
%    close(gcf)
end



