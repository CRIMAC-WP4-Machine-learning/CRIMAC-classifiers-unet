%
% Extract transducerdepth from raw files and store as nc.
% 

%% Look in the local disk instead
fld.data = 'E:\Sandeel_cruises\';
fld.scracth = 'E:\Sandeel_cruises\transducerdepths';

cruises = dir(fullfile(fld.data,'S2*'));
fldstr = fullfile('ACOUSTIC','EK60','EK60_RAWDATA');

for cr = 13%:length(cruises)
    disp(cruises(cr).name)
    % List raw files
    rf = dir(fullfile(cruises(cr).folder,cruises(cr).name,fldstr,'*.raw'));
    disp(length(rf))
    % Loop over files
    for j = 187:length(rf)
        % Read raw file to get the offset
        try
            [Dheader,Ddata]=readEKRaw(fullfile(rf(j).folder,rf(j).name));
            
            % Save the shit
            for i=1:length(Ddata.pings)
                transducerdepth = NaN;
                transducerdepth = [double(Ddata.pings(i).transducerdepth);...
                    Ddata.pings(i).time];
                % Frequency
                ds = ['/transducer/',num2str(Ddata.pings(i).frequency(1))];
                F = fullfile(fld.scracth,[rf(j).name(1:end-4),'.h5']);
                hdf5write(F, ds, transducerdepth)
            end
        end
    end
end

%% Create a single time series per year
% Load hdf file
F2=dir(fullfile(fld.scracth,'*.h5'));
for i=1:length(F2)
   %h5disp()
   Fn = fullfile(F2(i).folder, F2(i).name);
   try
       dum = h5read(Fn,'/transducer/200000');
       if std(dum(1,:))~=0
           warning(['Multiple depths in ',Fn])
           figure
           plot(dum(2,:),dum(1,:))
           datetick('x')
           title(Fn)
       end
   end
end


