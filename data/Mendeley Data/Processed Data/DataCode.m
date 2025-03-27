clc
clear all 
close all

%% DataCode.m file reads and processes the human breath radar data and reference data. 
%% After background removal with Linear Trend Subtraction,radar breath signals are 
%% extracted and the results are saved in Processed data file. To generate Processed Data, 
%% please use DataCode.m. Salsa Ancho radar module and ST life.augmented VL53L0X lidar 
%% sensor is used for data collection. This code is written by Cansu EREN to provide...
%% researchers to analyze the human radar data with reference data for future scientific
%% studies. 
%% The dataset address should be C:\Users\...\...\DataSet which is main dataset file location.  
%% This code is written in MATLAB R2020a. 
%% Author: Cansu EREN 
%% Copyright (C) 2023  Cansu EREN 

%% This program is free software: you can redistribute it and/or modify  it under the terms 
%% of the GNU General Public License as published by the Free Software Foundation, version
%% 3 of the License. Further information,please check GNU General Public License v3.0 .txt. 

%% This data is used under the terms of  ODC Open Database License (ODbL). Further 
%% information, please check GNU General Public License(ODbL).txt 

% STEP 1- Load radar parameters  
SampleN=256;  % Number of samples in fast time. 
SampleM=1099; % Number of samples in slow time. 
RangeRes=   0.0039; %Range resolution (m). 
TimeResRadar=0.0533; % Radar data time resolution (s).  
TimeDurRadar=TimeResRadar*SampleM; % Radar time duration (s). 
Range=1; % Range (m) 
SamplingFreqRadar=1/TimeResRadar; % Sampling frequency of  radar data (Hz). 
FreqResRadar=SamplingFreqRadar/SampleM; % Frequency resolution of radar (Hz). 
tRadar=0: TimeResRadar: TimeDurRadar-TimeResRadar; % Radar observation time vector (s).  
rr=0:RangeRes:Range-RangeRes; % Range vector (m). 
ffRadar=0: FreqResRadar: SamplingFreqRadar-FreqResRadar; % Frequency  vector radar (Hz). 
FreqDCRem=7; % Priori sample number to suppress DC frequency. 
PreRangeIndx=100; % This range index sets a threshold for range extraction. 

% STEP-2  Load reference  parameters 
TimeResRef=0.325; % Lidar time resolution (s). 
SamplingFreqRef= 1/TimeResRef; % Sampling frequency of Lidar data (Hz). 
tRef= (1:180).*TimeResRef; % Lidar observation time vector (s). 
SampleRef=length(tRef); % Number of samples Lidar. 
FreqResRef=SamplingFreqRef/SampleRef; % Frequency resolution of Lidar (Hz). 
ffRef=0:FreqResRef: SamplingFreqRef-FreqResRef; % Frequency resolution vector Lidar (Hz). 

% STEP-3 Select dataset location. 
QuestionStruct.Interpreter='tex';       
QuestionStruct.Default='Yes';      
Question=...
       questdlg('\fontsize{10}\color{black}Select main dataset file location to load',...
       'Question', 'Yes', 'No', 'Cancel', QuestionStruct);

switch Question
    case 'Yes'
      FileLocation = uigetdir('C:\'); 
    case 'No'
        MessageStruct.Interpreter='tex'; 
        MessageStruct.WindowStyle='modal'; 
        Message=msgbox({'\fontsize{10}\color{black}You should select main file location'}, 'ReadMe',...
            'help', MessageStruct) ;  
        clc
        clear all
        close all 
        return
    case 'Cancel'
        clc
        clear all
        close all 
        return
end

cd(FileLocation)

FileLocationRadar=fullfile(FileLocation, '\Raw Radar Data\'); 
FileLocationReference=fullfile(FileLocation,  '\Reference Data\'); 
FileLocationProcessed=fullfile(FileLocation, '\Processed Data\'); 

                    for kk=1:3

                                TempDirName=['Bandwidth', num2str(kk)]; 
                                SubFileLocationRadar= fullfile(FileLocationRadar, TempDirName); 
                                SubFileLocationReference= fullfile(FileLocationReference, TempDirName);                               
                                dirListingRadar = dir([SubFileLocationRadar, '/', '*.mat']);
                                dirListingReference= dir([SubFileLocationReference, '/', '*.mat']);
                    
                                % STEP 4-Radar and Reference data processing&saving
                                
                                    for ii=1:length(dirListingRadar) 
                        
                                                    % Data Loading 
                                                    load( [ SubFileLocationRadar, '\' dirListingRadar(ii).name]); 
                                                    dirListingRadar(ii).name
                                                    load( [ SubFileLocationReference, '\' dirListingReference(ii).name]); 
                                                    dirListingReference(ii).name
                                                                                
                                                    % Background removal with Linear Trend Subtraction
                                                    BS=detrend(bScan,1);  % bScan: Radar data name in workspace
                                                    BS=BS';
                    
                                                    % Extraction of target range index & radar breath signal.  
                                                    [~,RandeIndex]=max(BS(PreRangeIndx:SampleN,:)); 
                                                    EstimatedRangeIndex(ii)=fix(mean(RandeIndex))+PreRangeIndx;  
                                                    EstimatedRange(ii)=EstimatedRangeIndex(ii)*RangeRes; 
                                                    BreathSignalRadar=BS(EstimatedRangeIndex(ii),:); 
                                                    BreathSignalRadar=lowpass(BreathSignalRadar,1.5,SamplingFreqRadar); 

                                                    % Save background removed radar data & radar breath signal. 
                                                    save (strcat(FileLocationProcessed, TempDirName,"\BS_" , dirListingRadar(ii).name), "BS")
                                                    save (strcat(FileLocationProcessed, TempDirName, "\FilteredBreathRadar_" , dirListingRadar(ii).name), ...
                                                        "BreathSignalRadar")
                                              
                                                    % Spectrum analysis of radar breath signal using fft. 
                                                    SpectrumRadar=abs(fft(BreathSignalRadar));
                                                    SpectrumRadar=SpectrumRadar(1:fix(SampleM/2)); 
                                                    SpectrumRadar=SpectrumRadar/length(SpectrumRadar); 
                    
                                                    % Save radar breath spectrum. 
                                                    save (strcat(FileLocationProcessed,TempDirName,  "\SpectrumRadar_" , dirListingRadar(ii).name),...
                                                        "SpectrumRadar")
                        
                                                    % Frequency extraction of radar breath signal. 
                                                    [~,  FreqIndexRadar]=max(SpectrumRadar(FreqDCRem:end)); 
                                                    FreqIndexRadar=FreqIndexRadar+FreqDCRem-1; 
                                                    EstimatedFrequencyRadar(ii)=FreqIndexRadar*FreqResRadar; 
                                                    RadarPeakPowerBreath(ii)=mag2db(SpectrumRadar(FreqIndexRadar)^2); 
                                                    RadarNoisePowerBreath(ii)=mag2db( var (SpectrumRadar(FreqDCRem:FreqDCRem+40))/ (length(FreqDCRem:FreqDCRem+40)) ) ; 
                                                    RadarSignalToNoise(ii)=RadarPeakPowerBreath(ii)-RadarNoisePowerBreath(ii); 

                                                    % Spectrum analysis of reference breath signal using fft. 
                                                    Ref(1)=Ref(2); 
                                                    Ref=lowpass(Ref,1.5,SamplingFreqRef); 
                                                    SpectrumRef=abs(fft(Ref));
                                                    SpectrumRef=  SpectrumRef(1:(SampleRef/2)); 
                                                    SpectrumRef=SpectrumRef/(SampleRef/2); 
                                                
                                                    %  Frequency extraction of reference breath signal. 
                                                    [~, FreqIndexRef]= max(SpectrumRef(FreqDCRem:end));
                                                    FreqIndexRef=FreqDCRem+FreqIndexRef-1; 
                                                    EstimatedFrequencyRef(ii)=FreqIndexRef*FreqResRef; 
                                                    RefPeakPowerBreath(ii)=mag2db(SpectrumRef(FreqIndexRef)^2); 
                                                    RefNoisePowerBreath(ii)=mag2db( var( SpectrumRef(FreqDCRem:FreqDCRem+40) )/(length(FreqDCRem:FreqDCRem+40)) ); 
                                                    RefSignalToNoise(ii)=RefPeakPowerBreath(ii)-RefNoisePowerBreath(ii); 
                                                
                                                    %  Save reference breath spectrum & reference signal. %
                                                    save (strcat(FileLocationProcessed, TempDirName,  "\Spectrum_" , dirListingReference(ii).name), "SpectrumRef")
                                                    save (strcat(FileLocationProcessed, TempDirName,"\FilteredBreath_" , dirListingReference(ii).name), "Ref")
                                                      
                                    end 

                            % STEP 5- Saving of radar and reference data  matrixes for estimated breath frequency. 

                              save (strcat(FileLocationProcessed, TempDirName, "\EstimatedBreathFrequencyRadar"), ...
                                         "EstimatedFrequencyRadar")
                             save (strcat(FileLocationProcessed, TempDirName, "\RadarSignalToNoise"), ...
                                         "RadarSignalToNoise")
                             save (strcat(FileLocationProcessed, TempDirName, "\EstimatedBreathFrequencyRef"), ...
                               "EstimatedFrequencyRef")
                              save (strcat(FileLocationProcessed, TempDirName, "\RefSignalToNoise"), ...
                                         "RefSignalToNoise")

                        end 




