clc  
clear all 
close all 

%% ProcessedDataVisualization.m sample code presents a simple code for processed radar 
%% data visualization. Reader can select a random data type in the Processed Data 
%% File and visualize it. Salsa Ancho radar module and ST life.augmented VL53L0X lidar sensor 
%% is used for data collection. This code is written by Cansu EREN to provide researchers to...
%% analyze the human radar data with reference data for future scientific studies. 

%% This code is written in MATLAB R2020a. 
%% Author: Cansu EREN 
%% Copyright (C) 2023  Cansu EREN 

%% This program is free software: you can redistribute it and/or modify  it under the terms
%% of the GNU General Public License as published by the Free Software Foundation,  version
%% 3 of the License. Further information,please check GNU General Public License v3.0 .txt. 

%% This data is used under the terms of  ODC Open Database License (ODbL). Further 
%% information, please check GNU General Public License(ODbL).txt 

%%%%%%%%%%%%%%TABLE OF NAMINGS IN PROCESSED DATASET %%%%%%%%%%%%%%%%%
% The data file names are grouped in seven sections which are given below. 
% (1) SpectrumRadar_(...) spectrum data files of BreathRadar_(...)  using
% Fast Fourier Transform. Further information, see DataCode.m file.
% (2) Spectrum_Ref_(...) spectrum data files of  FilteredBreath_Ref_(...) using 
% Fast Fourier Transform. Further information, see DataCode.m file.
% (3) FilteredBreathRadar_(...) data files are the extracted radar breath signals from
% the background subtracted data. Further information, see DataCode.m file.
% (4) FilteredBreath_Ref_(...) data files are the measured reference breath signals with
% ST life.augmented VL53L0X lidar sensor.
% (5) BS_(...) data files are the data files after background removal using
% Linear Trend Subtraction. Further information, see DataCode.m file. 
% (6)  EstimatedBreathFrequencyRef  is the matrix files of detected breath frequencies of 
% reference. 
% (7) EstimatedBreathFrequencyRadar is the matrix files of detected breath frequencies
% of radar. 
% (8) RadarSignalToNoise is the matrix files of signal to noise ratios of
% breath signals. 
% (9) RefSignalToNoise is the matrix files of signal to noise ratios of
% reference breath signals. 

% (...DeltaR...) refers to variable range in "cm"s. 
% (....Angle...) is given as degree that corresponds to measumenent angle of radar device while
% doing experiments. 
% (...Band#...) refers to Lidar measurement taken for radar bandwidth selection. 
% (...FaceDown...), (...Lateral...) and (...Supine...) refer to human body orientations towards the...
% lidar sensor. 
% (...Trial#) refers to trial number of data. 

                                    %%TABLE OF NAMINGS IN MATLAB WORKSPACE%%   
% --Workspace Data Name--                          --DataName-- 
% BreathSignalRadar                                         FilteredBreathRadar_(...)
%  Ref                                                                  FilteredBreath_Ref_(...)
%  BS                                                                   BS_(...)
% SpectrumRadar                                             SpectrumRadar_(...)
% SpectrumRef                                                   Spectrum_Ref_(...)
% EstimatedBreathFrequencyRadar                 EstimatedBreathFrequencyRadar
% EstimatedBreathFrequencyRef                     EstimatedBreathFrequencyRef
%RadarSignalToNoise                                          RadarSignalToNoise
%RefSignalToNoise                                               RefSignalToNoise
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% STEP 1- Load radar parameters  

SampleM=1099; % Number of samples in slow time. 
RangeRes=   0.0039; %Range resolution (m). 
TimeResRadar=0.0533; % Radar  data time resolution (s).  
TimeDurRadar=TimeResRadar*SampleM; % Radar time duration (s). 
Range=1; % Range (m) 
SamplingFreqRadar=1/TimeResRadar; % Sampling frequency of  radar data (Hz). 
FreqResRadar=SamplingFreqRadar/SampleM; % Frequency resolution of radar (Hz). 
tRadar=0: TimeResRadar: TimeDurRadar-TimeResRadar; % Radar observation time vector (s).  
rr=0:RangeRes:Range-RangeRes; % Range vector (m). 
ffRadar=0: FreqResRadar: SamplingFreqRadar-FreqResRadar;%Frequency  vector radar(Hz). 
ffRadar=ffRadar(1:fix(length(ffRadar)/2));  % Frequency  vector radar (Hz). 
FreqDCRem=7; % Priori sample number to suppress DC frequency. 

%  STEP 2-Load reference  parameters 

TimeResRef=0.325; % Lidar time resolution (s). 
SamplingFreqRef= 1/TimeResRef; % Sampling frequency of Lidar data (Hz). 
tRef= (1:180).*TimeResRef; % Lidar observation time vector (s). 
SampleRef=length(tRef); % Number of samples Lidar. 
FreqResRef=SamplingFreqRef/SampleRef; % Frequency resolution of Lidar (Hz). 
ffRef=0:FreqResRef: SamplingFreqRef-FreqResRef; % Frequency resolution vector Lidar (Hz). 
ffRef=ffRef(1:fix(length(ffRef)/2));  % Frequency resolution vector Lidar (Hz). 

% STEP 3 - Selection of procesed dataset file location.
 FileLocation='Processed Data'; 
[File,Path] = uigetfile({'*.mat'}, 'Select a file', FileLocation); 
cd(Path) 
load(strcat(Path, File))

% STEP 4- Visualize data files. 

     if File(1:19) == "FilteredBreathRadar" 
                % Visualize radar breath data. 
                figure 
                plot(tRadar, BreathSignalRadar)
                xlabel('Time(s)', 'FontSize', 10, 'FontWeight','bold')
                ylabel('Amplitude',  'FontSize', 10, 'FontWeight','bold')
                title(strcat(File),  'FontSize', 12, 'FontWeight','bold', 'Color', 'k', 'Interpreter', 'none')

                             elseif File(1:18) == "FilteredBreath_Ref" 
                                        % Visualize reference  breath data. 
                                        figure 
                                        plot(tRef, Ref)
                                        xlabel('Time(s)', 'FontSize', 10, 'FontWeight','bold')
                                        ylabel('Amplitude',  'FontSize', 10, 'FontWeight','bold')
                                        title(strcat(File),  'FontSize', 12, 'FontWeight','bold', 'Color', 'k', 'Interpreter','none')
                        
                            elseif File(1:3)=="BS_" 
                                        % Visualize BS data. 
                                        figure 
                                        imagesc(tRadar, rr, BS)
                                        xlabel('Time(s)', 'FontSize', 10, 'FontWeight','bold')
                                        ylabel('Range(m)',  'FontSize', 10, 'FontWeight','bold')
                                        title(strcat(File),  'FontSize', 12, 'FontWeight','bold', 'Color', 'k', 'Interpreter', 'none')
                                        colormap (gray) 
                                        colorbar 
                        
                            elseif File(1:13)=="SpectrumRadar" 
                                        % Visualize spectrum data of radar breath. 
                                        figure 
                                        plot(ffRadar(FreqDCRem:end), SpectrumRadar(FreqDCRem:end))
                                        xlabel('Frequency(Hz)', 'FontSize', 10, 'FontWeight','bold')
                                        ylabel('Magnitude',  'FontSize', 10, 'FontWeight','bold')
                                        title(strcat(File),  'FontSize', 12, 'FontWeight','bold', 'Color', 'k', 'Interpreter', 'none')
                        
                           elseif File(1:12)== "Spectrum_Ref" 
                                        % Visualize spectrum data of reference breath. 
                                        figure 
                                        plot(ffRef(FreqDCRem:end), SpectrumRef(FreqDCRem:end))
                                        xlabel('Frequency(Hz)', 'FontSize', 10, 'FontWeight','bold')
                                        ylabel('Magnitude',  'FontSize', 10, 'FontWeight','bold')
                                        title(strcat(File),  'FontSize', 12, 'FontWeight','bold', 'Color', 'k', 'Interpreter', 'none')
                        
                        elseif File(1:16)=="RefSignalToNoise"
                                        % Load RefSignalToNoise.
                                        MessageStruct.Interpreter='tex'; 
                                        MessageStruct.WindowStyle='modal'; 
                                        Message=msgbox(...
                                            {'\fontsize{10}\color{black}Check the workspace to see signal to ratio matrix'},...
                                            'ReadMe', 'help', MessageStruct) ;   
                        
                        elseif File(1:18)=="RadarSignalToNoise"
                                  % Load RadarSignalToNoise.
                                        MessageStruct.Interpreter='tex'; 
                                        MessageStruct.WindowStyle='modal'; 
                                        Message=msgbox(...
                                            {'\fontsize{10}\color{black}Check the workspace to see signal to ratio matrix'},...
                                            'ReadMe', 'help', MessageStruct) ;   
                        
                        elseif File(1:27)=="EstimatedBreathFrequencyRef"
                                       % Load EstimatedBreathFrequencyRef. 
                                        MessageStruct.Interpreter='tex'; 
                                        MessageStruct.WindowStyle='modal'; 
                                        Message=msgbox(...
                                            {'\fontsize{10}\color{black}Check the workspace to see estimation matrix'},...
                                            'ReadMe', 'help', MessageStruct) ;    

                      elseif File(1:29)=="EstimatedBreathFrequencyRadar"    
                                        % Load EstimatedBreathFrequencyRadar. 
                                        MessageStruct.Interpreter='tex'; 
                                        MessageStruct.WindowStyle='modal'; 
                                        Message=msgbox(...
                                            {'\fontsize{10}\color{black}Check the workspace to see estimation matrix'},...
                                            'ReadMe', 'help', MessageStruct) ;    
             
        end 






