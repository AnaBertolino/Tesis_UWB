clc  
clear all 
close all 

%% RawRadarDataVisualization.m sample code presents a simple code for raw radar data 
%% visualization. Reader can select a random raw radar data type in the Raw Radar Data File
%% and visualize it. Salsa Ancho radar module is used for data collection. This code is written 
%% by Cansu EREN to provide researchers to analyze the human radar data with reference
%% data for future scientific studies. 
 
%% This code is written in MATLAB R2020a. 
%% Author: Cansu EREN 
%% Copyright (C) 2023 Cansu EREN 
 
%% This program is free software: you can redistribute it and/or modify it under the terms of
%% the GNU General Public License as published by the Free Software Foundation, version 3
%% of the License. Further information, please check GNU General Public License v3.0 .txt. 

%% This data is used under the terms of  ODC Open Database License (ODbL). Further 
%% information, please check GNU General Public License(ODbL).txt 
 
%%%%%%%%%%%%%%TABLE OF NAMINGS IN RAW RADAR DATA%%%%%%%%%%%%%%%%%%%
% (...DeltaR...) refers to variable range in "cm"s. 
% (....Angle...) is given as degree that corresponds to measumenent angle of radar device while
% doing experiments. 
% (...Band#...) refers to Lidar measurement taken for radar bandwidth selection. 
% (...FaceDown...), (...Lateral...) and (...Supine...) refer to human body orientations towards the...
% lidar sensor. 
% (...Trial#) refers to trial number of data.                                           

                                            %% Workspace Name%% 
% Data naming for all data selections are bScan in MATLAB workspace. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% STEP 1-Load radar parameters. 
SampleM=1099; % Number of samples in slow time. 
RangeRes=   0.0039; %Range resolution (m). 
TimeResRadar=0.0533; % Radar data time resolution (s).  
TimeDurRadar=TimeResRadar*SampleM; % Radar time duration (s). 
Range=1; % Range (m) 
tRadar=0: TimeResRadar: TimeDurRadar-TimeResRadar; % Radar observation time vector (s).  
rr=RangeRes:RangeRes:Range-RangeRes; % Range vector (m). 

% STEP 2 - Selection of raw dataset file location. 
[File,Path] = uigetfile({'*.mat'}, 'Select a file'); 
load(strcat(Path, File)); 
cd(Path)

% STEP 3- Visualize data files.  
  figure
  imagesc(tRadar,rr, bScan')
  xlabel('Time(s)', 'FontSize', 10, 'FontWeight','bold')
  ylabel('Range(m)',  'FontSize', 10, 'FontWeight','bold')
  title(strcat(File),  'FontSize', 12, 'FontWeight','bold', 'Color', 'k', 'Interpreter', 'none')
  colormap(gray)
  colorbar
 



