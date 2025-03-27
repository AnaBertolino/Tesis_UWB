clc  
clear all 
close all 

%% ReferenceDataVisualization.m sample code presents a simple code for reference data 
%% visualization. Reader can select a random reference data type in the Reference Data File
%% and visualize it. The reference signals are collected by ST life.augmented VL53L0X lidar 
%% sensor. This code is written by Cansu EREN to provide researchers to analyze the human
%% radar data with reference data for future scientific studies. 

%% This code is written in MATLAB R2020a. 
%% Author: Cansu EREN 
%% Copyright (C) 2023 Cansu EREN 

%% This program is free software: you can redistribute it and/or modify it under the terms..
%% of the GNU General Public License as published by the Free Software Foundation,  
%% version 3 of the License. Further information, please check GNU General Public License
%% v3.0 .txt. 

%% This data is used under the terms of  ODC Open Database License (ODbL). Further 
%% information, please check GNU General Public License(ODbL).txt 

%%%%%%%%%%%%%%TABLE OF NAMINGS IN REFERENCE DATA%%%%%%%%%%%%%%%%%%%
% (Ref...) refers to initial name of reference data.
% (...DeltaR...) refers to variable range in "cm"s. 
% (....Angle...) is given as degree that corresponds to measumenent angle of radar device while
% doing experiments. 
% (...Band#...) refers to Lidar measurement taken for radar bandwidth selection. 
% (...FaceDown...), (...Lateral...) and (...Supine...) refer to human body orientations towards the...
% lidar sensor. 
% (...Trial#) refers to trial number of data. 

                                            %% Workspace Name%% 
% Data naming for all data selections are "Ref" in MATLAB workspace.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% STEP 1 - Load reference equipment parameters.  

TimeResRef=0.325; % Lidar time resolution (s). 
SamplingFreqRef= 1/TimeResRef; % Sampling frequency of Lidar data (Hz). 
tRef= (1:180).*TimeResRef; % Lidar observation time vector (s). 

% STEP 2 - Selection of reference dataset file location.

[File,Path] = uigetfile({'*.mat'}, 'Select a file'); 
load(strcat(Path, File)); 
cd(Path)

% STEP 3- Visualize data files.  

figure 
plot(tRef, Ref)
xlabel('Time(s)', 'FontSize', 10, 'FontWeight','bold')
ylabel('Amplitude(mm)',  'FontSize', 10, 'FontWeight','bold')
title(strcat(File),  'FontSize', 12, 'FontWeight','bold', 'Color', 'k', 'Interpreter','none')


   



