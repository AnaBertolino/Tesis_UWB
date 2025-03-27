

clc
clear all 
close all

%% PaperFigures.m file visualize the human breath radar data and reference data and 
%% save them inside Processed Data File. Reader should select a random data type in the 
%% Processed Data File and visualize figures. This code is written by Cansu EREN to provide
%% researchers to analyze the human radar data with reference data for future scientific
%% studies. Salsa Ancho radar module and ST life.augmented VL53L0X lidar sensor is used for
%% data collection. 

%% This code is written in MATLAB R2020a. 
%% Author: Cansu EREN 
%% Copyright (C) 2023  Cansu EREN 

%% This program is free software: you can redistribute it and/or modify  it under the terms
%% of the GNU General Public License as published by the Free Software Foundation,  
%% version 3 of the License. Further information, please check GNU General Public 
%% License v3.0 .txt. 

%% This data is used under the terms of  ODC Open Database License (ODbL). Further 
%% information, please check GNU General Public License(ODbL).txt 

% STEP 1 - Selection of main dataset file location.

QuestionStruct.Interpreter='tex';       
QuestionStruct.Default='Yes';      
Question=...
       questdlg('\fontsize{10}\color{black}Select main dataset file location to load',...
       'Question', 'Yes', 'No', 'Cancel', QuestionStruct);


switch Question
    case 'Yes'
      File= uigetdir('C:\'); 
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

% STEP 2 - Plot figures.

for jj=1:3 

FileLocationProcessed=fullfile(File, 'Processed Data', strcat('Bandwidth', num2str(jj))); 
cd(FileLocationProcessed)
 load("EstimatedBreathFrequencyRef.mat"); 
 load("EstimatedBreathFrequencyRadar.mat"); 
 load("RefSignalToNoise.mat"); 
 load("RadarSignalToNoise.mat"); 
Alpha=mean(RadarSignalToNoise); 

figure(jj)
 histogram(EstimatedFrequencyRadar,[min(EstimatedFrequencyRadar):0.1:2],'FaceColor', "#0072BD"	)
 hold on
 histogram(EstimatedFrequencyRef,[min(EstimatedFrequencyRef):0.1:2],'FaceColor',"#D95319"	)
 legend({strcat("Alpha=", num2str(Alpha, '%.2f'), " dB"), ""})
 xlabel('Detected Breath Frequencies (Hz)', 'FontSize', 20, 'FontWeight','bold')
 ylabel('Number of Trials',  'FontSize', 18, 'FontWeight','bold', 'Color', 'k', 'Interpreter','none')
title(strcat('Bandwidth',num2str(jj)),  'FontSize', 18, 'FontWeight','bold', 'Color', 'k', 'Interpreter','none')
 set(gca, 'FontSize', 18)
 saveas(gca,strcat(FileLocationProcessed, '\Histogram', num2str(jj), '.fig'))

end   


for jj=1:3

FileLocation=fullfile(File, 'ErrorCalculation'); 
cd(FileLocation)
DataName=strcat('ErrorBandwidth', num2str(jj), ".mat"); 
 load(DataName); 
figure(jj+3)
 histfit(Error)
  legend({strcat("Mean=", num2str(mean(Error*100), '%.2f'), " dB"), ""})
 xlabel('Relative Error (Hz)', 'FontSize', 20, 'FontWeight','bold')
 ylabel('Error(%)',  'FontSize', 18, 'FontWeight','bold', 'Color', 'k', 'Interpreter','none')
title(strcat('Bandwidth',num2str(jj)),  'FontSize', 18, 'FontWeight','bold', 'Color', 'k', 'Interpreter','none')
 set(gca, 'FontSize', 18)
 saveas(gca,strcat(FileLocationProcessed, '\Error', num2str(jj), '.fig'))

end 
 
cd(FileLocationProcessed)



