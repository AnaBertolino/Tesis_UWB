clc
clear all 
close all 

%% ErrorCalculation.m file presents error analysis between human breath radar data and reference data. 
%% This code is written by Cansu EREN to provide researchers to analyze the human radar data with reference data for 
%% future scientific studies. 
%% The dataset address should be C:\Users\...\...\DataSet which is main dataset file location.  
%% This code is written in MATLAB R2020a. 
%% Author: Cansu EREN 
%% Copyright (C) 2023  Cansu EREN 

%% This program is free software: you can redistribute it and/or modify  it under the terms 
%% of the GNU General Public License as published by the Free Software Foundation, version
%% 3 of the License. Further information,please check GNU General Public License v3.0 .txt. 

%% This data is used under the terms of  ODC Open Database License (ODbL). Further 
%% information, please check GNU General Public License(ODbL).txt 

%%%%%%%%%%%%%%TABLE OF NAMINGS IN ERROR CALCULATION %%%%%%%%%%%%%%%%%%%
%ErrorBandwidth# refers to the relative error calculation between reference
%and radar data. 
% DataTableBandwidth# refers to table that contains estimation of breath
% frequencies, and  signal to noise ratios of data. 
% LocalErrorTable# are the error analysis of each measurement scene
% regarding angle, bandwidth, range and human posture. 

                                    %%TABLE OF NAMINGS IN MATLAB WORKSPACE%%   
% --Workspace Data Name--                          --DataName-- 
%               Data Table                                                  DataTable              %
%           ErrorBandwidth#                                           Error
%           LocalErrorTable#                                            ErrorTable 

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

TempVector=([1:18].*8) ;
TempVector=[1 TempVector 0]; 

%  STEP 2 - Error Analysis

for jj=1:3

FileLocation=fullfile(File, 'Processed Data', strcat('Bandwidth', num2str(jj))); 
SubFileLocationRadar=fullfile(File, 'Raw Radar Data', strcat('Bandwidth', num2str(jj))); 
FileLocationError=fullfile(File, 'ErrorCalculation\');
DataName=fullfile(FileLocationError, strcat('Bandwidth', num2str(jj))); 
dirListingRadar = dir([SubFileLocationRadar, '/', '*.mat']);
cd(FileLocation)

 load("EstimatedBreathFrequencyRef.mat"); 
 load("EstimatedBreathFrequencyRadar.mat"); 
 load("RefSignalToNoise.mat"); 
 load("RadarSignalToNoise.mat");  


             for kk=1:length(dirListingRadar)
                         Variables=dirListingRadar(kk).name; 
                         VariableName(kk)=string(Variables); 
             end 


DataTable=table(VariableName',EstimatedFrequencyRadar',RadarSignalToNoise',EstimatedFrequencyRef',RefSignalToNoise'); 
DataTable.Properties.VariableNames = ["DataName","EstimatedFrequencyRadar", "RadarSignalToNoise", "EstimatedFrequencyRef", "RefSignalToNoise"];
 Error=abs(DataTable.EstimatedFrequencyRadar-DataTable.EstimatedFrequencyRef)./DataTable.EstimatedFrequencyRef; 

save (strcat(FileLocationError, "\DataTable","Bandwidth", num2str(jj)), "DataTable")
save (strcat(FileLocationError, "\Error","Bandwidth", num2str(jj)), "Error")

  
                for ii=1:18

                    if ii==1
                                 ErrorTable=table(VariableName(TempVector(ii):TempVector(ii+1))', Error(TempVector(ii):TempVector(ii+1))); 
                                 ErrorTable.Properties.VariableNames = ["VariableName","Error"];
                                 save (strcat(FileLocationError, "Bandwidth", num2str(jj), "\LocalErrorTable", num2str(ii)), "ErrorTable")
                    else
                                 ErrorTable=table(VariableName(TempVector(ii)+1:TempVector(ii+1))', Error(TempVector(ii)+1:TempVector(ii+1))); 
                                 ErrorTable.Properties.VariableNames = ["VariableName","Error"]; 
                                 save (strcat(FileLocationError, "Bandwidth", num2str(jj), "\LocalErrorTable", num2str(ii)), "ErrorTable")

                end 

end 
end 






