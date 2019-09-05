function [ config ] = readConfigJson()
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here

[fileName,filePath,~] = uigetfile('*.json','Select a Config file');
configFileFullPath = fullfile(filePath,fileName);

config = jsondecode(fileread(configFileFullPath));


end

