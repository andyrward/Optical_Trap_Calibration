function [path,filedata,fileVoltage] = getfiles()
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[file,path] = uigetfile('*.*');

f=find(file=='.')

if ~isempty(f)

    fileVoltage=strcat(file(1:f-1),'_Voltage',file(f:end));

else

  fileVoltage=strcat(file(1:f-1),'_Voltage.dat');

end

filedata=file;

end