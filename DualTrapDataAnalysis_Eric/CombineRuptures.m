function [Fdata,FEdata] = CombineRuptures( N )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

Fdata=[];
FEdata={};
for i=1:N
    
    [FileName,PathName] = uigetfile;
    FileName
    
    load(FileName);
    Fdata=[Fdata;FtransTemp];
    
    for j=1:length(FEtransTemp)
        loc=length(FEdata)+1;
        FEdata{loc}=FEtransTemp{j};
    end
    
    
end

end

