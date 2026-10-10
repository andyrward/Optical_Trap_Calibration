function [ Fdata, FEdata ] = CombineData( Ntot )
%Combines all of the rupture fitting results from different fitting
%sessions Ntot is the number of files you want to combine.
[filename,pathname] = uigetfile('*.*','Select file');
filename
load(filename);
Fdata=FtransTemp;

FEdata=FEtransTemp;

N=Fdata(end,8);
for i = 2:Ntot
[filename,pathname] = uigetfile('*.*','Select file');
load(filename);
filename

FtransTemp(:,8)=FtransTemp(:,8)+N;
Fdata=[Fdata;FtransTemp];
FEdata=[FEdata,FEtransTemp];
N=Fdata(end,8);
end

end

