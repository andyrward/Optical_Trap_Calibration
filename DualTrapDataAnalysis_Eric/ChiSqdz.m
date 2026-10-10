function [ ChiSq ] = ChiSqdz( k, zo, dz, zb,Fb,za,Fa )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here


za_m=Fa/k+zo+dz;
zb_m=Fb/k+zo;

dZb=(zb_m-zb).^2;
dZa=(za_m-za).^2;

ChiSq=sum(dZb)+sum(dZa);

end

