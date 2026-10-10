function [ dz ] = dzF(Fb,zb,Fa,za,dF,F )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here


f=find(Fb>F-dF & Fb<F+dF);

z1=mean(zb(f));

f=find(Fa>F-dF & Fa<F+dF);

z2=mean(za(f));

dz=[z2-z1,z1,z2];
end

