function [ Fout ] = WLC_Mod( lp,Lc,K,z )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

for i=1:length(z)

eq=@(F)(4.1/lp*(1/4/(1-z(i)/Lc+F/K)^2-1/4+z(i)/Lc-F/K)-F);
if i==1
Fout(i)=fzero(eq,Fwlc(lp,Lc,z(i)));
else
    
Fout(i)=fzero(eq,Fout(i-1));
end


Fout=Fout';
end

