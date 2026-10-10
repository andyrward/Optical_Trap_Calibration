function [ F ] = DualWLCMod( lp,Lc,K,z,lb,dL )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here


z1=z(1:lb-1);
z2=z(lb:end);

F1=WLC_Mod(lp,Lc,K,z1);
F2=WLC_Mod(lp,Lc+dL,K,z2);
s1=size(F1);
s2=size(F2);

if (s1(1)==1)
    F1=F1';
end
if s2(1)==1
    
    F2=F2';
    
end
F=[F1;F2];


end

