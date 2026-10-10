function [ ChiSq ] = ChiSqdz_WLC( lp,Lc, dz, zb,Fb,Wb,za,Fa,Wa )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here


Fa_m=(1/4*(1-za/Lc).^(-2)- 1/4 + za/Lc)*4.1/lp;
Fb_m=(1/4*(1-zb/(Lc-dz)).^(-2)- 1/4 + zb/(Lc-dz))*4.1/lp;

dFb=((Fb_m-Fb).^2./Wb);
dFa=((Fa_m-Fa).^2./Wa);

ChiSq=sum(dFb)+sum(dFa);

end

