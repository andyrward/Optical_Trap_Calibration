function [ Fout,z,dout ] = WLC_Mod_DT( lp,Lc,K,Fmax,delta,Rsum,k1,k2,dstart,Fg )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

check=1;
i=1;
d=dstart;

while check==1

eq=@(F)(4.1/lp*(1/4/(1-(d-F*(1/k1+1/k2)-Rsum)/Lc+F/K)^2-1/4+(d-F*(1/k1+1/k2)-Rsum)/Lc-F/K)-F);

Fout(i)=fzero(eq,Fg);
Fout(i);

z(i)=d-Fout(i)*(1/k1+1/k2)-Rsum;

dout(i)=d;
if Fout(i)>=Fmax
    check=0;
end
Fg=Fout(i);
d=d+delta;
i=i+1;


end


Fout=Fout';
end

