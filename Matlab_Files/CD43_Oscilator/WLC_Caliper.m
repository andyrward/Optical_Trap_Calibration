function [etot] = WLC_Caliper(ldna,lp,Fdna,K,lloop,lploop)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

[edna] = WLC_ext(ldna,lp,Fdna,K);
eloop = WLC_ext(lloop,lploop,Fdna,K);

etot=edna+eloop;

end