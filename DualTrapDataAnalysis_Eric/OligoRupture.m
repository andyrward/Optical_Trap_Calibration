function [ Frup ] =OligoRupture( lr,Nbp )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

xs=(7+0.7*Nbp)/10;

v=10^(3-0.5*Nbp);




Frup=4/xs*log(lr*xs/4/v);

end

