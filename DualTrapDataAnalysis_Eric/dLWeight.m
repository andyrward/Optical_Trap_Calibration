function [ W ] = dLWeight( dL,s,m )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

W=1/s/sqrt(2/pi)*exp(-(dL-m).^2/2/s^2);
end

