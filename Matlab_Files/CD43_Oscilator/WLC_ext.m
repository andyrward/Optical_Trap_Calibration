function [edna] = WLC_ext(ldna,lp,Fdna,K)
%dnaforce calculates the extension of dna as aunction of force, the dna can
%also be in series with spring with stiffness ks. ldna is length of the
%dna, lp is the persistance length, K is the stretch modulus of the dna
%~1000pN
%   uses extensible wormlike chain model, generally valid in the 2-25pN
%   regime from Gelles and Block paper 1997.

%K=1000;
edna = ldna*(1-1/2*(4.1./Fdna/lp).^(0.5)+Fdna/K);


end

