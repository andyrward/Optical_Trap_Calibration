function[lp,dz,Lc,E] = Getdz_WLC(zb,Fb,za,Fa,dzstart,Lf,dF)



[zbb,Fsb,Wb] = FZBin( zb, Fb,dF );
[zab,Fsa,Wa] = FZBin( za, Fa,dF );


Chisq=@(x)ChiSqdz_WLC(x(1),x(2),x(3),zbb,Fsb,Wb,zab,Fsa,Wa);

lpg=25;Lcg=Lf;dzg=dzstart;

p=fminsearch(Chisq,[lpg,Lcg,dzg]);


lp=p(1);
dz=p(3);
Lc=p(2);


dze=0.8*dz:0.01*dz:1.2*dz;
for i=1:length(dze)
    X(i)=Chisq([lp,Lc,dze(i)]);
end

pe=polyfit(dze,X,2);

E=sqrt(1/pe(1));

end

