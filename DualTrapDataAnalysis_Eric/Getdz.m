function[k,dz,zo,E] = Getdz(zb,Fb,za,Fa)

pb=polyfit(Fb,zb,1);
pa=polyfit(Fa,za,1);

%za=Fa/k+zo+dz
%zb=Fb/k+zo

dzg=pa(2)-pb(2);
zog=pb(2);
kg=1/pb(1);




display(['dz initial estimate: ' num2str(dzg),' nm']);
Chisq=@(x)ChiSqdz(x(1),x(2),x(3),zb,Fb,za,Fa);

p=fminsearch(Chisq,[kg,zog,dzg]);

display(['dz Final estimate: ' num2str(p(3)),' nm']);
k=p(1);
dz=p(3);
zo=p(2);


dze=0.8*dz:0.01*dz:1.2*dz;
for i=1:length(dze)
X(i)=Chisq([k,zo,dze(i)]);
end

pe=polyfit(dze,X,2);

E=sqrt(1/pe(1));
end

