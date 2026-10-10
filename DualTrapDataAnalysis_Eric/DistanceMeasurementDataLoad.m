
[filename,pathname]=uigetfile('*.*');
d1=csvread([pathname,filename]);
figure

 dl=d1(1:2:end,1);
    dh=d1(2:2:end,1);
    plot(dl);hold on; plot(dh);
bounds=input('Enter Region :');
if isempty(bounds)

else
    
    dl=dl(bounds(1):bounds(2));
    dh=dh(bounds(1):bounds(2))
end

%Look at the mean delta extension for the same distance
dEcl=dl(2:end)-dl(1:end-1);
dEch=dh(2:end)-dh(1:end-1);
dEc=[dEcl;dEch];

dE=dh-dl;
b=0.3;
xb=min(dl(:,1))-b:b:max(dh(:,1))+b;
ch=hist(dh,xb);
cl=hist(dl,xb);
figure
histfit(dl);
hold on;
histfit(dh);
stairs(xb,cl,'b');
stairs(xb,ch,'r')
figure
histfit(dE);
hold on;
histfit(dEc);

