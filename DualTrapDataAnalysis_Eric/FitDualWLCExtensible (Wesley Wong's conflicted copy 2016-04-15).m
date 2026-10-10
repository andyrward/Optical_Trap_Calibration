function [ fd,err,F1,z1,F2,z2] = FitDualWLCExtensible( lpg,Lcg,zb,Fb,za,Fa,Fmin,Fmax,dF,dLg )
%Fit two extensible wormlike chain models simultaneiously to two force
%extension curves
%   Detailed explanation goes here

f=find(Fb>Fmin & Fb<Fmax);

zm=zb(f);
Fm=Fb(f);
zmin=min(zm);
zmax=max(zm);

f=find(zb>zmin & zb< zmax);

zb=zb(f);
Fb=Fb(f);





f=find(Fa>Fmin & Fa<Fmax);

zm=za(f);
Fm=Fa(f);
f=find(zm >zb(end)+2*dLg);

zm(f)=[];
Fm(f)=[];


zmin=min(zm);
zmax=max(zm);

f=find(za>zmin & za< zmax);

za=za(f);
Fa=Fa(f);




if dF>0
    [zbb,Fbb,Wb] = FZBin( zb, Fb,dF);
    [zab,Fab,Wa] = FZBin( za, Fa,dF);
    
else
    
    zbb=zb';Fbb=Fb';Wb=zeros(1,length(Fb))+1; 
    zab=za';Fab=Fa';Wa=zeros(1,length(Fa))+1; 
end

x=[zbb';zab'];

F=[Fbb';Fab'];
lb=length(zbb)+1;

Fmodel=@(b,x)DualWLCMod(b(1),b(2),b(3),x,lb,b(4));

W=[Wb';Wa'];

bg=[lpg,Lcg,500,dLg];

[fd,r,J,cov,mse]=nlinfit(x,F,Fmodel,bg,'Weights',W);

ci2=nlparci(fd,r,'covar',cov);

err=(ci2(:,2)-ci2(:,1))/2;

figure(3)
subplot(2,1,1)
hold off
plot(x,F);
hold on;
plot(fd)

zmin=min(zb);
z1=zmin:zbb(end);
z2=zmin:zab(end);

F1=WLC_Mod( fd(1),fd(2),fd(3),z1 );
F2=WLC_Mod( fd(1),fd(2)+fd(4),fd(3),z2 );



end

