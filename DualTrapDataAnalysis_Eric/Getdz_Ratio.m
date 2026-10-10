function [ dz_out,F_out ] = Getdz_Ratio( Fbefore,Fafter,z_before,z_after, dF )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
Fmin=min(Fbefore);
Fmax=max(Fbefore);

Fsearch=Fmin:dF:Fmax;

dz=[];
for i = 1:length(Fsearch)
    
    switch(i)
        
        case(1)
            fb=find(Fbefore<Fsearch(i+1));
            fa=find(Fafter<Fsearch(i+1));
        case(length(Fsearch))
            fb=find(Fbefore >Fsearch(i));
            fa=find(Fafter >Fsearch(i));
        otherwise
            fb=find(Fbefore<Fsearch(i+1) & Fbefore >Fsearch(i));
            fa=find(Fafter<Fsearch(i+1) & Fafter >Fsearch(i));
    end
    
    
    zb=z_before(fb);
    za=z_after(fa);
    
    dz(i,1)=mean(za)-mean(zb);
    dz(i,2)=mean(zb);
    dz(i,3)=mean(za);
    dz(i,4:5)=[std(zb),std(za)];
    dz(i,6:7)=[length(zb),length(za)];
    dz(i,8)=sqrt(std(zb)^2/length(zb)+std(za)^2/length(za));
    
    
end


fn=find(dz(:,6)<=1);
dz(fn,:)=[];
Fsearch(fn)=[];
fn=find(dz(:,7)<=1);
dz(fn,:)=[];
Fsearch(fn)=[];
sum(dz(:,1).*dz(:,8))/sum(dz(:,8));
mean(dz(:,1));

dz_out=dz(:,1);
F_out=Fsearch;
end

