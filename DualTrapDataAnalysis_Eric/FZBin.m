function [zbb,Fsb,W] = FZBin( zb, Fb,dF )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

Fsb=[min(Fb):dF:max(Fb)];
zbb=[];
W=[];
N=length(Fsb);

for i=1:N
    
    if i<length(Fsb)
        f=find(Fb>=Fsb(i) &Fb<Fsb(i+1));
    else
        f=find(Fb>Fsb(i-1) &Fb<=Fsb(i));
    end
    
    
    if length(f)>1
        zbb(end+1)=mean(zb(f));
        W(end+1)=std(zb(f))/sqrt(length(f));
    else
        zbb(end+1)=0;
        W(end+1)=length(f);
        
    end
end

f=find(W<=1);
Fsb(f)=[];zbb(f)=[];W(f)=[];

if length(Fsb)==0
    
    Fsb=Fb;
    zbb=zb;
    W=zeros(length(zbb),1)+1;
end

end

