function [m,delta,pv,hv] = RunningStep(x,n,dn,vis,tail)
%UNTITLED15 Summary of this function goes here
%   Detailed explanation goes here


if vis==1
    
    figure
end

Ntot=numel(x);

Nj=numel(1:dn:Ntot-2*n);
is=1:dn:Ntot-2*n;
index=(is-1)/dn+1;
m=zeros(1,Nj);
pv=zeros(1,Nj);
delta=zeros(1,Nj);
hv=zeros(1,Nj);

for j=1:Nj
    i=is(j);
    x1=x(i:i+n-1);
    x2=x(i+n:i+n+n-1);
    [h,p]=ttest2(x1,x2,'Vartype','unequal','Tail',tail);
    m(j)=i+n;
    pv(j)=p;
    delta(j)=mean(x2)-mean(x1);
    hv(j)=h;
    if vis==1
        plot(x);hold on;
      p1=[i:i+n-1];
        plot(p1,x1,'ro');
        p2=[i+n:i+n+n-1];
        plot(p2,x2,'go');hold off;
        pause(0.01);
    end
    
end

end

