function [fd,err,Ft,x] = FitWLC( lpg,Lcg,z,F,zmin,zmax,mod,dF )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

f=find(z>=zmin &z<=zmax);

x=z(f);
F=F(f);

switch(mod)
    
    case(0)
        %Inextensible WLC
        Fmodel=@(b,x)(Fwlc(b(1),b(2),x));
        
        bg=[lpg;Lcg];
        
        [fd,r,J,cov,mse]=nlinfit(x,F,Fmodel,bg);
        
        % ci1=nlparci(fd,r,'Jacobian',J);
        
        ci2=nlparci(fd,r,'covar',cov);
        
        err=(ci2(:,2)-ci2(:,1))/2;
        
        Ft=Fwlc(fd(1),fd(2),x);
    case(1)
        %Extensible WLC
        
        [zbb,Fb,Wb] = FZBin( x, F,dF);
        
        Fmodel=@(b,x)WLC_Mod(b(1),b(2),b(3),x)
        
        bg=[lpg;Lcg;700];
        
        [fd,r,J,cov,mse]=nlinfit(zbb',Fb',Fmodel,bg);
        
        ci2=nlparci(fd,r,'covar',cov);
        
        err=(ci2(:,2)-ci2(:,1))/2;
        Ft=WLC_Mod(fd(1),fd(2),fd(3),x);
end


end

