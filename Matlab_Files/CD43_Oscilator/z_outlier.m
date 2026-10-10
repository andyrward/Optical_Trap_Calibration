function [zout] = z_outlier(z,zmin,zmax)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

f=find(z>zmax | z<zmin);

if ~isempty(f)

    for i=1:length(f)
        ic=f(i);
        if (ic>11 & ic<length(z)-10)
            z(ic)=median([z(ic-10),z(ic+10)]);
        else
            if ic<11
            z(ic)=z(ic+1);
            else
                z(ic)=z(ic-1);
            end
        end

end
end
zout=z;


end