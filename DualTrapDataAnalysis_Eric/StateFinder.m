function [ StatePoints ] = StateFinder (StateNumber,State)
%Finds the time points for different state regions for dual trap
%experiments by searching for discontiuities in the state

f=find(State==StateNumber);

if ~isempty(f)
    
df=diff(f);
fstate=find(df~=1);

if ~isempty(fstate)
    


for i=1:length(fstate+1)
   
    if i==1
        t1=f(1);
        t2=f(fstate(i));
    else
        t1=f(fstate(i-1)+1);
        t2=f(fstate(i));
    end
    
    StatePoints(i,:)=[t1,t2];
    
    
end

 StatePoints(end+1,:)=[f(fstate(end)+1),f(end)];

else
 
    StatePoints(1,:)=[f(1),f(end)];
    
end


else
    StatePoints=[];
    
end

