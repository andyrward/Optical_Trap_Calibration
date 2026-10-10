

for r=1:length(tethers(:,1))
    
    t1=tethers(r,1);
    t2=tethers(r,2);
    tend=tethers(r,3);
    xm=mean(handles.x1s(t1:t2));
    handles.x1s(t1:tend)=handles.x1s(t1:tend)-xm;
    
end

handles.k=handles.Power*handles.kp;
handles.F=handles.x1s.*handles.k';
D=str2num(get(handles.avgD,'String'));

if handles.trackcheck==1
    handles.ext=handles.x2-handles.x1-D;
    
else
    
    handles.ext=handles.x2s-handles.x1s-2;
end

figure(1);

subplot(2,1,1);hold off;plot(handles.time,handles.F);



FE=StateFinder(27,handles.state);


for i=1:length(FE(:,1))
    
    t1=FE(i,1);
    t2=FE(i,2);
    subplot(2,1,1);
    
    hold on;
    
    
    
    plot(handles.time(t1:t2),handles.F(t1:t2),'r');
    
    subplot(2,1,2);
    if i>1
        hold on;
    else
        hold off;
    end
    plot(handles.ext(t1:t2),handles.F(t1:t2));
    
end