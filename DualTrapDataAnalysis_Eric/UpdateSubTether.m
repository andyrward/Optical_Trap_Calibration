i1=handles.subtethers(handles.cFE,1);
i2=handles.subtethers(handles.cFE,2);


i1r=handles.refpull(handles.cFE,1);
i2r=handles.refpull(handles.cFE,2);
handles.Frp=handles.Fc(i1r:i2r);
handles.zrp=handles.extc(i1r:i2r);
handles.z=handles.extc(i1:i2);
handles.Ft=handles.Fc(i1:i2);
handles.tsub=handles.tc(i1:i2);

%% Calculate dF

N=str2num(get(handles.NumPoints, 'String'));
difft=[];
dF=[];
dz=[];
for i=N+1:length(handles.Ft)-N-1
    
    difft(end+1)=handles.tsub(i);
    dF(end+1)=mean(handles.Ft(i+1:i+1+N))-mean(handles.Ft(i-N:i-1));
    dz(end+1)=mean(handles.z(i+1:i+1+N))-mean(handles.z(i-N:i-1));
    
end

figure(2);
subplot(3,1,2);
handles.deltamode=get(handles.CheckdZ,'value');

if handles.deltamode==0;
    
    plot(difft,dF);
else
    plot(difft,dz);
end




handles.dF=(dF);
handles.dz=dz;
handles.difft=difft;

subplot(3,1,3);
hold off;
plot(handles.z,handles.Ft);
hold on;
plot(handles.zrp,handles.Frp,'y');

if ~isempty(handles.zref)
    hold on;
    plot(handles.zref,handles.Fref,'r');
    
end
xlim([500,3100]);
ylim([0,80]);
subplot(3,1,1)
hold off;
plot(handles.tc,handles.Fc,handles.tsub,handles.Ft,'r');
