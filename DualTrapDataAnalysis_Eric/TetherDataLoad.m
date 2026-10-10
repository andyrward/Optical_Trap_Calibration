%%
Fmax=75;

handles.subtethers=[];
r=handles.ct;
tethers=handles.tethers;

t1=tethers(r,1);
t2=tethers(r,2);t1
tend=tethers(r,3);

handles.x1sc=handles.x1s(t1:tend);
handles.x2sc=handles.x2s(t1:tend);
handles.x1m=handles.x1s(t1:t2);

handles.Fc=handles.F(t1:tend);


handles.x1c=handles.x1(t1:tend);
handles.x2c=handles.x2(t1:tend);

handles.extc=handles.ext(t1:tend);

handles.tc=handles.time(t1:tend);
handles.statec=handles.state(t1:tend);

handles.x1sc=handles.x1s(t1:tend);
handles.x1m=handles.x1s(t1:t2);
%%
%%

figure(2);

subplot(3,1,1);
hold off;
plot(handles.tc,handles.Fc);

handles.FE=find(handles.statec==27);

f=find(handles.state==31);

if ~isempty(f)
    if ~isempty(handles.FE)
        handles.subtethers=StateFinder(27,handles.statec);
    end
    handles.subtethers=[handles.subtethers;StateFinder(31,handles.statec)];
else
    if ~isempty(handles.FE)
        handles.subtethers=StateFinder(27,handles.statec);
    end
end

%%
%%
if ~isempty(handles.subtethers)
    
    handles.NumFERuns=(length(handles.subtethers(:,1)));
    set(handles.FEruns,'String',[num2str(handles.NumFERuns),' Total FE Runs']);
    
    set(handles.CurrentFE,'String',num2str(1));
    
    handles.cFE=1;
    
    FE=handles.FE;
    
    hold on;
    
    plot(handles.tc(FE),handles.Fc(FE),'r');
    %%
    
    
    handles.tsub=handles.tc(handles.FE);
    handles.z=handles.extc(FE);
    handles.Ft=handles.Fc(FE);
    
    if ~isempty(handles.z)
     fd=find(handles.z<0);
    handles.z(fd)=[];
    handles.Ft(fd)=[];
    handles.tsub(fd)=[];
    
    end
    subplot(3,1,3);
    hold off;
    
    plot(handles.z,handles.Ft);
    xlim([1500,5000]);
    ylim([-1,Fmax]);
    
   
    
    hold on;
    
    
    
    N=str2num(get(handles.NumPoints, 'String'));
    difft=[];
    dF=[];
    dz=[];
    for i=N+1:length(handles.Ft)-N-1
        
        difft(end+1)=handles.tsub(i);
        dF(end+1)=mean(handles.Ft(i+1:i+1+N))-mean(handles.Ft(i-N:i-1));
        dz(end+1)=mean(handles.z(i+1:i+1+N))-mean(handles.z(i-N:i-1));
        
    end
    
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
    
end

if ~isempty(handles.subtethers)
    ref=StateFinder(24,handles.statec(handles.subtethers(1,2):end));
    handles.refpull=ref+handles.subtethers(1,2)-1;
    
    if ~isempty(ref)
        i1r=handles.refpull(handles.cFE,1);
        i2r=handles.refpull(handles.cFE,2);
        handles.Frp=handles.Fc(i1r:i2r);
        handles.zrp=handles.extc(i1r:i2r);
        
        
        subplot(3,1,3);
        hold on;
        plot(handles.zrp,handles.Frp,'y');
        xlim([200,6000]);
        ylim([-1,Fmax]);
    end
    
end

if ~isempty(handles.zref)
    
    subplot(3,1,3);
    hold on;
plot(handles.zref,handles.Fref,'r');
end
set(handles.CurrentTether,'String',num2str(handles.ct));