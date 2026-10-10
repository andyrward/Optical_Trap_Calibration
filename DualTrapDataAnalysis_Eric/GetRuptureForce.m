F=handles.Ft;
z=handles.z;
t=handles.tsub;

handles.deltamode=get(handles.CheckdZ,'value');

if handles.deltamode==0;
    dF=-handles.dF;
else
    dF=handles.dz;
end
difft=handles.difft;

Min_dF=str2num(get(handles.Min_dF,'String'));
Max_dF=str2num(get(handles.Max_dF,'String'));

[pks,locs]= findpeaks(dF,'MINPEAKHEIGHT',Min_dF);

f=find(pks<Max_dF);




if ~isempty(f)
    
    figure;
    plot(difft,dF);
    
    index=locs(f);
    
    t_transition=difft(index)
    hold on;
    plot(t_transition,pks(f),'ro');
    
    if length(t_transition )>1
        t_transition
        i=input('Enter time index! : ');
        t_transition=t_transition(i);
    end
    
    
    
    index=find(t==t_transition);
    
    Ftrans=F(index);
    
    dIndex=str2num(get(handles.dIndex,'String'));
    
    
    index=index-dIndex;
    
    figure(2);
    subplot(3,1,1);
    hold on;
    plot([t(index),t(index)],[0,Ftrans],'g');
    dt=str2num(get(handles.dt,'string'));
    
    t1=t_transition-dt;
    t_shift=t(index);
    
    f=find(t<t_shift  & t>t1);
    tfit=t(f);
    ffit=F(f);
    
    
    fd=fit(tfit,ffit,'poly1');
    
    FpS=fd.p1;
    
    figure(3);
    hold off;
    plot(tfit,ffit);hold on; plot(fd);
    pr=predint(fd,[t_transition-1:0.01:t_transition],0.68);
    hold on;
    
    plot([t_transition-1:0.01:t_transition],pr(:,1),'g');
    plot([t_transition-1:0.01:t_transition],pr(:,2),'g');
    ci=confint(fd,0.68);
    dp=abs(ci(1,:)-ci(2,:))/2;
    
    Naverage=str2num(get(handles.NumPoints,'String'));
    
    Ftrans_sub=F(index-Naverage:index);
    
    %     Ftrans=mean(Ftrans_sub)
    %
    %     Ftrans_error=std(Ftrans_sub)/sqrt(length(Ftrans_sub))
    %
    
    
    Ftrans=feval(fd,t_transition)
    pr=predint(fd,t_transition,0.68);
    
    Ftrans_error=(pr(2)-pr(1))/2;
    errorbar(t_transition,Ftrans,Ftrans_error,'bo','MarkerFaceColor','black');
    
    Nafter=5;
    index_after=index+dIndex+Nafter;
    
    Fafter=F(index_after:end);
    z_after=z(index_after:end);
    
    f=find(Fafter>Ftrans+std(Ftrans_sub));
    
    if ~isempty(f)
        
        
        Fafter(f(1):end)=[];
        z_after(f(1):end)=[];
        
        Ftrans_after_sub=Fafter(1:Naverage);
        Ftrans_after=mean(Ftrans_after_sub);
        
        
        Fbefore=F(1:index);
        z_before=z(1:index);
        
        f=find(Fbefore < Ftrans_after -std(Ftrans_after_sub));
        
        Fbefore(f)=[];
        z_before(f)=[];
        
        figure(2);
        subplot(3,1,3);
        hold on;
        plot(z_after,Fafter,'y',z_before,Fbefore,'g');
        
        display('Rupture Found')
        
        %% Find change in extension
        %Calculate change in tether length
        
        Fmin=min(Fbefore);
        Fmax=max(Fbefore);
        dF=0.5;
        Fsearch=Fmin:dF:Fmax;
        
        dz=[]
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
        sum(dz(:,1).*dz(:,8))/sum(dz(:,8))
        mean(dz(:,1))
        
    else
        display('No Post Rupture Data')
        dz=[];
        z_after=[];
        z_before=[];
        Fafter=[];
        Fbefore=[];
    end
    %%
    %     figure
    % plot(dz(:,1),Fsearch)
    % plot(Fsearch,dz(:,1))
    
    handles.Ftrans=Ftrans;
    handles.Ftrans_error=Ftrans_error;
    handles.FpS=FpS;
    
    
    FEtransition.zbefore=z_before;
    FEtransition.zafter=z_after;
    FEtransition.Fbefore=Fbefore;
    FEtransition.Fafter=Fafter;
    FEtransition.dz=dz;
    
    handles.FEtransition=FEtransition;
    
else
    display('Nothing Found')
end

