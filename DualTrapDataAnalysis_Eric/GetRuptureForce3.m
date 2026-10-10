F=handles.Ft;
z=handles.z;
t=handles.tsub;

figt=figure(3);

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
    
    subplot(2,1,1);hold off;
    plot(difft,dF);
    
    index=locs(f);
    
    t_transition=difft(index);
    hold on;
    plot(t_transition,pks(f),'ro');
    pkv=pks(f);
    
    for i=1:length(t_transition)
        text(t_transition(i),1.1*pkv(i),num2str(i),'FontWeight','bold');
    end
    
    tcut =0;
    if length(t_transition )>1
        t_transition;
        
        i=input('Enter time index! : ');
        
        if i>1
            tcut=t_transition(i-1);
            
            
        end
        
        t_transition=t_transition(i);
        
    end
    
    if tcut>0
        
        icut=find(t==tcut);
        t(1:icut)=[];
        F(1:icut)=[];
        z(1:icut)=[];
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
    
    figure(figt);
    subplot(2,1,2)
    hold off;
    plot(tfit,ffit);hold on; plot(fd);
    
    pr=predint(fd,[min(tfit):0.01:max(tfit)],0.68);
    hold on;
    
    %     plot([min(tfit):0.01:max(tfit)],pr(:,1),'g');
    %     plot([min(tfit):0.01:max(tfit)],pr(:,2),'g');
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
    %     errorbar(t_transition,Ftrans,Ftrans_error,'bo','MarkerFaceColor','black');
    
    Nafter=5;
    index_after=index+dIndex+Nafter;
    
    Fafter=F(index_after:end);
    z_after=z(index_after:end);
    
    useRef=get(handles.UseRefPull,'Value');
    
       if useRef==1
           z_after=handles.zref
           Fafter=handles.Fref
%             z_after=flipud(handles.zrp);
%             Fafter=flipud(handles.Frp);
            za=z_after;
            Fa=Fafter;
        end
    
    
    
    dFs=str2num(get(handles.dzParam,'String'));
    
    Fm=min(Fafter(1:10));
    Fmax=max(Fafter);
    f=find(Fafter==Fmax);
    Zmax=z_after(f);
    
    
    f=find(Fafter<=Fm+dFs & Fafter>=Fm & z_after<Zmax);
    
    
    
    if ~isempty(f)
        
        
        za=z_after(f);
        Fa=Fafter(f);
        
        
        Fbefore=F(1:index);
        z_before=z(1:index);
        
        Fm=max(Fbefore);
        
        
        f=find(Fbefore>=Fm-dFs);
        
        zb=z_before(f);
        Fb=Fbefore(f);
        
     
        
        figure(2);
        subplot(3,1,3);
        hold on;
        plot(za,Fa,'y',zb,Fb,'g');
        
        display('Rupture Found')
        
        %% Find change in extension
        %Calculate change in tether length
        
        
        
        %         [k_search,dz_search,zo_search,E_search] = Getdz(zb,Fb,za,Fa);
        %          dz_search=Getdz_Shift( Fbefore,Fafter,z_before,z_after,0.5 );
        if useRef==0
        [dz_search,F_out]=Getdz_Shift( Fb,Fa,zb,za,0.2 );
        
        else
            dz_search=dzF(Fbefore,z_before,Fafter,z_after,0.2,20);
        end
        
        zr=dz_search(:,3)./dz_search(:,2);
        
        display(['dL from manual shift: ',num2str(median(dz_search(:,1))),' nm']);
        
        lpg=str2num(get(handles.lpg,'String'));
        Lc1g=str2num(get(handles.Lc1g,'String'));
        z1=str2num(get(handles.zstart,'String'));
        z2=str2num(get(handles.zend,'String'));
        dLGuess=str2num(get(handles.dLG,'String'));
        %         [fd,err,Ft,x] = FitWLC( lpg,Lc1g,z_before,Fbefore,z1, z2,1,0.3 );
        
        [ fd,err,F1,z1,F2,z2 ] = FitDualWLCExtensible( lpg,Lc1g,z_before,Fbefore,z_after,Fafter,z1,z2,0.2,dLGuess );
        
        lp=fd(1);
        Lc1=fd(2);
        dLc1=err(2);
        dlp=err(1);
        %         [lp,dz,Lc,E] = Getdz_WLC(zb,Fb,za,Fa,median(dz_search),2700,0.5);
        %         dz=median(zr)*Lc1-Lc1;
        dz=fd(4);
        handles.dLFit=fd;
        handles.dLerr=err;
        display(['lp from WLC fit : ',num2str(lp),' nm ']);
        display(['Looped Contour from WLC fit : ',num2str(Lc1),' nm ']);
        display(['dL from Fitdata: ',num2str(dz),' nm']);
        display(['Error in dL fit: ',num2str(err(4)), 'nm']);
        %         zbp=1000:max(zb);
        %         Fbp=WLC_Mod(lp,Lc1,fd(3),zbp);
        
        %         zap=1000:0.95*(Lc);
        %         Fap=Fwlc(lp,Lc,zap);
        %
        
        %         plot(zbp,Fbp,'r',zap,Fap,'b');
        
        plot(z1,F1,'r',z2,F2,'r');
        
        
        
        
    else
        
        display('No Post Rupture Data')
        dFs=[];
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
    
    FEtransition.dz=[dz,median(dz_search(:,1)),lp,Lc1];
    
    FEtransition.lp=lp;
    FEtransition.dzshift=dz_search;
    
    handles.FEtransition=FEtransition;
    
    
else
    display('Nothing Found')
end


